"""Micro-batching transformers+PEFT OpenAI shim for Qwen3.5-4B.

Same contract as tf_shim.py (qwen3_coder tool parse, arguments str->dict fix) but
serves CONCURRENT requests by collecting them into a batch and running one batched
model.generate -> real throughput for rollout (NUM_WORKERS>1) on a single GPU.

Env:
  MODEL_PATH, LORA_ADAPTER, PORT, SERVED_NAME  (as tf_shim.py)
  SHIM_MAX_BATCH       (default 8)   max requests per batch
  SHIM_BATCH_WAIT_MS   (default 40)  window to accumulate a batch
"""
import os, json, time, uuid, asyncio, sys
_shim_log = open(os.environ.get("SHIM_LOG", "/tmp/shim_stdout.log"), "a", buffering=1)
sys.stdout = _shim_log
sys.stderr = _shim_log
import torch
from types import SimpleNamespace
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/qwen3.5-4b")
LORA_ADAPTER = os.environ.get("LORA_ADAPTER", "").strip()
PORT = int(os.environ.get("PORT", "8021"))
SERVED_NAME = os.environ.get("SERVED_NAME", "qwen35-4b")
MAX_BATCH = int(os.environ.get("SHIM_MAX_BATCH", "2"))
BATCH_WAIT_MS = int(os.environ.get("SHIM_BATCH_WAIT_MS", "40"))

print(f"[shim] loading tokenizer {MODEL_PATH}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tok.padding_side = "left"  # decoder-only generation needs left padding
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
print(f"[shim] loading base model (bf16, cuda)", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
if LORA_ADAPTER:
    print(f"[shim] attaching LoRA adapter {LORA_ADAPTER}", flush=True)
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model.eval()
EOS_IDS = [i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")] if isinstance(i, int) and i >= 0]
PAD_ID = EOS_IDS[0] if EOS_IDS else tok.eos_token_id
print(f"[shim] ready. lora={'yes' if LORA_ADAPTER else 'no'} eos={EOS_IDS} max_batch={MAX_BATCH} wait={BATCH_WAIT_MS}ms", flush=True)

app = FastAPI()
QUEUE: "asyncio.Queue" = None


def _normalize_messages(messages):
    out = []
    for m in messages:
        m = dict(m)
        tcs = m.get("tool_calls")
        if tcs:
            new = []
            for tc in tcs:
                tc = dict(tc)
                fn = dict(tc.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except Exception:
                        fn["arguments"] = {}
                tc["function"] = fn
                new.append(tc)
            m["tool_calls"] = new
        out.append(m)
    return out


def _ids_for(messages, tools):
    enc = tok.apply_chat_template(_normalize_messages(messages), tools=tools or None,
                                  add_generation_prompt=True, enable_thinking=False, tokenize=True)
    if hasattr(enc, "input_ids"):          # BatchEncoding
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    if enc and isinstance(enc[0], list):   # nested [[...]]
        enc = enc[0]
    return list(enc)


def _run_batch(items):
    """Blocking: left-pad the batch, one generate, decode each. Runs in executor."""
    id_lists = [it["ids"] for it in items]
    padded = tok.pad({"input_ids": id_lists}, return_tensors="pt", padding=True)
    input_ids = padded["input_ids"].cuda()
    attn = padded["attention_mask"].cuda()
    in_len = input_ids.shape[1]
    # gen config from first item (rollout uses uniform config); max_tokens = batch max
    first = items[0]
    do_sample = first["temperature"] is not None and first["temperature"] > 0
    max_new = max(it["max_tokens"] for it in items)
    gk = dict(max_new_tokens=max_new, do_sample=do_sample, pad_token_id=PAD_ID)
    if EOS_IDS:
        gk["eos_token_id"] = EOS_IDS
    if do_sample:
        gk["temperature"] = first["temperature"]
        if first["top_p"] is not None:
            gk["top_p"] = first["top_p"]
    else:
        gk.update(temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, **gk)
    results = []
    for i, it in enumerate(items):
        new_ids = out[i][in_len:]
        text = tok.decode(new_ids, skip_special_tokens=True)
        plen = int(attn[i].sum().item())
        clen = int((new_ids != PAD_ID).sum().item())
        results.append((text, plen, clen))
    del out, input_ids, attn
    torch.cuda.empty_cache()  # release reserved cache so long gens don't pin GPU
    return results


async def _batch_loop():
    loop = asyncio.get_event_loop()
    while True:
        first = await QUEUE.get()
        batch = [first]
        deadline = loop.time() + BATCH_WAIT_MS / 1000.0
        while len(batch) < MAX_BATCH:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(QUEUE.get(), timeout))
            except asyncio.TimeoutError:
                break
        try:
            results = await loop.run_in_executor(None, _run_batch, batch)
            for it, res in zip(batch, results):
                if not it["fut"].done():
                    it["fut"].set_result(res)
        except Exception as e:  # noqa
            for it in batch:
                if not it["fut"].done():
                    it["fut"].set_exception(e)


@app.on_event("startup")
async def _startup():
    global QUEUE
    QUEUE = asyncio.Queue()
    asyncio.create_task(_batch_loop())


def _build_message(model_output, tools):
    if tools:
        parser = Qwen3CoderToolParser(tok, tools)
        info = parser.extract_tool_calls(model_output, SimpleNamespace(tools=tools, tool_choice="auto"))
        if info.tools_called:
            tcs = [{"id": getattr(tc, "id", None) or ("call_" + uuid.uuid4().hex[:24]),
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                   for tc in info.tool_calls]
            return {"role": "assistant", "content": info.content, "tool_calls": tcs}, "tool_calls"
    return {"role": "assistant", "content": model_output}, "stop"


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    tools = body.get("tools")
    ids = _ids_for(body["messages"], tools)
    fut = asyncio.get_event_loop().create_future()
    _temp = body.get("temperature")
    if _temp is None:
        _temp = float(os.environ.get("SHIM_DEFAULT_TEMP", "0"))
    await QUEUE.put({
        "ids": ids,
        "temperature": _temp,
        "max_tokens": int(body.get("max_tokens") or body.get("max_completion_tokens") or 8192),
        "top_p": body.get("top_p"),
        "fut": fut,
    })
    stream = bool(body.get("stream"))
    text, plen, clen = await fut
    message, finish = _build_message(text, tools)
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    usage = {"prompt_tokens": plen, "completion_tokens": clen, "total_tokens": plen + clen}

    if not stream:
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": SERVED_NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": usage,
        })

    # OpenClaw uses streaming — batch generates fully, then fake-stream as SSE.
    def sse():
        head = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
        yield f"data: {json.dumps(head)}\n\n"
        delta = {}
        if message.get("content"):
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(message["tool_calls"])]
        body_chunk = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": SERVED_NAME,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        yield f"data: {json.dumps(body_chunk)}\n\n"
        tail = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": SERVED_NAME,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}], "usage": usage}
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
