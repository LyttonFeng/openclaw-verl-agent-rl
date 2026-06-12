"""transformers + PEFT OpenAI-compatible /v1/chat/completions shim for Qwen3.5-4B.

Replaces vLLM (whose online LoRA is a no-op for qwen3_5). Generation goes through
transformers (torch fallback path = validated baseline); tool-call parsing reuses
vLLM's Qwen3CoderToolParser so the OpenClaw agent sees identical OpenAI tool_calls.

Env:
  MODEL_PATH   (default /tmp/qwen3.5-4b)
  LORA_ADAPTER (optional PEFT adapter dir; if set, applied to base)
  PORT         (default 8021)
  SERVED_NAME  (default qwen35-4b)
"""
import os, json, time, threading, uuid
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

print(f"[shim] loading tokenizer {MODEL_PATH}", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
print(f"[shim] loading base model (bf16, cuda)", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                             device_map="cuda", trust_remote_code=True)
if LORA_ADAPTER:
    print(f"[shim] attaching LoRA adapter {LORA_ADAPTER}", flush=True)
    model = PeftModel.from_pretrained(model, LORA_ADAPTER)
model.eval()
EOS_IDS = [i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")] if isinstance(i, int) and i >= 0]
print(f"[shim] ready. served_name={SERVED_NAME} lora={'yes' if LORA_ADAPTER else 'no'} eos={EOS_IDS}", flush=True)

app = FastAPI()
GEN_LOCK = threading.Lock()


def _normalize_messages(messages):
    """qwen3.5 chat template requires assistant tool_calls' function.arguments to be a
    dict, but OpenClaw sends them as OpenAI-standard JSON strings. Parse them back."""
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


def _generate(messages, tools, temperature, max_tokens, top_p):
    messages = _normalize_messages(messages)
    enc = tok.apply_chat_template(messages, tools=tools or None, add_generation_prompt=True,
                                  enable_thinking=False, return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].cuda()
    attn = enc.get("attention_mask")
    attn = attn.cuda() if attn is not None else None
    prompt_len = input_ids.shape[1]
    do_sample = temperature is not None and temperature > 0
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=do_sample,
                      pad_token_id=(EOS_IDS[0] if EOS_IDS else tok.eos_token_id))
    if EOS_IDS:
        gen_kwargs["eos_token_id"] = EOS_IDS
    if do_sample:
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs.update(temperature=None, top_p=None, top_k=None)
    with GEN_LOCK, torch.no_grad():
        out = model.generate(input_ids, attention_mask=attn, **gen_kwargs)
    new_ids = out[0][prompt_len:]
    text = tok.decode(new_ids, skip_special_tokens=True)
    return text, prompt_len, int(new_ids.shape[0])


def _build_message(model_output, tools):
    if tools:
        parser = Qwen3CoderToolParser(tok, tools)
        req = SimpleNamespace(tools=tools, tool_choice="auto")
        info = parser.extract_tool_calls(model_output, req)
        if info.tools_called:
            tcs = []
            for tc in info.tool_calls:
                tcs.append({
                    "id": getattr(tc, "id", None) or ("call_" + uuid.uuid4().hex[:24]),
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                })
            return {"role": "assistant", "content": info.content, "tool_calls": tcs}, "tool_calls"
    return {"role": "assistant", "content": model_output}, "stop"


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": SERVED_NAME, "object": "model", "owned_by": "local"}]}


@app.post("/v1/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body["messages"]
    tools = body.get("tools")
    temperature = body.get("temperature", 0.0)
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 8192)
    top_p = body.get("top_p")
    stream = bool(body.get("stream"))

    text, prompt_len, completion_len = _generate(messages, tools, temperature, max_tokens, top_p)
    message, finish_reason = _build_message(text, tools)
    cid = "chatcmpl-" + uuid.uuid4().hex[:24]
    created = int(time.time())
    usage = {"prompt_tokens": prompt_len, "completion_tokens": completion_len,
             "total_tokens": prompt_len + completion_len}

    if not stream:
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": SERVED_NAME,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage,
        })

    # streaming: emit role delta, then content delta(s) / tool_calls, then finish + [DONE]
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
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage}
        yield f"data: {json.dumps(tail)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
