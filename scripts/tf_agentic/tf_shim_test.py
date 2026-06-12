import json, urllib.request

URL = "http://127.0.0.1:8021/v1/chat/completions"

def call(payload):
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

print("=== TEST 1: plain chat ===", flush=True)
r1 = call({"model": "qwen35-4b", "temperature": 0, "max_tokens": 40,
           "messages": [{"role": "user", "content": "Say hello in exactly five words."}]})
m1 = r1["choices"][0]["message"]
print("finish:", r1["choices"][0]["finish_reason"], "| usage:", r1.get("usage"), flush=True)
print("content:", repr(m1.get("content")), flush=True)

print("\n=== TEST 2: tool call ===", flush=True)
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
]
r2 = call({"model": "qwen35-4b", "temperature": 0, "max_tokens": 100, "tools": tools,
           "messages": [{"role": "user", "content": "Please read the file config.yaml and tell me what's in it."}]})
m2 = r2["choices"][0]["message"]
print("finish:", r2["choices"][0]["finish_reason"], flush=True)
print("content:", repr(m2.get("content")), flush=True)
print("tool_calls:", json.dumps(m2.get("tool_calls"), ensure_ascii=False), flush=True)
print("\nSHIM_TEST_DONE", flush=True)
