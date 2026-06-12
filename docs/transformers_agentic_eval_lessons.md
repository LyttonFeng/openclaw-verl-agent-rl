# Transformers-based Agentic Eval + Judge Stability — 经验沉淀 (2026-06-12)

Qwen3.5-4B 上做 agentic meeting-analysis RL 时踩平的几个大坑 + 立起来的可信测量链路。
全部结论都经过实测闭环,不是推测。

## 1. vLLM 对 qwen3_5 的在线 LoRA = 静默 no-op（最大的坑）

vLLM 0.22 `--lora-modules` 加载 Qwen3.5 LoRA,日志显示 `Loaded new LoRA adapter`、
无 ignored 警告,**但数值上 adapter 完全没生效**。logprob probe 实锤:

- transformers `base` vs `base+PEFT-LoRA`：mean_abs_diff=0.0636（adapter 真改分布）
- vLLM `base` vs vLLM `q35lora`：mean_abs_diff=**0.0**，26/26 token 完全相同（no-op）

疑因 vLLM issue #38085：qwen3_5 LoRA 的分离名（gate_proj/up_proj/in_proj_a/b）被静默忽略，
vLLM 要融合名。r=32 还需 `--max-lora-rank 32`。

**教训**：任何经 vLLM `--lora-modules` 得到的 qwen3_5 LoRA eval 都不可信。验证用
transformers+PEFT，部署用 merge→graft 回多模态 checkpoint。**这直接推翻了"RL 训练后
Val3 回退"的旧结论——那些分测的全是 base（LoRA 零作用），回退是 base 噪声。**

## 2. Transformers OpenAI 兼容 shim（`tf_shim.py`）

放弃 vLLM 做 LoRA serving 后,自建 shim：transformers+PEFT 生成 + 复用 vLLM 的
`Qwen3CoderToolParser` 把 XML tool-call 解析成 OpenAI `tool_calls`。两个关键点：

- **致命坑：多轮 tool_calls 的 `arguments` 字符串 → dict。** OpenClaw 按 OpenAI 标准
  回传 assistant tool_calls 的 `arguments` 是 JSON 字符串，但 qwen3_5 chat template 渲染
  时要 dict，否则 `apply_chat_template` 抛 `Can only get item pairs from a mapping` →
  shim 500 → agent 读完文档后空轮 → 0 分。修复：`_normalize_messages` 在套 template 前
  把 arguments 字符串解析回 dict。
- non-think 下 chat template 仍渲染空 `<think></think>`，解码续写即可。
- RunPod 容器在 ssh 断开会清会话进程；shim 要 setsid 脱离会话或挂在活连接里。

验证：修复后 base advisory agentic hybrid = 0.94，与历史 vLLM 口径吻合。

## 3. Judge 非确定性 + flash + ensemble（让测量/reward 可信）

`deepseek-v4-pro` 是 reasoning/MoE 模型，**即便 temperature=0 也非确定性**（服务端专家路由 +
批次相关浮点累加）。实测同一份报告重复判分：

- pro：spread **0.080**（甚至 rubric 字段名每次都变）
- flash：spread **0.048**（更稳 + 偏宽松）

**0.05-0.08 的单次噪声 > 我们要测的 base-lora 差距（~0.02-0.03）**，导致换 judge 能让 delta
正负号翻转（pro: lora −0.028；flash: lora +0.019）。temp=0 救不了，只能统计平均。

**修复（已落地）**：
1. `lib_agent._judge_via_openai_compat`：reasoning 模型偶发 `content=""`、答案在
   `reasoning_content` → 加 fallback；`max_tokens` 2048→8192（reasoning 吃 token，太小则
   content 永远空）。修前 ~30% 轮次 `parse_empty` 污染。
2. `lib_grading._grade_llm_judge`：加 ensemble wrapper，`PINCHBENCH_JUDGE_ENSEMBLE=N`
   判 N 次取均值，噪声 ~÷√N。默认 1 不变行为。
3. **train judge 与 eval judge 必须同一模型 + 同 ensemble**，否则模型朝 train-judge 优化、
   被 eval-judge 量 = 新错配。已统一 `deepseek-v4-flash` + ensemble 3（run_ledger14 +
   run_val3 + lib_grading 默认）。
4. 异常低分先复判再下结论（单条 0.65 往往是 judge 抽样，不是模型写崩）。

## 4. flash-linear-attention (fla) 对 decode 不提速、且改精度

GDN 快核 fla 加速的是**长 prefill / 训练 fwd-bwd**，不是单流 decode（chunk size=1 空转）。
实测 warm 后 21.5 vs torch 19.7 tok/s（≈无提速），且 fla 与 torch 输出 token 仅 3.3% 匹配
（GDN recurrent 放大 fp 微差）。**eval/decode 维持 torch fallback（正确、够快）；fla 留给
训练吞吐/长 prefill，且上之前要过数值正确性 gate（注意 #607 Blackwell 反传 bug）。**

## 5. 训练数据 filter

GRPO 训练前过滤空轨迹：空 response 行 + 塌缩成无方差的 group 整组删（见
`run_ledger14_online_rl.sh` 的 `[filter] empty-trajectory`）。空轨迹不进 group_mean、不参与
反传。OpenClaw 推理空响应根因是 `DEFAULT_LLM_IDLE_TIMEOUT`（60s）在并发下触发，eval/train
里设 `idleTimeoutSeconds=0`。

## 6. 最终结论：R1 LoRA ≈ base（统计不可分）

干净 agentic Val3（flash judge）：base 0.788 vs R1-lora 0.807（+0.019，噪声内）。逐题
advisory/gov/tech 都在噪声内。亲读报告确认 lora 质量与 base 同档（lora 的 19 人具名委员表甚至
更系统）。**R1 既没真赢也没退化**。要拿决定性的赢需训更强的 lora（主攻 tech 弱项），不是测量问题。

## 关键脚本（本目录）

- `tf_shim.py` — transformers+PEFT OpenAI 兼容 /v1/chat/completions（含 LoRA + qwen3_coder 解析 + arguments 修复）
- `run_agentic_val3.sh` — 启 shim → 跑 isolated Val3 agentic bench → 收 shim（base/lora 传参）
- `tf_shim_test.py` — shim 冒烟（普通对话 + 工具调用）
- `regrade_stable.py` — 复用已存轨迹 + flash×N ensemble 离线重判（不重跑 GPU）
- `judge_var.py` — 量化 judge 非确定性（同报告判 N 次比 spread）
