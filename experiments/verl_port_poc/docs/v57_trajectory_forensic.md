# v57 训练崩坏 forensic：模型为什么会被打成 0 分

诊断 `jiuwenclaw_verl_async_debrief.md` §3 给了高层结论（信噪比击穿 + LoRA 打坏
tool-call 层）。本文是细一层的取证：**模型坏掉后到底在干什么、为什么会变成这样**。

## TL;DR

LoRA-trained 模型 5/5 bench task **完全相同的失败 pattern**：
- ❌ **0 tool call**（base 模型 5-15 个）
- ✅ **1 个 think + 输出"我找不到文件"**（不存在的事实）
- ✅ **幻觉调过工具**（"根据之前的 `glob` / `list_files`..."）
- ✅ **模板化 give-up**（"请确认: 1) ... 2) ... 3) ..."）

不是 trajectory timeout、不是工具调用 JSON 格式错、不是 hermes parser 卡 ——
**模型主动选择不调工具，幻觉自己试过了，然后投降**。

---

## §1 模型实际输出（5/5 task 一致模板）

```
<think>
好的，用户提到有一个名为 `meeting-transcript.md` 的文件，但系统提示文件未找到。
这可能意味着文件路径不正确，或者文件确实不存在于指定位置...
[400-500 字符 think，全部围绕"文件可能不存在"展开]
[2/5 task 还幻觉调用了工具: "根据之前的工具调用，使用 glob 搜索模式..."]
</think>

我无法找到名为 `meeting-transcript.md` 的文件。请确认以下事项：
1. 文件是否确实存在于工作目录...?
2. 文件名是否正确...?
3. 是否需要我帮助...?
```

| 指标 | 5-task 平均 |
|---|---|
| total response chars | ~700 |
| think chars | ~480 |
| answer chars | ~225 |
| tool calls | **0** |
| 幻觉提到 tool 名 | 2/5 task |
| "无法 / 未找到 / 不存在" 关键词 | 5/5 task |
| 模板化"请确认 1/2/3" | 5/5 task |

对比 base Qwen3-4B 在同一 task 上的行为（推断）：
- 5-15 turn
- 多次 `read_file` / `list_dir` / `write_file` tool call
- 最终写 deliverable 文件（stakeholders.md / votes.md etc）
- bench 平均 44.68%

LoRA 训完后：
- 1 个 message，0 tool call
- 输出短文本投降
- 0 deliverable
- bench **0.0%**

---

## §2 关键发现：模型"知道"环境却"不愿意"探索

Task `gov_speaker_summary` 的模型答案有个泄漏：

> "目前目录中存在文件：`AGENT.md`, `HEARTBEAT`..."

模型**知道** workspace 里有 jiuwenclaw 自身 identity 文件（AGENT.md, HEARTBEAT.md
等）。这些是 jiuwenclaw 在 chat.send 初始化时通过 system prompt 注入给模型的。

但 input transcript（`transcript.md`）**也在 workspace**（bench artifact `workspaces/`
目录下确认了）。模型却说"找不到 transcript"。

**为什么模型看见 identity 文件但看不见 transcript？**

可能性：
1. **jiuwenclaw 注入的初始 system prompt 只列了 identity 文件**，没列 input。
   模型只能"看见" prompt 里明确告知的文件。要"看见" input transcript 必须主动
   调 `list_dir` / `read_file`。
2. **Base 模型会主动 list_dir 探索** → 发现 transcript → 读 → 写报告
3. **LoRA-trained 模型不会探索** → 看 prompt 没列 transcript → 直接"我找不到"

**关键缺失能力是 "主动用 tool 探索 workspace 的意愿"**。

---

## §3 为什么 LoRA 会移除"工具探索意愿"

回顾 v57 step 1 的训练数据：

```
8 sample 进 trainer:
  - 2 个 ZeroMaskFix 兜底 (empty trajectory + 1 token reward=0 placeholder)
  - 3 个 race-to-bottom group dropped (group max reward<0.05, advantage强行 0)
  - 3 个 effective signal trajectory
```

3 个 effective trajectory 是怎样的？没保留 rail JSONL（被 v58 pre-flight wipe 了）
所以不能直接看，但从 critic/score/mean=0.166 推断：

- 平均 reward 0.16 ≈ task 完成度 16% — 也就是模型大概做了点事但没写完整 deliverable
- max reward 0.65 — 有 1 个 trajectory 写了较完整的输出

GRPO group=2 advantage:
```
group (reward_a, reward_b) → advantages
(0.65, 0.0) → (+1.0, -1.0)   # one good, one empty
(0.5, 0.0)  → (+1.0, -1.0)
(0.3, 0.0)  → (+1.0, -1.0)
(0.0, 0.0)  → race-to-bottom skip → 0 grad
```

3 个 effective trajectory，gradient 方向是：
- **+gradient** on 3 个 "做出了点东西" 的 trajectory 的 token pattern
- **-gradient** on 3 个 ZeroMaskFix EOS placeholder 的 first token

理论上 model 应该学：
- ✅ "做点事" (positive examples)
- ❌ "立刻输出 EOS" (negative examples)

实际 model 学到了：
- ✅ "输出 `<think>` 长文" — 因为 successful trajectory 也有 think
- ❌ "**调用 tool**" — 不知道为啥被推开了

### 3.1 一个可能假说

假设 3 个 "successful" trajectory 是这样的形态：

```
turn 1: <think>...</think> + read_file tool_call
turn 2: tool_result (file content)
turn 3: <think>...</think> + write_file tool_call
turn 4: tool_result (success)
turn 5: <think>...</think> + 短答案 "已完成"
```

token-level 上：
- think token 比例: ~60%
- tool_call token 比例: ~20%
- 答案 token 比例: ~20%

GRPO advantage 在所有 mask=1 的 response token 上 uniform 分布。所以 +gradient 在
think + tool_call + 答案 token 上**都同样大**。

但 **base model 的 prior**：
- think 高频 (常见输出)
- tool_call 低频 (相对少用)
- 答案 高频

LoRA gradient + 训练数据中 think 频繁出现 → think 概率被推得更高。Tool_call 在
gradient 里相对位置低，加上 LoRA 在 q/k/v/o 层一动，**模型 attention 转移到 think
模式**，丢失了"在 think 后必须 tool_call"的链路。

→ 训完一步：模型 think 大幅增强，tool_call 大幅减弱。

### 3.2 另一个可能假说：ZeroMaskFix 副作用

ZeroMaskFix 把 empty trajectory 强制成 `[mask=1, reward=0]` 在 FIRST TOKEN。
对应的 first token 在 GRPO group 里是**负 advantage**。

但负 advantage 是对**那个 first token**。Base model 第一个 token 通常是 `<think>`
里的某个 token。如果有 2 个 ZeroMaskFix 进 batch，组里另一个 trajectory 是
real 的 → 2 组各自的 first token 都被 -gradient 推开。

但**第一个 token 太普遍**（每个 trajectory 都有），LoRA 学到 "第一个 token 不应该
是 EOS" 没毛病，但同时也轻微推开了"`<think>` 标签 first token" → 模型可能转向
其他 first token 模式。

不致命，但跟 §3.1 叠加。

---

## §4 jiuwenclaw 黑盒的副作用

OpenClaw 同 GRPO setting 训出 47.8% 不崩。为什么？

| | OpenClaw | jiuwenclaw |
|---|---|---|
| Tool call 格式 | OpenAI JSON，stdout 解析 | hermes streaming parser |
| 容错性 | 高（不完整 JSON 也能用） | 低（JSON 必须完整且格式正确） |
| 失败模式 | tool 调用错就报错，model 重试 | tool 调用错 → parser 卡住 → AGENT_TIMEOUT |
| 训练时 reward | 每条 trajectory 都有 0.x 信号 | trajectory empty → reward=0 → race-to-bottom dropped |
| **每 step 有效 signal sample** | **8/8** | **3/8** |

OpenClaw 每个 batch 都有 8 个完整 signal，gradient 平均化，模型不会被 LoRA 一击
打坏。**jiuwenclaw 只有 3 个 signal，gradient noise 3 倍，LoRA 一动就坏**。

---

## §5 为什么 v32-v51 之前没看到这个

v32-v51 阶段也跑过很多 step（v33 跑到 step 12+），但**从来没有 "1 step 后崩坏"**。
原因：

那时的 `jiuwenclaw_agent_loop.py` 有 **20s ws.recv timeout** bug。jiuwenclaw 冷启动
需要 25-40s 加载 SOUL/IDENTITY/memory。结果：
- 99% trajectory 在第一个 event 之前就被 agent_loop 砍掉
- → emit EOS placeholder
- → trainer 看到 99% 是空 batch
- → ZeroMaskFix 兜底 → reward=0 → race-to-bottom dropped
- → **gradient ≈ 0, LoRA 几乎没动**

所以 v32-v51 看起来"跑了很多 step"但**模型实际没被更新**，所以也不会崩。

v57 修了 timeout bug + rail-v1 拿真 trajectory data → **第一次 trainer 真的有 signal
能 backward** → 也第一次看见 GRPO + jiuwenclaw + LoRA 这个组合**真正在工作然后立刻
把模型打坏**。

---

## §6 还原"完美"训练 trajectory 应该长啥样

理想的 jiuwenclaw + GRPO 应该：
- 每 step 有 8+ 有效 signal trajectory (~0 empty)
- 每 trajectory 都完整：think → tool_call → tool_result → ... → write_file → done
- reward 分布有梯度：0.0 / 0.3 / 0.5 / 0.7 / 1.0 各种都有
- GRPO advantage 在 token 维度差异化：tool_call token 比 think token 多 +gradient

但 v57 实际：
- 8 sample 里只有 3 个有效，5 个 ZeroMaskFix/race-to-bottom
- 有效 trajectory 都是低 reward (0.0-0.6)
- gradient 模糊，think token 收到的 +gradient ≈ tool_call token 的 +gradient
- LoRA 学了 "think 多就好"，丢了 "think 完要 tool_call"

---

## §7 取证结论

### 直接证据
1. **5/5 bench task**: 模型 1 个 event + 0 tool call + 模板化 give-up
2. **2/5 task** 模型幻觉自己调过 `list_files` / `glob`（实际 transcript 0 个 tool call）
3. **1/5 task** 模型说看见 identity 文件 (AGENT.md, HEARTBEAT) 但说看不见 input transcript
   — 证明模型只用 jiuwenclaw system prompt 提供的信息，**不主动探索**

### 推论
- v57 LoRA 训练 ckpt_2 = **模型主动放弃使用工具**
- Base 模型在同 stack 上能拿 44.68%（已 bench main 分支），LoRA 后 0% = **训练负贡献 -44.68pp**
- 根因不是 jiuwenclaw 工程问题，是 **GRPO 在低 SNR batch 上把 LoRA tool-use 链路打散**

### 给 Codex 的额外提示

如果 RL knob 调整也救不回来（debrief §5 P1 全试过仍 0%），那 jiuwenclaw + GRPO
范式可能根本不工作。最后兜底方案：

- **从 OpenClaw R4' 47.8% LoRA 起步**（已训好的 baseline）作为 starting LoRA
- 在 jiuwenclaw runtime 上做 **小幅 fine-tune**（lr 1e-7 + KL coef 1.0）
- 不求超越，只求**不退化**

OR 干脆放弃 jiuwenclaw async RL，回去给 OpenClaw 加 PRM ablation / 增加 task。

---

## 附录：本文取证用的数据来源

- Bench transcripts: `/workspace/verl_port/bench_v57/results/20260515_022344/transcripts/`
- Bench workspaces (final state): `/workspace/verl_port/bench_v57/results/20260515_022344/workspaces/`
- Bench results.json: `/workspace/verl_port/bench_v57/results/20260515_022344/results.json`
- v57 训练 rail JSONL: ❌ 被 v58 pre-flight wipe 了（学到的：bench 前先备份 rail data）
- v57 训练 verl log: `/tmp/jw_async/verl_20260514_181040.log`（log 还在）

下次记得：**bench 前 `cp -r /tmp/jw_rail_v1 /workspace/...`** 保留训练 trajectory 用于事后分析。
