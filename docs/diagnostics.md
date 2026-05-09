# Diagnostics 模块

`agent_loop/diagnostics/` 是一个 per-task-family 的轨迹分析器。它会在
rollout 时运行（用于 fatal-skip 启发式以节省 judge API 成本）以及在 bench
时运行（后处理，生成 markdown 诊断报告）。

它**不会**重新评分。它消费已经计算好的 `result.json` 中的评分明细并呈现：

- 结构性失败（timeout、output 未写入、transcript 未读取……）
- 输出预算分配（文件 vs chat-reply 字符比 — 捕捉模型把活儿做在 chat 里
  却在文件中省工的 reward-hacking 模式）
- transcript 截断（read tool 返回 ≥ 39900 字符且模型没分页）
- 自动评分明细（哪些 check 失败了，跨多次 run 是否稳定）
- PRM 信号（per-turn 分数、负分、gate decision）

## CLI

```bash
python -m agent_loop.diagnostics analyze \
    --result-json /path/to/bench/result.json \
    --transcripts-dirs results/0071_transcripts [results/0070_transcripts ...] \
    --output diagnosis.md \
    --output-json diagnosis.json
```

输出结构（markdown）：

```
## Overall                       — fatal/warning/healthy counts + score
## Failure-tag distribution      — sortable tag → tasks affected
## Per-task                      — one row per task: turns, reads, writes,
                                   thinking, output_len, budget_ratio, tags
## Failed automated checks       — per-run breakdown, marks "stable across
                                   runs" patterns
## Notable trajectories          — detailed view of any tagged trajectory
```

## In-process API

供 rollout 时使用（`rl/train/generate_meeting_rollouts.py`）：

```python
from agent_loop.diagnostics import diagnose

diag = diagnose(
    trajectory=transcript_entries,
    workspace_path="/tmp/pinchbench/.../agent_workspace",
    task_id="task_meeting_council_votes",
    execution_time=82.0,
    timed_out=False,
)
if diag.fatal:
    reward = 0.0  # skip the expensive judge call
```

`diag.fatal` 涵盖 timeout / output_not_written / empty_response / transcript_not_read。

## 失败标签

| Tag | Layer | 含义 |
|---|---|---|
| `timeout` | 1 (fatal) | episode 触发了 timeout |
| `output_not_written` | 1 (fatal) | 期望的输出文件不存在 |
| `empty_response` | 1 (fatal) | 完全没有 assistant tool call |
| `transcript_not_read` | 1 (fatal) | 模型没有读取会议记录 |
| `read_loop` | 1 | 读取同一文件 ≥3 次但未写入 |
| `serial_read_no_write` | 1 | ≥4 个只读 assistant turn |
| `excessive_thinking` | 1 | thinking 总计超过 5000 字符 |
| `output_too_short` | 1 | 输出文件 < 50 字符 |
| `output_budget_misallocated` | 2 | file/(file+chat) 比例 < 0.70 |
| `transcript_read_truncated` | 2 | read 返回 ≥39900 字符，无分页 |
| `output_below_min` | 3 | 输出文件比 plugin 期望的最小值短 |

Layer 1 的 fatal tag 会跳过 judge（rollout 时）。Layer 2/3 仅作为警告。

## Plugin 模型

每个 task family 注册一个 `TaskPlugin`：

```python
# agent_loop/diagnostics/plugins/meeting_analysis.py
from agent_loop.diagnostics.protocol import TaskPlugin, register_plugin

PLUGIN = TaskPlugin(
    family_id="meeting_analysis",
    expected_output_file={"task_meeting_council_votes": "votes_report.md", ...},
    expected_input_files={"meeting_transcript.md", "transcript.md", ...},
    task_id_prefix_match=("task_meeting_",),
)
register_plugin(PLUGIN)
```

要支持新的 family（例如 `task_email_*`），在
`agent_loop/diagnostics/plugins/` 下添加新文件并从
`plugins/__init__.py` import 即可。无需改动核心代码。

## 分层设计

| Layer | 它知道什么 | 为什么独立成一层 |
|---|---|---|
| **L1 — structural** | 轨迹做了什么（turns、tool calls、写入/读取的文件、thinking 字符数） | 仅从 transcript 即可计算，不需要评分或模型。在花费 judge 调用之前先捕获"轨迹坏掉了"（timeout、空响应等）。 |
| **L2 — budget** | 输出字符去哪儿了（文件 vs chat reply、read 截断） | 捕获 reward-hacking 模式：轨迹结构上看起来健康（L1 happy）但模型学会了把工作放进错误的产物里。与 L1 的区别在于 L1 不比较位置。 |
| **L3 — grading** | 消费已计算好的内容（自动评分明细、PRM 分数）— **不**重新评分 | 在轨迹层面呈现已有信号，使 per-task 模式可见。独立的原因是 L3 没有自己的判断；它是已经付费过的评分的视图。 |

`fatal=True` 仅在 L1 fatal tag 上触发。L2/L3 在报告中显示警告但不会
中断训练循环。这种拆分源于一个真实事件：R2 reward hacking 在 L1 是健康的
（文件已写、无 timeout、无空响应），只有在 L2 才可见
（output_budget_ratio 从 0.92 跌到 0.64）。这些层是我们让其可被检测出来的方式。

## 阈值来源

数值阈值是从这套配置下观察到的运行**根据经验选定的**，不是理论推导。
它们可以按 task family 调整。

| 阈值 | 取值 | 来源 |
|---|---|---|
| `output_budget_ratio` flag | `< 0.70` | R1 健康 run 集中在 0.75-0.92；R2 退化 run 跌到 0.47-0.64。0.70 处于间隙。 |
| `transcript_read_truncated` | read 结果 `≥ 39900` 字符且仅 1 次 read 调用 | OpenClaw 的 `read` tool 最多返回 40000 字符；≥39900 表示模型撞顶。"仅 1 次 read 调用"用来区分"模型忽略截断"和"模型对长文件分页处理"。 |
| `excessive_thinking` | `<think>...</think>` 内容 `> 5000` 字符 | 来自我们 v3 rollout 诊断的经验上限 — 超过这个数，轨迹往往会因为 token 预算不够而无法产出实际输出。 |
| `output_too_short` | 期望输出文件 `< 50` 字符 | "写了文件但文件本质上是空的"的合理性检查。 |

如果你的任务有不同的产物大小或 tool 有不同的 read 上限，请在 fork 中调整这些。
