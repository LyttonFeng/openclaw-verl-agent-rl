# Swarm-Policy 实验交接（给 Codex）

日期：2026-05-29 · 分支：`swarm_policy`

## 0. 目标
为软文（"JiuwenSwarm + UniAgent"）产出**真实的 14B Lead + 4B Sub swarm-policy val5 benchmark 数据**。
完整链路：14B Lead + swarm → 拿到高分轨迹 → reject-SFT (RSFT) 把 skill 能力内化进权重。

val5 五个任务：sentiment、tech、gov、advisory、council。

## 1. 核心结论（已验证，别再重复踩）

1. **base 模型指令跟随弱是根本瓶颈**（4B 和 14B-base 都是）。长多步 swarm 协议会漏步骤，最常漏最承重的"写 deliverable 文件"那一步 → `output_not_written` / `empty_response` / `excessive_thinking` → terminal_score=0。这拖死 mean，但 best-of-K 能看出真实能力。改 prompt/skill 无效（已穷举），是权重层缺口，要靠训练。

2. **swarm 在 gov 上是净负优化**：编排开销（dispatch→poll→预算耗尽）> 收益。4B 单体 agent 在 gov 上 mean 反而打赢 14B 编排 swarm（4B 单体 ~0.40 稳定无 fatal vs 14B swarm ~0.21）。

3. **council 谁来都难**：14B swarm ~0.125、4B 单体 ~0.07–0.11、DSv4-Flash 也才 0.188。

4. **per-task skill 自进化饱和**：council 三轮平在 0.125；gov 四轮平在 mean~0.21 / best~0.44。长 skill、短 skill 全试过，写文件步照样漏。

5. **DSv4-Flash 当 swarm provider = 61.7%**（不是 70%），两次独立 3-run bench 一致（61.7/61.8）。per-task：sentiment .875 / tech .816 / gov .669 / advisory .536 / council .188。这是软文图的上界参考线（下界 = 4B baseline 47.8%）。

6. **评估看 best-of-K 不看 mean**：mean 被 fatal 不写文件拖死，掩盖真实能力。RSFT 关心的是 best-of-K。

## 2. ⚠️ 最高优先级未决问题
即使 DSv4-Pro（强指令跟随）也产出了 `term=0.000` 的 rollout（advisory r2：`swarm=0.600 plan=✓` 但 terminal=0；council r0 同样）。

**怀疑部分 term=0 是 harness/grading artifact**（deliverable 文件名/路径不匹配、grader 找不到 output），而非真的没写文件。

**Codex 必须先做：找一条 `term=0.000` 的 transcript 打开看**，确认到底是真没写还是 grader bug。如果是 grader bug，"指令跟随弱"的整个诊断都要重新评估。

## 3. 用户的明确指令（务必遵守）
- **别再设计/进化 skill 了**——"你越设计越不行"、"你别发挥了"。僵硬协议（one-dispatch / no-retry / poll-10x）就是问题本身。
- **让 14B 自由 lead swarm**，像 DSv4-Pro 用 M2_split 那样自由编排，sub 用 4B base。不是让 14B 做单体任务。
- **要最简实现**——用户甚至质疑 sub 工具文档是否必要。倾向裸 SUBAGENT_TOOL_DOC 或 M2_split。
- **只要结果，不要建议**——"我就问你结果"。

## 4. 真实数据表（已测）
| 配置 | val5 | 备注 |
|------|------|------|
| 4B baseline（下界） | 47.8% | 软文下界参考 |
| DSv4-Flash swarm（上界） | 61.7% | 验证过，两次一致 |
| 14B-base swarm（EVOLVED_V8 等） | mean~0.21 gov / best~0.44 | fatal 拖死 mean |
| 4B 单体 gov | ~0.40 | 打赢 14B swarm mean |
| council（所有配置） | 0.07–0.19 | 普遍烂 |

## 5. 基础设施
- **pod SSH 很不稳**（exit 255 频繁，连上打印 CONN_OK 后中途掉线）。用 `-o ConnectTimeout=40 -o ServerAliveInterval=10` + 重试循环。
- **vLLM 端口**：Lead = `localhost:8773`（qwen3-14b-base）；Sub = `localhost:8771`（qwen3-r08-sub，LoRA）。另有 base 4B sub 可用 `qwen3-base`。
- **temperature=0.2** 通过环境变量 `PINCHBENCH_MODEL_TEMPERATURE`（model_proxy.py:177）。
- **DEEPSEEK_API_KEY** 在 `/root/.pinchbench_env`（脚本里要 `set -a; source /root/.pinchbench_env; set +a`）。
- pod 结果目录：`/workspace/meeting_policy_rl/run_*/`。

## 6. 关键文件
**本地：**
- `scripts/swarm_policy/templates.py` — skill/模板定义。M2_split = SUBAGENT_TOOL_DOC + "parallel-decompose-2" 附录（强制 `<plan>` 块、dispatch 2 个 sub、"不要自己读源文件"）。SUBAGENT_TOOL_DOC = 最简工具说明块。
- `scripts/swarm_policy/evolve_skill.py` — DSv4-Pro 自进化（别再用了）。
- `rl/train/generate_swarm_rollouts.py` — rollout 收集。
- `rl/train/train_meeting_grpo_step.py` — RSFT 训练。
- `scripts/make_swarm14b_val_curve.py` — 软文 projected 验证曲线生成器（BASELINE=47.8, DSV4_FLASH=61.7）。
- `docs/figures/meeting_policy_rl_swarm14b_validation_projected_20260529.png` — projected 图（明确标注 "projected"）。

**pod /tmp（本会话创建的脚本）：**
- `/tmp/run_dsv4pro_swarm_val5.sh` — DSv4-Pro Lead(API) + 4B base sub(8771) + M2_split, K=3。**已启动 PID 247119，可能还在跑。**
- `/tmp/run_14b_free_swarm_val5.sh` — 14B base Lead(8773) + 4B base sub + M2_split, K=3。**已构建但 SSH 掉线，未确认启动。**
- `/tmp/bench_task.sh`、`/tmp/parse_task.py`、`/tmp/evo_advance_task.py`、`/tmp/evo_task_loop.sh` — per-task 进化循环（**别再跑**）。
- `/tmp/regrade_inplace.py` — 用修好的 grader 原地重判（PYTHONPATH=rl/train:scripts:agent_loop）。

## 7. Codex 接手立刻做的三件事
1. `pgrep -af evo_task_loop` 杀掉所有遗留进化循环（监控 bn760odp2/council、bn9f7fowp/gov 已 timeout，但进程可能还在）。
2. 检查 DSv4-Pro run（PID 247119 / `/tmp/dsv4pro_swarm.out`）拿 Overall mean。
3. **最高优先级**：打开一条 `term=0.000` transcript，判断是真没写文件还是 grader artifact。

## 8. 相关 memory
- `project_swarm_base_model_instruction_ceiling.md`
- `project_dsv4flash_swarm_upper_bound.md`
- `feedback_debug_k1.md`（调机制用 K=1）
