# Swarm Policy Strategy Types

| 策略 | Lead 角色 | Sub-agent 角色 | 难点 | 适合训练什么 |
|---|---|---|---|---|
| Lead-Orchestrated Swarm | 持续控制任务全过程 | 辅助抽取 / 验证 | 长链路控制、停止时机、落盘交付 | orchestration policy |
| Candidate Swarm | 选择 / 合并候选结果 | 独立完成任务 | 候选质量、选择判断、冲突合并 | selector / merger policy |

`Lead-Orchestrated Swarm` 学的是如何带队完成任务；`Candidate Swarm` 学的是如何从团队候选结果中选出或合成更可靠的答案。
