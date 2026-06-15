# 周报 2026-06-15:qwen3.5-4b Agentic RL —— 端到端训练跑通 + 稳定委员会 reward

## 〇、相对上周(06-08)的跃迁

上周交付的是**数据生产 pipeline 的设计** + reward = `0.6·verifiable + 0.4·rubric`(独立打分),并明确两个未做项:**"跑一轮端到端训练验证"(列为"接下来最重要的一件事")** 和 **"pairwise comparison 替代独立打分"(P1 TODO)**。

本周把这两件都做了,而且更进一步:

| 维度 | 上周(06-08) | 本周(06-15) |
|---|---|---|
| 端到端训练 | 未验证(只有数据) | **跑通,多轮 on-policy 迭代,真出 LoRA + eval** |
| reward 形态 | verifiable + rubric **独立打分** | **RULER 相对委员会**(pairwise/listwise)+ rubric 注入 + base 参考锚 |
| base 模型 / serving | Qwen3-4B + vLLM | **Qwen3.5-4B + transformers+harness(弃 vLLM)** |
| judge | DSV4 Pro 单 judge | **异构委员会(ds-flash/qwen-max/minimax-M3)+ 顺序一致性 + null 校准** |
| 产出 | 训练数据 | **可比较的 RL 结论 + 一批方法论 insight** |

---

## 一、架构变化:为什么训推都改走 transformers+harness(弃 vLLM)

**vLLM 在 qwen3.5 上的在线 LoRA 是 no-op**(实测:vLLM base==lora logprob 完全相同;PEFT base vs base+lora 差 0.064)。即用 vLLM 服 LoRA 测的全是 base,所有 vLLM-lora 评测无效。

决策:**整条链路走 transformers**——
- 推理/rollout/eval:transformers+PEFT 的 OpenAI 兼容 shim(`tf_shim.py`),加载 base+attach LoRA,供 OpenClaw agent 调用;
- 训练:transformers 上的 PEFT GRPO(continue-train via `--lora-path`,logprobs 也挂同一 adapter);
- LoRA 本身保留(all-linear r32),只是不靠 vLLM 服。

---

## 二、稳定 judge:pairwise 异构委员会(做"准"+做"稳")

agentic RL 的主观题(advisory/gov 等)用单 judge 绝对打分极不稳(flash/minimax 是 MoE/reasoning,temp=0 仍非确定,同报告 ensemble 能 0.4~0.95)。本周定型一套稳定评分法:

1. **pairwise 相对判**(eval)/ **listwise 相对排名**(训练 reward)——避开绝对标尺漂移(RULER 思想);
2. **顺序一致性**:每对双序判,翻则记 tie(位置偏见→tie 不→噪声);
3. **异构委员会**:ds-flash + qwen-max + minimax-M3,跨族偏见不相关,多数聚合;
4. **null 校准**:base-vs-base 必须 ≈ 全 tie(实测 9/9 全 tie=零残余偏见,先验证方法再信结论)。

**战果:该法抓出了 automated 单口径漏判/被刷的真相**(见下)。

---

## 三、训练 reward:RULER rank + llm_rubric + base 参考 + hybrid

- **RULER 式 listwise 相对打分**当 GRPO reward(整组 K 条相对排名,GRPO 只吃组内相对值);
- **注入 pinchbench 人工写的 `llm_rubric`**(RULER 原生支持自定义 rubric)——相对排名 + 人工标准正交叠加,无 rubric 时回退通用;
- **base 参考锚(放法B)**:把一条 base 报告放进 judge prompt 做"校准尺",**只校准、不计分、不进 GRPO advantage 归一**(保住组内 spread);
- **hybrid:`reward = AUTO_W·automated + (1-AUTO_W)·committee`**(加法)。

---

## 四、实验结果(committee_w1 → blend → w2 → w3 → w4)

主口径 = **committee(相对质量 pairwise vs base)**;automated 为弱覆盖代理(参考)。

| 模型 | 配置 | committee vs base | automated OVERALL |
|---|---|---|---|
| round-1e | 旧 flash 绝对 reward | tech 注水被全胜 | (auto 被刷虚高) |
| committee_w1 | 纯 committee,off-policy 复用 | advisory LOSS / gov 平 / tech WIN | 0.833 |
| committee_blend | +llm_rubric,off-policy | advisory LOSS / gov 平 / tech WIN | 0.833 |
| **committee_w2** ★ | **AUTO_W=0.5,on-policy** | **advisory 平 / gov WIN / tech 平 → 1赢0输2平** | **0.852** |
| committee_w3 | AUTO_W=0.7,on-policy | advisory LOSS / gov 平 / tech 平 → 0赢1输 | 0.796 |
| committee_w4 | AUTO_W=0.2,on-policy | (验证中) | — |

**committee_w2(on-policy,AUTO_W=0.5)是当前最佳:committee 口径净胜 base(gov 显著赢、无败绩)。** 是 on-policy(非调权重)修好了 advisory 的 committee 退步。

---

## 五、关键 insights(本周的"金子")

1. **automated 是会被刷的弱指标,committee 才是有意义口径。** round-1e 把 tech 报告冲到 16K 注水,automated 刷到 0.944,但委员会全票判 base 赢(质量更差)。**主观题进退要看相对委员会,别信单 judge 绝对分。**

2. **automated ↔ committee 在"覆盖 vs 质量"轴上冲突。** automated=presence/coverage(偏长、看不出注水);committee=质量/grounding(惩罚注水)。底部一致(写没写),顶部对打(最大覆盖≠最高质量)。铁证:AUTO_W 0.5→0.7,advisory automated 涨了但 committee 崩、gov/tech 也被带坏,**三列全面更差**。→ **automated 该当"软门/地板"而非高权重并列项,AUTO_W 要低(≤0.3),committee 做主驱动。**

3. **harness 检查纪律(踩坑总结):一份"写出来却被打 0 分"的报告,几乎一定是 harness bug(超时/工作区同步/路径),不是模型问题。** 本周因 rollout 超时(360s 对 71K 文档不够)→ 写了却评分时文件没就位 → 假 auto=0 → advisory 整组被过滤、压根没训。已做成 `rollout_healthcheck.py`:rollout 后训练前自动标红 ALL-TIMEOUT / WRITTEN-BUT-AUTO=0 / NO-WRITE,有问题就停,不浪费训练。

4. **on-policy 是关键,不是调权重。** off-policy 复用 base rollout 时 advisory 永远 LOSS(全完整样本无对比);on-policy(temp=1.0 采样多样)+ 正确评分,才把 advisory 从 LOSS 拉到平。

---

## 六、TODO / 下一步

| 优先级 | 事项 | 状态 |
|---|---|---|
| P0 | committee_w4(AUTO_W=0.2)验证"更偏 committee 更好" | 跑中 |
| P0 | 从当前最佳(committee_w2)**真·重 rollout 迭代**(on-policy,贵但干净)把 advisory 从平推成赢 | 待 w4 结论后起 |
| P1 | 把 automated 从加权项改成"软门/floor gate"实现 | 待设计 |
| P1 | speaker_nasa(NASA 超长文档)结构性超时:更长 timeout 或文档分块 | 待修 |
| P2 | 严格 ablation 拆分(on-policy / base-ref / init 单变量) | 规划 |

---

## 七、思考

- **"评测口径"本身是一等公民。** 本周最大的认知升级不是某个 reward 技巧,而是确认了 **"committee(相对质量)是目标、automated 是会误导的代理"**,并量化了二者的冲突。这条会决定以后所有 reward 权重设置。
- **agentic RL 的瓶颈常在 harness,不在算法。** 多次"分数差"最后都查出是超时/服错 adapter/工作区路径——所以把"低级 bug 自动检测"做进流水线(rollout_healthcheck)比调超参更值钱。
- **承接上周的 flywheel 愿景**:数据生产(上周)→ 稳定 reward + 端到端训练(本周)→ 下一步是让"从最佳模型重 rollout"的 on-policy 飞轮真正转起来,每轮难度/对比跟着模型能力走。

---

文档与脚本(github `naive_ppo_qwen35`):`docs/committee_rl_methodology.md`(原理)、`docs/committee_reward_ablation.md`(全谱系结果+冲突分析)、`docs/SETUP.md`(复现)、`docs/auto_vs_committee_tradeoff.html`(可视化)、`scripts/tf_agentic/*`(全套训练+评测脚本)。
