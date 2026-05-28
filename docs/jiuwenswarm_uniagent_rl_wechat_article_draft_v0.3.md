# 《jiuwenswarm+uniagent 打造智能体在线RL》

让 Agent 会调用工具，已经不稀奇了。

给它一个任务，它能拆步骤、查资料、写文件、调用 API，也能在一个长上下文里完成不少复杂工作。再往前一步，JiuwenSwarm 已经可以把多个 Agent 组织成一个团队：有人负责规划，有人负责执行，有人负责审校，有人负责汇总。

但真正把 Agent 放进真实使用场景里，问题很快就会出现。

同一个任务，为什么这次会先查资料，下次却直接写结论？为什么有些场景里主 Agent 明明知道该调用工具，却总是在关键节点漏掉验证？为什么团队协作时，有时候拆工拆得很好，有时候却把简单任务拆成一堆无效子任务？为什么一个垂类场景反复出现，系统还是不能越用越稳？

这些问题不完全是“模型会不会”的问题，更是 policy 的问题。

单 Agent 有主 agent policy：什么时候规划、什么时候调用工具、什么时候验证、什么时候停止。多 Agent 有 swarm policy：什么时候组队、怎么分工、任务如何交接、谁来审核、失败后怎么重试。

如果说 Harness Engineering 解决的是“如何把模型组织成能行动的 Agent”，Coordination Engineering 解决的是“如何把多个 Agent 组织成能协作的团队”，那么下一步就很自然：

**让这些行动策略和协作策略，也能在真实任务中持续学习。**

这就是 JiuwenSwarm + UniAgent 这次要补上的在线 RL 能力。它不是为了重新训练一个大模型，也不是把所有问题都推给更大的参数量，而是把真实任务中的轨迹、失败、反馈和过程信号转成训练信号，让 Agent policy 和 Swarm policy 在使用中持续变强。

这里的“在线”，指的是用户持续使用，系统持续采集真实任务轨迹，后台持续训练和评测；当新策略通过发布门控后，再通过 adapter 热更新到推理服务中。用户不需要停下来等训练，也不需要手动切换模型。下一次继续使用时，Agent 就可以带着更新后的策略回来。

对用户来说，这件事最直观的体验就是：**越用越懂你**。

UniAgent 在这套体系中承担 agent rollout 和训练底座：它把 agent 运行拆成 model、tool、env 三层，让模型后端、工具系统和执行环境可以解耦替换；同一套 interaction stack 既能跑大规模 agent 采样，也能接到 verl 做 RL 训练。这一点非常关键，因为在线 RL 最怕“训练时一套环境，推理时另一套环境”。

## 一、为什么 Agent 需要在线 RL

过去我们优化 Agent，通常有几条路。

第一条路是改 prompt。哪里错了，就在 system prompt 里加一句约束；哪里不稳定，就补一段流程说明。

第二条路是改 Harness。补工具、改 tool description、加 Rail、写 skill、加 stop condition，让 Agent 的运行环境更适合某个场景。

第三条路是换更强的模型。主模型从小参数换成大参数，很多问题确实会立刻缓解。

这些方法都有效，但它们也都有边界。

Prompt 很容易越写越长，最后变成一堆互相牵制的规则；Harness 可以把工具和流程搭好，但不一定能让模型在每个状态下都做出最优选择；换大模型成本高，而且不是所有私有化、低延迟、低成本场景都能接受。

更关键的是，很多 Agent 失败并不是因为它完全不知道答案，而是因为它在长链路里做错了几个策略选择。

比如会议分析任务里，它知道应该识别参会方，却漏掉了上下文中后半段出现的机构；它知道应该按角色分组，却把委员会成员、政府机构和商业公司混在一起；它知道需要核对证据，却在生成最终答案前没有回看原文。

这些错误很适合通过 RL 来修。

因为 RL 关注的不是“标准答案长什么样”，而是“在一个任务过程中，什么行动会让最后结果更好”。对 Agent 来说，这件事尤其重要：它的能力不是一次输出完成的，而是在规划、工具调用、观察、修正、汇总之间一步一步形成的。

JiuwenSwarm + UniAgent 的在线 RL，就是把这条链路闭起来：

真实任务产生轨迹，轨迹进入分析模块，任务结果、用户反馈和过程信号暴露 policy 缺点；数据模块构造专项训练样本，RL 更新主 agent 或团队协作策略；新的 adapter 通过发布门控后热更新回推理运行时。

这不是一次离线实验，而是一条面向长期运行的自进化链路。用户在前台继续工作，系统在后台完成轨迹积累、训练、评测和热更新；当同类任务再次出现时，Agent 能更贴近用户的偏好、流程和判断标准。

## 二、哪些场景真的需要 RL

不是所有 Agent 场景都应该上 RL。

有些问题靠 prompt 就能解决，有些问题靠补工具就能解决，有些问题靠写 skill 更快。RL 的价值在于：当系统已经能跑通，但在高频、长链路、策略敏感的任务里反复出现同类失败时，它可以把这些失败转成可学习的 policy 更新。

在 JiuwenSwarm + UniAgent 的体系里，最适合在线 RL 的场景可以归纳为三类：主 agent policy、场景 policy 和 swarm policy。

### 1. 主 agent policy：让受限模型在长链路任务里更稳定

很多场景不会一上来就把最强模型放在主 agent 位置。

原因很现实：成本、延迟、并发、部署环境、数据边界，都会限制主模型的参数规模。对企业是私有化和成本问题，对个人用户也是响应速度、设备资源和使用价格问题。4B、7B、14B 这类模型在很多场景里已经“接近可用”，但一旦进入长链路任务，就会出现不稳定。

它们可能会：

- 规划得出来，但执行中容易偏离；
- 会调用工具，但调用时机不稳定；
- 知道要验证，但经常在最后一步省略；
- 能完成简单任务，但复杂任务里容易过早结束；
- 单轮输出不错，多轮状态管理不稳。

这种情况下，RL 不是要把小模型变成全能大模型，而是补强主 agent policy，让它在固定运行环境里形成更稳定的行动策略。

换句话说，我们不追求“什么都补”，而是补那些在真实任务轨迹里反复出现、又能通过训练修正的能力：什么时候拆任务、什么时候查证据、什么时候调用工具、什么时候让团队成员介入、什么时候不要继续扩写。

这类场景的核心判断标准是：**模型基础能力已经接近可用，主要失败来自规划、工具调用、验证、停止条件和多轮状态管理的不稳定。**

### 2. 场景 policy：让高频垂类任务越用越贴合用户

RL 最怕没有稳定目标。

如果任务类型每天都变、成功标准说不清、反馈方式全靠主观判断，RL 很难形成可靠收益。相反，只要用户有明确的高频场景，RL 的价值就会变得很清楚。这个用户可以是企业团队，也可以是个人用户。

比如：

- 个人用户每天都让 Agent 整理日程、邮件、笔记和待办；
- 创作者反复让 Agent 按自己的风格写选题、脚本、标题和分发文案；
- 学生或研究者持续让 Agent 读论文、做文献卡片、整理实验记录；
- 开发者每个项目都让 Agent 跑固定格式的代码审查、测试和 issue 归因；
- 企业团队每天都要做会议纪要、stakeholder 分析、竞品更新、投研或合规报告；
- 每次任务都能通过规则、测试集、人工反馈或用户偏好判断好坏。

这类场景有三个优点。

第一，轨迹足够多。高频任务会持续产生成功和失败样本，不需要为了训练额外造大量离线数据。

第二，目标足够稳定。同一类任务反复出现，系统能学到可复用的场景 policy，而不是追着随机需求跑。

第三，反馈足够明确。只要任务结果、用户反馈或自动评测能稳定反映问题，RL 就可以围绕这些问题做专项提升。

因此，适合在线 RL 的场景通常需要同时满足四个条件：**高频、稳定、可评估、能闭环。** 高频提供轨迹，稳定提供可学习分布，反馈和评测提供优化信号，闭环让系统能把下一次使用变得更贴合用户。

### 3. Swarm policy：让多 Agent 团队学习更优协作路径

单 Agent 的问题是“自己怎么做任务”。

Swarm 的问题更复杂：一支团队应该怎么协作。

在 JiuwenSwarm 里，一个任务可能由多个 Agent 共同完成。Leader 需要拆解目标、安排成员、管理依赖；Teammate 需要认领任务、独立执行、汇报结果；审校 Agent 需要判断是否通过；必要时还要返工、重试、换人。

这里面有大量策略选择：

- 一个任务到底要不要组队；
- 应该拆成几个子任务；
- 先研究再写作，还是边研究边写；
- 哪些任务可以并行，哪些必须串行；
- 什么时候需要审校 Agent 介入；
- Teammate 卡住时是继续等待、重新分配，还是让 Leader 改计划；
- 最终汇总时应该信任成员结果，还是回到原始证据重新验证。

这些选择共同构成 swarm policy：面向 AgentTeam 的任务分解、角色分配、交接审校和失败恢复策略。

不同用户、不同场景、不同团队偏好，对 swarm policy 的要求并不一样。个人创作者可能更重视风格一致和素材复用；研究者可能更重视证据链和引用检查；开发者可能更重视测试、回滚和最小改动；企业团队可能更重视审批、合规和交付格式。

只靠固定模板，很难覆盖所有协作偏好。

RL 的价值在于：它可以从真实团队执行轨迹里学习“什么样的分工和交接更容易成功”，把团队协作从手写规则推进到数据驱动的策略优化。

这也是 JiuwenSwarm 在线 RL 最重要的方向之一：优化对象从单个 Agent 的行动策略，进一步扩展到 AgentTeam 的协作策略。

## 三、为什么这里需要 UniAgent

在线 RL 要真正落到 Agent 场景里，需要的不只是一个训练算法，还需要一套稳定的采样和训练底座。

Agent 的训练样本不是普通的 prompt-response pair。一次有效样本往往包含多轮推理、工具调用、环境 observation、状态更新、失败恢复和最终 reward。只有把这些过程稳定记录下来，RL 才能真正学习“在任务过程中如何行动”，而不是只学习最后应该怎么说。

UniAgent 提供的，正是这样一套面向 agent interaction 的 rollout 和训练底座。

它把一个 agent 任务拆成三个稳定接口：

- **model**：负责推理和决策，可以接 vLLM、SGLang、内部推理网关或 OpenAI-compatible 服务；
- **tool**：负责让模型感知和行动，工具 schema、工具安装、工具执行都在同一套机制下管理；
- **env**：负责保存任务状态和执行动作，支持为每个样本启动独立 sandbox。

这并不意味着用户请求必须经过 UniAgent。用户仍然可以直接使用 JiuwenSwarm；Gateway/Rail 在旁路负责请求路由、轨迹采集和训练数据沉淀。UniAgent 的价值，是把这些真实轨迹进一步放到标准化的 model/tool/env 交互环境中做大规模 rollout、重放和 RL 训练。

更重要的是，UniAgent 天然面向大规模 agent interaction。它支持多任务并行执行，每个样本可以拥有自己的 sandbox，系统收集多轮交互轨迹和 reward，再进入后续训练。对在线 RL 来说，这意味着后台训练不依赖前台用户请求同步执行，而是可以基于沉淀轨迹扩展到大规模 rollout。

在训练侧，UniAgent 可以把同一套 rollout stack 接到 verl。也就是说，训练样本不是普通的 prompt-response pair，而是一次完整的多轮工具交互：启动环境、运行 agent、调用工具、执行动作、计算任务 reward，再把结果送回 RL trainer。

这也是我们强调 fully async 的原因。Agent 任务天然长短不一：有的几轮结束，有的要几十轮；有的工具调用很快，有的 sandbox 操作很慢；有的样本很快拿到 reward，有的需要更长时间评估。如果强行同步等待整批 rollout 结束，训练资源和 rollout 资源都会互相拖住。

UniAgent 的 fully async 和 partial rollout 能力，正好对应 long-horizon agent RL 的工程痛点：rollout worker 和 training worker 可以独立推进，训练不必被最慢的任务卡住，长链路任务也能持续产生可用训练信号。

所以在这套方案里，分工可以理解为：

- **JiuwenSwarm** 负责前台用户体验、团队协作和场景侧 policy：谁来规划、谁来执行、谁来审校、失败后如何返工；
- **Gateway/Rail** 负责请求路由、旁路采集和训练轨迹沉淀；
- **UniAgent** 负责后台标准化 rollout 和训练底座：model、tool、env、sandbox、并行 rollout、reward、verl 训练连接；
- **RL** 负责把真实任务轨迹暴露出的策略缺点，转成下一轮 policy 更新。

这三层合在一起，才是“智能体在线 RL”真正可落地的形态。

## 四、JiuwenSwarm + UniAgent 的在线 RL 闭环

这套系统的核心目标很直接：让真实任务产生的数据，反过来推动 Agent 变强。

整体链路可以分成六步。

1. 用户请求进入 JiuwenSwarm，由 Leader 或主 agent 决定是否组队、如何拆解任务；
2. Gateway/Rail 在旁路完成请求路由、轨迹采集和 reward 记录，不要求用户显式进入训练系统；
3. 沉淀下来的真实轨迹进入 UniAgent rollout stack，在 model、tool、env 三层中做重放、采样和扩展；
4. 轨迹分析模块从真实任务轨迹、任务结果和用户反馈中定位缺点；
5. 数据模块构造专项训练数据和独立评测集；
6. RL 训练完成后，通过发布门控的 adapter 热更新回推理运行时，下一次同类任务自动使用新策略。

<figure class="figure flow-figure" aria-label="JiuwenSwarm + UniAgent 在线 RL 闭环">
<svg viewBox="0 0 980 560" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="flow-title flow-desc">
  <title id="flow-title">JiuwenSwarm + UniAgent 在线 RL 闭环</title>
  <desc id="flow-desc">在线 RL 闭环由前台使用、旁路采集、后台训练和热更新四个相邻泳道组成。</desc>
  <defs>
    <filter id="card-shadow" x="-10%" y="-10%" width="120%" height="130%"><feDropShadow dx="0" dy="5" stdDeviation="6" flood-color="#0f172a" flood-opacity="0.08" /></filter>
    <style>
      .title { font: 800 25px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #111827; }
      .sub { font: 500 14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #64748b; }
      .lane-title { font: 800 17px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
      .card-title { font: 800 14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #111827; }
      .card-sub { font: 500 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #64748b; }
      .small { font: 700 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #64748b; }
      .note { font: 700 13px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #334155; }
    </style>
  </defs>

  <rect x="0" y="0" width="980" height="560" rx="24" fill="#f8fafc" />
  <text x="48" y="58" class="title">JiuwenSwarm + UniAgent 在线 RL 闭环</text>
  <text x="48" y="88" class="sub">用户直接使用 JiuwenSwarm；Gateway/Rail 旁路沉淀轨迹；UniAgent 在后台做 rollout 和训练。</text>

  <rect x="48" y="120" width="250" height="356" rx="18" fill="#e8f6f3" stroke="#94d2c8" stroke-width="1.4" />
  <rect x="328" y="120" width="170" height="356" rx="18" fill="#ffffff" stroke="#94d2c8" stroke-width="1.4" />
  <rect x="528" y="120" width="250" height="356" rx="18" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.4" />
  <rect x="808" y="120" width="124" height="356" rx="18" fill="#ffffff" stroke="#d1d5db" stroke-width="1.4" />

  <text x="74" y="154" class="lane-title" fill="#0f766e">前台使用</text>
  <text x="354" y="154" class="lane-title" fill="#0f766e">旁路采集</text>
  <text x="554" y="154" class="lane-title" fill="#334155">后台训练</text>
  <text x="870" y="154" text-anchor="middle" class="lane-title" fill="#0f766e">热更新</text>

  <g filter="url(#card-shadow)">
    <rect x="78" y="188" width="190" height="62" rx="13" fill="#ffffff" stroke="#0f766e" stroke-width="1.3" />
    <text x="173" y="213" text-anchor="middle" class="card-title">用户任务</text>
    <text x="173" y="234" text-anchor="middle" class="card-sub">真实请求 / 反馈</text>
    <rect x="78" y="286" width="190" height="62" rx="13" fill="#ffffff" stroke="#0f766e" stroke-width="1.3" />
    <text x="173" y="311" text-anchor="middle" class="card-title">JiuwenSwarm</text>
    <text x="173" y="332" text-anchor="middle" class="card-sub">规划 / 组队 / 协作</text>
    <rect x="78" y="384" width="190" height="62" rx="13" fill="#ffffff" stroke="#0f766e" stroke-width="1.3" />
    <text x="173" y="409" text-anchor="middle" class="card-title">任务结果</text>
    <text x="173" y="430" text-anchor="middle" class="card-sub">交付 / 下次继续</text>

    <rect x="354" y="248" width="118" height="116" rx="14" fill="#ffffff" stroke="#0f766e" stroke-width="1.3" />
    <text x="413" y="278" text-anchor="middle" class="card-title">Gateway / Rail</text>
    <text x="413" y="306" text-anchor="middle" class="card-sub">请求路由</text>
    <text x="413" y="328" text-anchor="middle" class="card-sub">轨迹采集</text>
    <text x="413" y="350" text-anchor="middle" class="card-sub">reward 记录</text>

    <rect x="558" y="172" width="190" height="54" rx="12" fill="#ffffff" stroke="#64748b" stroke-width="1.2" />
    <text x="653" y="194" text-anchor="middle" class="card-title">Trajectory Store</text>
    <text x="653" y="213" text-anchor="middle" class="card-sub">真实轨迹</text>
    <rect x="558" y="248" width="190" height="54" rx="12" fill="#ffffff" stroke="#64748b" stroke-width="1.2" />
    <text x="653" y="270" text-anchor="middle" class="card-title">UniAgent Rollout</text>
    <text x="653" y="289" text-anchor="middle" class="card-sub">model / tool / env</text>
    <rect x="558" y="324" width="190" height="54" rx="12" fill="#ffffff" stroke="#64748b" stroke-width="1.2" />
    <text x="653" y="346" text-anchor="middle" class="card-title">Analysis + Data</text>
    <text x="653" y="365" text-anchor="middle" class="card-sub">失败诊断 / 专项数据</text>
    <rect x="558" y="400" width="190" height="54" rx="12" fill="#ffffff" stroke="#64748b" stroke-width="1.2" />
    <text x="653" y="422" text-anchor="middle" class="card-title">RL Trainer</text>
    <text x="653" y="441" text-anchor="middle" class="card-sub">fully async / adapter</text>

    <rect x="826" y="198" width="88" height="78" rx="14" fill="#e8f6f3" stroke="#0f766e" stroke-width="1.2" />
    <text x="870" y="229" text-anchor="middle" class="card-title">Adapter</text>
    <text x="870" y="251" text-anchor="middle" class="card-sub">评测通过</text>
    <rect x="826" y="326" width="88" height="92" rx="14" fill="#e8f6f3" stroke="#0f766e" stroke-width="1.2" />
    <text x="870" y="357" text-anchor="middle" class="card-title">新策略</text>
    <text x="870" y="379" text-anchor="middle" class="card-sub">热更新</text>
    <text x="870" y="400" text-anchor="middle" class="card-sub">下次使用</text>
  </g>

  <g stroke="#0f766e" stroke-width="2.4" stroke-linecap="round" fill="none">
    <path d="M173 250 V286" />
    <path d="M173 348 V384" />
    <path d="M268 317 H354" stroke-dasharray="6 8" />
  </g>
  <text x="282" y="302" class="small">旁路</text>

  <g stroke="#64748b" stroke-width="2.4" stroke-linecap="round" fill="none">
    <path d="M472 306 H528" />
    <path d="M528 306 V199 H558" />
    <path d="M653 226 V248" />
    <path d="M653 302 V324" />
    <path d="M653 378 V400" />
  </g>

  <g stroke="#0f766e" stroke-width="2.4" stroke-linecap="round" fill="none">
    <path d="M748 427 H778" />
    <path d="M778 427 V237 H826" />
    <path d="M870 276 V326" />
  </g>
  <text x="785" y="222" class="small" fill="#0f766e">Release Gate</text>
  <text x="808" y="452" class="small" fill="#0f766e">adapter 热更新到推理运行时</text>
</svg>
<figcaption>图 1：用户请求直接进入 JiuwenSwarm；Gateway/Rail 旁路沉淀真实轨迹；UniAgent 在后台提供 rollout 和训练底座；adapter 通过发布门控后热更新到推理运行时，下一次同类任务自动使用新策略。</figcaption>
</figure>

这条链路里，最重要的不是“能不能训”，而是训练信号从哪里来，以及这些信号是不是来自真实 agent 运行链路。

JiuwenSwarm + UniAgent 面向真实运行环境定义训练任务。真实用户请求、真实工具返回、真实 sandbox 状态、真实失败轨迹，才是 Agent policy 和 swarm policy 最应该学习的对象。

所以我们强调三点。

第一，训推 runtime 要一致。训练数据来自真实执行链路，而不是另起一套和线上环境不一样的模拟环境。UniAgent 的同一套 interaction stack 既服务大规模执行，也连接 RL 训练，可以减少训练环境和推理环境之间的偏差。

第二，训练和推理要异步。用户请求不能被训练阻塞，rollout 也不应该被最慢的样本卡住。轨迹持续进入队列，后台按阈值触发训练；fully async 让 rollout worker 和 training worker 解耦推进，训练完成后再通过发布门控决定是否发布。

第三，模型更新要热更新、可回退。每次 RL 训练产出的 adapter 都有版本、指标和适用范围。只有在独立评测和发布门控中确认收益后，才会热更新到推理侧；如果指标回退，就停在训练侧，不影响线上运行。

简单说，在线 RL 不是“边用边冒险”，而是“用户边用，系统边积累；后台训练，通过发布门控后热更新；下一次使用时，Agent 变得更贴合这个用户和这个场景”。

## 五、算法上怎么做：从真实轨迹到专项提升

这套 RL 训练流程不会只依赖最终分数。

Agent 任务的难点在于长链路。最终 reward 虽然重要，但信号相对稀疏：一个任务失败了，可能是开头规划错了，可能是中间漏查证据，可能是工具调用参数错了，也可能是最后总结时把信息合并错了。

仅依赖最终成功或失败，难以定位具体的策略偏差。

因此，JiuwenSwarm + UniAgent 的 RL 训练会围绕真实任务轨迹展开两类处理。

### 1. 基于真实任务轨迹的失败模式诊断与专项数据构造

专项数据不是从改 prompt 开始，而是从轨迹归因开始。

每个任务样本会被保存成一条可回放的 interaction trace：用户输入、Agent 的中间决策、工具调用、observation、文件写入、最终答案、任务得分和用户反馈都保留下来。轨迹分析模块不只看“最后对不对”，而是定位偏差发生在哪个过程节点：规划阶段、信息检索阶段、证据绑定阶段、结构化输出阶段，还是最终校验阶段。

以 PinchBench meeting_analysis 实验为例，第一版专项数据先聚焦 single agent 轨迹中最稳定复现的几类问题：

| 能力弱项 | 轨迹中的表现 | 数据构造方式 |
|---|---|---|
| 长上下文覆盖不稳 | Agent 读完开头几段后就开始写结论，后半段新出现的参会方、决策和行动项没有进入最终答案 | 构造 `pre_final_audit` 样本，要求按 early / middle / late 分段检查，再做全局合并 |
| 证据台账不完整 | 中间过程没有沉淀“信息项-来源-证据”的证据台账，最终答案里出现判断但无法回到原文定位 | 构造 `evidence_ledger` 样本，要求先整理关键事实、出处和不确定项，再生成最终文件 |
| 输出前校验不足 | 最终答案直接收敛，没有反查原文；vote、owner、deadline、数字或状态容易被写漏、写反 | 构造 `pre_final_audit` 样本，要求在写最终答案前检查漏项、错绑、unsupported claim |
| 实体关系绑定弱 | 人名或机构识别出来了，但所属组织、发言立场、对应议题没有绑定清楚 | 通过 stakeholder / attendee / summary 类任务间接覆盖，在 gold 中检查实体、角色和证据是否一致 |
| 结构化分组不稳定 | 任务要求按角色或机构分类，模型有时能分类，有时把成员、机构、外部参与方混成一组 | 通过原任务 schema 和 `evidence_ledger` 间接覆盖，训练模型先整理中间结构再输出 |

这些样本不会被简单写成“注意覆盖完整”这类软约束，而是转成可以进入 RL 的过程信号：是否继续扫描材料、是否维护证据台账、是否在收敛输出前做反查、实体和证据是否能稳定绑定。

这样做的价值在于，训练目标从“提高一个总分”变成“修正一组稳定复现的 single agent 策略偏差”。对 meeting_analysis 来说，第一版数据重点不是学习某个样本答案，而是强化全局覆盖、证据沉淀、输出前校验，以及实体和结构化信息的稳定组织。

### 2. Swarm policy 的团队协作策略优化

在多 Agent 场景中，RL 的优化对象不再局限于单个 Agent 的行动选择，还包括团队层面的分工、依赖、交接、审校和返工策略。

Swarm policy 对应 AgentTeam 执行过程中的一组可观察决策：

- Leader 是否应该组队；
- 如何选择成员角色；
- 如何拆任务和设置依赖；
- Teammate 什么时候主动认领；
- 什么时候请求 Leader 决策；
- 什么时候需要审校；
- 失败后是重试、返工、换人，还是收敛输出。

过去这些策略更多依赖规则、模板和 prompt 约束，能够支撑基础协作流程，但难以根据用户场景和真实执行反馈持续优化。

在线 RL 可以把团队执行轨迹变成训练信号。一次成功的协作，不只沉淀最终答案，也沉淀任务拆分、依赖安排、消息沟通、审校返工这些过程。一次失败的协作，也能暴露团队 policy 的问题：是不是拆得太碎、是不是没有审校、是不是成员重复工作、是不是 Leader 太晚介入。

这让 JiuwenSwarm 的自进化不只停留在单个 Agent 的工具使用能力上，而是进入团队层面的策略优化。

## 六、Meeting Analysis 实验：先在长链路任务上评估

我们先选择 PinchBench 的 meeting_analysis 场景做评估。

这个场景很适合测试 Agent policy。因为它不是简单问答，而是要求模型从较长会议材料中识别信息、抽取角色、组织结构、形成最终分析。它既考察文本理解，也考察过程策略：是否回看材料、是否覆盖完整、是否能按场景语义分组、是否能在输出前做一致性检查。

当前实验基于 Qwen3-4B，训练方法以 GRPO 为主，并对比 terminal reward 和 swarm policy 训练信号。

Swarm policy 训练信号关注的不是单个答案片段，而是任务过程中的协作决策：是否需要拆分任务、是否需要补充检索、是否触发审校、是否在输出前回到原文验证，以及失败后应该继续修正还是收敛输出。它把 meeting_analysis 这类长链路任务中的“团队策略选择”显式纳入优化目标。

### 实验设置

| 项目 | 设置 |
|---|---|
| 场景 | PinchBench meeting_analysis |
| 主模型 | Qwen3-4B |
| 训练方法 | GRPO |
| 对比组 | Baseline / GRPO + terminal reward / GRPO + terminal reward + swarm policy |
| 执行形态 | UniAgent-style 多轮 interaction，保留工具调用、环境 observation 和 reward 轨迹 |
| 评测方式 | 独立评测集多次运行取平均 |
| 观察重点 | overall score、收敛速度、stakeholder 覆盖、结构化输出质量 |

阶段性结果里，RL 后的 meeting_analysis 分数已经出现可观察提升。Terminal reward 能把 baseline 往上推；加入 swarm policy 训练信号后，模型在信息覆盖、结构化组织和审校触发上的表现更稳定，早期轮次就能超过 terminal-only 更后面的 peak。

下图展示了独立评测得分在训练轮次中的变化趋势。长链路 agent RL 通常不是单调上升，而是在探索和收敛之间波动前进。

<figure class="figure" aria-label="meeting_analysis 独立评测收敛曲线">
<svg viewBox="0 0 980 430" role="img" xmlns="http://www.w3.org/2000/svg" aria-labelledby="chart-title chart-desc">
  <title id="chart-title">meeting_analysis evaluation score</title>
  <desc id="chart-desc">Baseline、GRPO terminal 和 GRPO plus swarm policy 三条独立评测曲线对比，展示波动中收敛形态。</desc>
  <rect x="0" y="0" width="980" height="430" rx="18" fill="#f8fafc" />
  <text x="44" y="48" fill="#111827" font-size="24" font-weight="700">meeting_analysis evaluation score</text>
  <text x="44" y="78" fill="#64748b" font-size="14">PinchBench meeting_analysis evaluation trend</text>
  <g stroke="#e5e7eb" stroke-width="1"><line x1="86" y1="330" x2="910" y2="330" /><line x1="86" y1="278" x2="910" y2="278" /><line x1="86" y1="226" x2="910" y2="226" /><line x1="86" y1="174" x2="910" y2="174" /><line x1="86" y1="122" x2="910" y2="122" /></g>
  <g fill="#64748b" font-size="12"><text x="42" y="334">50%</text><text x="42" y="282">52%</text><text x="42" y="230">54%</text><text x="42" y="178">56%</text><text x="42" y="126">58%</text></g>
  <line x1="86" y1="340" x2="910" y2="340" stroke="#64748b" stroke-width="2" /><line x1="86" y1="340" x2="86" y2="108" stroke="#64748b" stroke-width="2" />
  <g fill="#64748b" font-size="12" text-anchor="middle"><text x="120" y="366">R0</text><text x="195" y="366">R1</text><text x="270" y="366">R2</text><text x="345" y="366">R3</text><text x="420" y="366">R4</text><text x="495" y="366">R5</text><text x="570" y="366">R6</text><text x="645" y="366">R7</text><text x="720" y="366">R8</text><text x="795" y="366">R9</text><text x="870" y="366">R10</text></g>
  <polyline points="120,314 195,314 270,314 345,314 420,314 495,314 570,314 645,314 720,314 795,314 870,314" fill="none" stroke="#94a3b8" stroke-width="3" stroke-dasharray="8 8" />
  <polyline points="120,314 195,283 270,301 345,242 420,260 495,200 570,221 645,190 720,198 795,182 870,187" fill="none" stroke="#0f766e" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" />
  <polyline points="120,314 195,142 270,176 345,158 420,148 495,164 570,135 645,145 720,128 795,134 870,123" fill="none" stroke="#7c3aed" stroke-width="4" stroke-linejoin="round" stroke-linecap="round" />
  <g font-size="13"><rect x="608" y="38" width="18" height="4" rx="2" fill="#94a3b8" /><text x="636" y="45" fill="#374151">Baseline</text><rect x="608" y="64" width="18" height="4" rx="2" fill="#0f766e" /><text x="636" y="71" fill="#374151">GRPO terminal</text><rect x="608" y="90" width="18" height="4" rx="2" fill="#7c3aed" /><text x="636" y="97" fill="#374151">GRPO + swarm policy</text></g>
  <text x="207" y="130" fill="#7c3aed" font-size="13" font-weight="700">57.2%</text><text x="507" y="188" fill="#0f766e" font-size="13" font-weight="700">55.0%</text>
</svg>
<figcaption>图 2：PinchBench meeting_analysis evaluation trend。swarm policy 训练信号在早期轮次带来更快提升，长链路 RL 整体呈现波动中收敛。</figcaption>
</figure>

这条曲线体现了三个现象。

第一，Agent RL 不一定每一轮都单调上涨。长链路任务的独立评测曲线会有波动，尤其是模型探索到新的策略时，短期可能出现回退。

第二，terminal reward 可以提升最终效果，但通常需要更多轮次才能稳定。

第三，swarm policy 训练信号能加快收敛，因为它把“最后错了”进一步拆成“任务过程中哪类策略选择导致了错误”，更适合 meeting_analysis 这类长链路任务。

### 阶段性结果表

| 配置 | Best score | 相对 baseline | 备注 |
|---|---:|---:|---|
| Baseline | 50.6% | - | Qwen3-4B，rope=2 |
| GRPO + terminal reward | 55.0% | +4.4% | R5 peak |
| GRPO + terminal reward + swarm policy | 57.24% | +6.64% | R1 即超过 terminal-only peak |

这组结果指向一个工程化结论：在 meeting_analysis 这种高频、可评估、策略敏感的长链路任务里，真实任务轨迹 + 失败模式诊断 + RL 训练，能让受限参数量的主 agent 获得稳定可见的策略收益。

### PinchBench meeting_analysis case study

为了进一步观察能力变化，我们从 PinchBench meeting_analysis 场景中选取代表性样本做 case study。

单样本 case study 的目标不是替代整体评测得分，而是分析模型在具体任务过程中的行为差异。我们关注三类可解释变化：

1. **实体覆盖率**：模型是否识别出 baseline 漏掉的关键 stakeholder、机构或角色；
2. **关系归因准确性**：模型是否能把人物、机构、立场和发言上下文正确绑定；
3. **输出结构稳定性**：模型是否能按政府机构、商业组织、委员会成员、其他参与方等维度组织结果。

case study 的对比维度如下：

| 对比项 | Baseline | RL 后 |
|---|---|---|
| stakeholder 数量 | 漏掉后半段出现的参与方 | 覆盖前后文主要参与方 |
| 组织结构 | 平铺列表，分类不稳定 | 按角色和机构类型分组 |
| 证据利用 | 部分结论缺少原文依据 | 关键判断能回到会议文本 |
| 输出稳定性 | 多次运行差异较大 | 格式和覆盖范围更稳定 |

这类样本的价值在于，它同时覆盖长文本理解、多实体抽取、关系归因和结构化输出，能够更清楚地观察 swarm policy 训练对任务过程的影响。

## 七、这套能力和 Auto Harness、AgentTeam、UniAgent 的关系

openJiuwen 之前已经在两个方向上做了很多工程化工作。

Auto Harness 解决的是 Harness 如何自动优化：prompt、tool、skill、rail、context、stop condition 等组件，可以围绕评测闭环自动迭代。

AgentTeam 解决的是多个 Agent 如何协同：Leader 负责规划和管理，Teammate 负责执行，团队通过任务、消息、共享工作区和事件机制完成协作。

UniAgent 解决的是 agent 执行和训练如何统一：model、tool、env 三层解耦，sandbox 承载真实执行状态，并行 rollout 收集多轮交互轨迹，同一套 interaction stack 可以接到 verl 做 RL。

JiuwenSwarm + UniAgent 在线 RL 接在这几层之后。

它不替代 Auto Harness，也不替代 AgentTeam，而是补上 policy 层的自进化：

- Harness 提供行动环境；
- AgentTeam 提供协作结构；
- UniAgent 提供可扩展的执行、采样和训练底座；
- RL 学习在这个环境和结构里怎么做决策。

如果一个问题可以通过补工具、改 prompt、写 skill 解决，就应该先走 Auto Harness；如果一个问题来自多人协作链路，就应该用 AgentTeam 把分工和状态管理起来；如果一个问题需要稳定采集多轮工具交互和 reward，就用 UniAgent 把执行和训练链路统一起来；如果一个问题在高频任务中反复出现，且已经有足够轨迹和可用反馈或评测信号，就进入在线 RL。

RL 不是万能按钮。它更适合用于那些已经具备可执行链路、能够稳定采集真实任务轨迹、并且存在明确反馈或评测信号的策略学习问题；工具缺失、prompt 不清、环境不稳定等工程问题，仍然应该先在 Harness 和运行时层面解决。

## 八、写在最后

Agent 变强，不只靠更大的模型。

模型当然重要，但真实系统里，Agent 的能力还来自 Harness、工具、记忆、技能、协作机制，以及最容易被忽略的 policy。

主 agent policy 决定它如何行动，swarm policy 决定一支 AgentTeam 如何协作。过去这些策略主要靠人工设计、prompt 约束和固定规则。JiuwenSwarm + UniAgent 的在线 RL，把它们变成可以被真实任务轨迹驱动、被独立评测检验、被版本化发布的工程能力。

这件事的意义在于：Agent 不再只是被写出来、配置出来、部署出来，而是可以在真实任务里持续学习。

当一个用户有稳定的高频垂类场景，当一个受限参数量模型需要逼近更强策略能力，当一支 AgentTeam 需要形成自己的协作方式，在线 RL 就不再是论文里的训练算法，而是智能体系统继续进化的一块基础设施。

让 Agent 学会行动。

让 Swarm 学会协作。

让真实任务，成为下一轮智能体进化的数据来源。
