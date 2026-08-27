# 与 Eureka 的差异化说明

> 本文档说明本 skill 与 NVIDIA Eureka（LLM 自动生成 reward 的研究原型）的差异化定位。

## Eureka 是什么

Eureka（Ma et al., 2023, ICLR 2024, arXiv:2310.12931）是 NVIDIA Research 提出的基于 LLM 的奖励函数自动生成算法。核心流程：

1. 从环境源码中提取"骨架代码"（reward-relevant 上下文，通过 `prune_env.py` 裁剪）
2. LLM（GPT-4）**零样本**生成 16 个 reward 函数候选
3. 每个候选跑一次 RL 训练，收集标量评估指标（episode return / success rate）
4. 把训练曲线、代码片段、文本反馈喂回 LLM，做**上下文进化优化**（in-context evolutionary optimization）
5. 迭代默认 5 轮，输出最佳 reward

**报告效果**：在 83% 的任务上超越人类专家，平均归一化改进 52%。

## Eureka 的局限性

| 局限 | 说明 |
|------|------|
| **仅生成 reward** | 不生成 observation、action、termination、curriculum、domain randomization 参数 |
| **绑定 Isaac Gym** | 原版基于已停更的 Isaac Gym Preview 4，不原生支持 Isaac Lab |
| **IsaacLabEureka 受限** | 官方移植版仅支持 `DirectRLEnv`，不支持 manager-based env（Isaac Lab 推荐范式） |
| **非产品** | README 明确声明 "strictly for research purposes, not an official product from NVIDIA" |
| **算力门槛高** | 16 个候选 × 8GB 显存，论文用 4×V100 |
| **依赖手写 success metric** | 每个任务需要人工定义 `success_metric` 函数 |
| **无 sim-to-real 闭环** | 原版仅仿真评估（DrEureka 后续补充） |
| **绑定 OpenAI API** | 硬绑 GPT-4，不支持本地开源模型，有成本与可复现性隐患 |
| **错误处理粗糙** | LLM 生成代码出错就跳过该 iteration |
| **不利用内置 MDP 库** | 完全从零生成，不复用 Isaac Lab 的 `mdp.rewards` 函数库 |

## 本 skill 的差异化

### 1. 原生 Isaac Lab manager-based 工作流

Eureka/IsaacLabEureka 生成裸 reward 函数（`def reward(env): ...`）。本 skill 直接生成 Isaac Lab 的 `RewardTermCfg` 三元组并注入 `RewardsCfg`：

```python
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import math

@configclass
class RewardsCfg:
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
```

这符合 Isaac Lab 推荐的 manager-based 范式，输出可直接挂载到 env config。Eureka 的 IsaacLabEureka 移植版仅支持 `DirectRLEnv`，本 skill 覆盖 manager-based env。

### 2. 复用内置 reward 函数库

Isaac Lab 自带 `isaaclab.envs.mdp.rewards` 模块，提供 20+ 个经过验证的 reward 函数（`track_lin_vel_xy_exp`、`flat_orientation_l2`、`joint_torques_l2` 等）。本 skill 优先以这些内置函数组合，降低代码错误率。Eureka 完全从零生成，不利用这个库。

详见 `resources/isaac_lab_api.json` 中的 `builtin_rewards` 字段。

### 3. 全流程，不止 reward

Eureka 只产出 reward 函数体。本 skill 同时产出：
- **weight 初值**：基于工程经验启发式（main objective : constraint ≈ 1 : 0.1-0.5）
- **退火策略**：curriculum 期间动态调权重
- **终止条件建议**：与 reward 配套的 termination 配置
- **完整的 `RewardsCfg`**：可直接挂载到 env config

### 4. 静态校验 + 运行时验证闭环

Eureka 仅用标量 return 曲线作为反馈。本 skill 在送回 LLM 前做：
- PyTree shape 校验（reward tensor 必须是 `(num_envs,)`）
- NaN/Inf 检测
- reward 量级归一化诊断
- Isaac Lab API 调用正确性检查

把"训练崩溃"转化为结构化反馈，迭代更高效。

### 5. 训练诊断（Eureka 完全没有）

Eureka 不做训练失败诊断。本 skill 的**模块 2（训练失败诊断系统）**从 tensorboard 日志识别 15 种 failure mode，给出针对性修复建议。这是 Eureka 完全缺失的能力。

### 6. 可解释性

Eureka 是黑盒（LLM 生成，结果看运气）。本 skill 的诊断引擎基于规则推理，每条建议可追溯到 `resources/failure_modes.json` 中的具体条目，可解释、可调试。

### 7. 支持本地/开源 LLM

Eureka 硬绑 OpenAI API。本 skill 的设计不依赖特定 LLM provider，可在 Claude、GPT、本地开源模型（Qwen、DeepSeek-Coder 等）上运行。

## 不与 Eureka 竞争的方面

诚实说明——本 skill **不是** Eureka 的替代品，定位不同：

- **自动化程度**：Eureka 是全自动迭代优化（16 候选 × 5 轮）。本 skill 是半自动（一次生成 + 用户反馈 + 静态校验），不做大规模并行训练评估。
- **超越人类的 reward 发现**：Eureka 能发现人类想不到的 reward 设计。本 skill 不会——它基于 curated 的工程模式，输出符合工程直觉。
- **算力需求**：本 skill 不需要 4×V100，普通笔记本即可运行静态分析与代码生成。

**定位**：本 skill 是**工程副驾驶**，不是**自动 reward 发现引擎**。两者互补：Eureka 适合探索性研究（"找一个超越人类的 reward"），本 skill 适合日常工程开发（"快速生成一个靠谱的 reward + 诊断训练问题"）。

## 来源

- [Eureka 项目主页](https://eureka-research.github.io/)
- [Eureka 论文 arXiv:2310.12931](https://arxiv.org/abs/2310.12931)
- [Eureka GitHub](https://github.com/eureka-research/eureka)
- [IsaacLabEureka (官方移植)](https://github.com/isaac-sim/IsaacLabEureka)
- [DrEureka (sim-to-real 后续)](https://github.com/eureka-research/dreureka)
- [NVIDIA 官方博客](https://blogs.nvidia.com/blog/eureka-robotics-research/)
