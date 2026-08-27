# 训练失败模式知识库

> 本文档是 RL 训练失败诊断的知识库。结构化数据见 `resources/failure_modes.json`，本文档提供方法论、分类与可读解释。

## 如何使用

**诊断流程**（由模块 2 自动执行）：
1. `scripts/log_analyzer.py` 解析 tensorboard 日志，识别异常症状（reward 平台、梯度爆炸、entropy 塌缩等）
2. `scripts/diagnosis_engine.py` 把症状组合匹配到 `resources/failure_modes.json` 中的 failure mode
3. 输出：症状清单 + 候选病因排序 + 修复建议（按优先级）

**人工使用**：遇到训练问题时，可按本文档的"调试方法论"逐步排查，或直接查"失败模式分类"定位问题。

---

## 调试方法论

来自 Andy Jones（RL Debugging Systems）与 OpenAI Spinning Up 的核心原则：

### 原则 1：假设有 bug，而不是 RL 难

> "Broken RL code fails silently." — Spinning Up

RL 代码出错时通常不报错，而是表现为"训练不收敛"。**第一反应应该是找 bug，而不是调超参**。

### 原则 2：用 probe environment 验证 pipeline

Andy Jones 的 probe environments（按复杂度递增）：
1. **常数 reward**：单 obs 单 action，reward=1。验证 value function 能否学到 1/(1-γ)。
2. **依赖 action 的 reward**：reward = action。验证 policy 能否选对方向。
3. **依赖 obs 的 reward**：reward = obs。验证观察-动作-奖励的对应关系。
4. **延迟 reward**：reward 在 N 步后给。验证 GAE / discount 计算。

如果 probe env 都跑不通，是 pipeline bug，不是超参问题。

### 原则 3：用 reference implementation 对照

先用 SB3 / CleanRL 的 PPO 在 CartPole / Pendulum 上验证你的训练 loop。能跑通再换自己的 env。**不要**在自己 env 上从头调。

### 原则 4：检查数据 pipeline

常见 bug（按发生频率）：
- obs / action / reward / done 错位（time-shift）
- done 信号错误（timeout 应 truncated=True，不是 done=True）
- advantage 符号反了
- log-prob 维度不对（broadcast 错误，SB3 提到 "broadcast mistake fails silently"）
- 观测未归一化
- reward 未归一化导致梯度爆炸

### 原则 5：一次只改一个变量

调参时**不要**同时改 lr、clip、batch_size。一次改一个，观察效果。否则无法归因。

---

## 失败模式分类

按病因类别分 9 类，共 15 个 failure mode：

### 1. Reward 设计类（reward_design）
- `reward_hacking` — reward 上升但行为不符合任务意图
- `reward_signal_too_sparse` — reward 长期为 0，无学习信号
- `reward_scale_imbalance` — 多 term 量级不匹配，某项主导

### 2. 训练稳定性类（training_instability）
- `policy_collapse` — policy 突然退化不恢复
- `gradient_explosion` — 梯度飙升，loss 变 NaN
- `gradient_vanishing` — 梯度消失，policy 不更新
- `training_divergence` — 训练全程不收敛

### 3. 探索失败类（exploration_failure）
- `entropy_collapse` — policy 过早确定性化
- `local_optimum` — 卡在次优解

### 4. Critic 失败类（critic_failure）
- `value_function_divergence` — critic 不收敛

### 5. 数据管道类（data_pipeline）
- `observation_normalization_issues` — 观测未归一化

### 6. 行为质量类（behavior_quality）
- `action_rate_jittering` — 动作高频抖动，真机不可部署

### 7. 多任务类（multi_task）
- `catastrophic_forgetting` — 学新忘旧

### 8. 部署类（deployment）
- `sim_to_real_gap` — 仿真到真实迁移失败

### 9. 性能类（performance）
- `sample_inefficiency` — 学得太慢

---

## 常见症状速查

| 症状 | 首先怀疑的 failure mode |
|------|------------------------|
| reward 上升但行为不对 | reward_hacking |
| reward 突然下降不恢复 | policy_collapse |
| loss 出现 NaN/Inf | gradient_explosion |
| grad_norm 持续 <1e-4 | gradient_vanishing |
| entropy 前 100 iter 降到 ~0 | entropy_collapse |
| value loss 不收敛 | value_function_divergence |
| reward 长期平台 | local_optimum |
| reward 长期为 0 | reward_signal_too_sparse |
| 训练完全不收敛 | training_divergence 或 observation_normalization_issues |
| reward 正常但动作抖 | action_rate_jittering |
| 仿真好真机差 | sim_to_real_gap |

---

## 数值阈值的注意事项

`failure_modes.json` 中的数值阈值分两类：
- **[文献]**：有明确出处（如 PPO target_kl=0.01 来自 Spinning Up）
- **[经验]**：来自社区默认 config 或 blog（如 entropy<0.1 为异常）

**诊断引擎设计原则**：优先使用**相对变化**作为触发条件，绝对阈值仅作辅助。例如：
- ✅ "KL 单步增量超过训练均值 5-10 倍" — 相对变化，鲁棒
- ⚠️ "KL > 0.05" — 绝对阈值，不同任务可能不同

原因：不同任务的 reward 量级、entropy 范围差异大，绝对阈值会误报。

---

## 关键来源

- [OpenAI Spinning Up](https://spinningup.openai.com/) — RL 入门与调试哲学
- [Stable-Baselines3 RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) — 工程实践
- [Stable-Baselines3 Dealing with NaNs](https://stable-baselines3.readthedocs.io/en/master/guide/checking_nan.html) — 数值问题排查
- [Andy Jones — Debugging RL Systems](https://andyljones.com/posts/rl-debugging.html) — probe environments 方法论
- [The 37 Implementation Details of PPO](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) — PPO 实现细节
- [PPO 原论文 (Schulman 2017)](https://arxiv.org/abs/1707.06347)
- [SAC 原论文 (Haarnoja 2018)](https://arxiv.org/abs/1801.01290)
- [Ng 1999 — Potential-based shaping](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)
- [OpenAI — Faulty Reward Functions in the Wild](https://openai.com/index/faulty-reward-functions/)
- [legged_gym](https://github.com/leggedrobotics/legged_gym) — 四足训练默认 reward scales
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [Lee et al. 2020 — ANYmal sim-to-real](https://arxiv.org/abs/2010.11251)
- [Hwangbo et al. 2019 — ANYmal agile locomotion](https://arxiv.org/abs/1901.08652)

每个 failure mode 的完整数据（症状、病因、修复、验证）见 `resources/failure_modes.json`。
