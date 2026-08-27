# Reward 设计模式知识库

> 本文档是 Isaac Lab reward 函数设计的知识库。结构化数据见 `resources/reward_patterns.json`，本文档提供设计哲学、模式分类与组合启发式。

## 如何使用

- **生成 reward**：`scripts/reward_synthesizer.py` 会从 `reward_patterns.json` 检索 pattern 并组合
- **理解设计**：阅读本文档了解为什么这样设计、何时用哪个 pattern
- **调参参考**：每个 pattern 的权重范围与坑都有说明

---

## Reward 组合哲学

Isaac Lab 的 reward 是**多 term 加权求和**。设计一个好的 reward 配置等同于回答三个问题：

1. **主目标是什么？** → 选 1-2 个 task_reward（如 velocity tracking）
2. **需要约束什么行为？** → 加 stability / energy / smoothness penalty
3. **量级如何平衡？** → 调权重使各 term 贡献均衡（无单 term 占比 >50%）

**核心原则**：
- 主目标权重 1.0-2.0，主导总奖励
- 惩罚项权重为负，量级远小于主目标（penalty : reward ≈ 0.01-0.5）
- 能耗 penalty（如 torques）量级极小（-1e-5）因为物理量本身量级大
- 可选 penalty 用 weight=0.0 保留在 config 中便于实验切换

---

## Pattern 分类

### 1. Task Reward（任务奖励，正权重）

主导项，定义任务目标。

| Pattern | 权重 | 用途 |
|---------|------|------|
| `linear_velocity_tracking` | 1.0 | 跟踪前进/侧向速度指令 |
| `angular_velocity_tracking` | 0.5 | 跟踪偏航角速度指令 |
| `feet_air_time` | 0.125-1.0 | 鼓励有节奏步态（非拖步） |

### 2. Stability Penalty（稳定性惩罚，负权重）

保持机器人直立与航向稳定。

| Pattern | 权重 | 用途 |
|---------|------|------|
| `flat_orientation_l2` | 0.0 ~ -10.0 | 惩罚机身倾斜（投影重力 xy） |

### 3. Energy Penalty（能耗惩罚，负权重）

促进高效低能耗步态，sim-to-real 必需。

| Pattern | 权重 | 用途 |
|---------|------|------|
| `joint_torques_l2` | -1e-5 ~ -1e-4 | 惩罚关节力矩 |
| `action_rate_l2` | -0.01 | 惩罚动作变化率（平滑） |

### 4. Smoothness Penalty（平滑性惩罚，负权重）

减少高频抖动，保护硬件。

| Pattern | 权重 | 用途 |
|---------|------|------|
| `action_rate_l2` | -0.01 | 一阶平滑（Δa） |
| (可选) `action_smoothness_2` | -1e-3 | 二阶平滑（Δ²a，来自 Walk These Ways） |

### 5. Gait Reward（步态奖励，正权重）

鼓励特定步态模式。

| Pattern | 权重 | 用途 |
|---------|------|------|
| `feet_air_time` | 0.125-1.0 | 鼓励 Trot 步态 |
| (高级) `tracking_contacts_shaped` | varies | 基于时钟的显式步态成型（Walk These Ways） |

### 6. Safety Penalty（安全惩罚，负权重）

防止机械限位冲击与摔倒。

| Pattern | 权重 | 用途 |
|---------|------|------|
| `joint_pos_limits` | 0.0 ~ -10.0 | 惩罚越过软位置限位 |
| `is_terminated` | -1.0 ~ -10.0 | 惩罚非超时终止（摔倒等） |

---

## 典型组合配方

### 配方 A：平地四足行走（基础版）

```python
@configclass
class RewardsCfg:
    # 主目标
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"std": math.sqrt(0.25), "command_name": "base_velocity"})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, weight=0.5, params={"std": math.sqrt(0.25), "command_name": "base_velocity"})
    # 稳定性
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # 能耗与平滑
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 步态
    feet_air_time = RewTerm(func=mdp_loc.feet_air_time, weight=0.125, params={"threshold": 0.5, "command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT")})
    # 安全
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0})
```

### 配方 B：崎岖地形四足

在配方 A 基础上：
- 加 `flat_orientation_l2` 权重 -1.0 到 -5.0（强姿态约束）
- 开启 `joint_pos_limits` 权重 -5.0 到 -10.0
- 加 `is_terminated` 权重 -1.0 到 -5.0
- 考虑 `base_height_l2` 保持高度

### 配方 C：高动态任务（跳跃/跨步）

- 提高 `flat_orientation_l2` 权重（防止空中翻滚）
- 加 `body_lin_acc_l2` 惩罚刚体加速度
- 用 Walk These Ways 的时钟步态奖励替代 `feet_air_time`

---

## 组合启发式规则

来自 `reward_patterns.json` 的 `composition_heuristics`：

1. **主任务奖励**（velocity tracking）：weight 1.0-2.0，主导项
2. **辅助任务奖励**（angular tracking）：weight 0.5-1.0，约主项一半
3. **稳定性惩罚**（orientation, ang_vel）：weight -0.05 到 -2.0
4. **能耗惩罚**（torques）：weight -1e-5 到 -1e-4（量级小因 torque 量大）
5. **平滑性惩罚**（action_rate, joint_acc）：weight -0.01 到 -1e-3
6. **安全惩罚**（joint_pos_limits, is_terminated）：weight -1.0 到 -10.0（按需开启）
7. **步态奖励**（feet_air_time）：weight 0.125-1.0（取决于实现滤波）
8. **禁用可选项**：weight=0.0（保留在 config 中便于实验）
9. **per-step 总奖励量级目标**：[-1, 1] 区间（Andy Jones 经验）

---

## 常见陷阱总览

1. **only_positive_rewards 截断**：legged_gym 默认开启，会把负 penalty 截断为 0 导致惩罚失效。解决：关闭此选项，或在截断后单独添加 penalty。

2. **量级失衡**：torques 默认 -1e-5 不是 bug，是因为 τ² 量级大。手动调时不要盲目改 1 个数量级以上。

3. **σ 选择**：velocity tracking 的 σ=0.25 是验证过的默认值。过小稀疏学习慢，过大策略懒散。

4. **sum vs mean**：joint_torques_l2 用 sum 不是 mean。换 mean 会破坏量级平衡，需重新调权重。

5. **feet_air_time 零指令置零**：防止原地踏步刷奖励。这是设计意图，不是 bug。

6. **flat_orientation_l2 默认关闭**：因为 lin_vel_z_l2 + ang_vel_xy_l2 已隐式约束姿态。开启前先确认是否真需要。

---

## 关键来源

- [legged_gym (ETH RSL)](https://github.com/leggedrobotics/legged_gym) — 四足训练事实标准代码库
  - [legged_robot_config.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py) — 默认 reward scales
  - [legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py) — reward 函数实现
- [IsaacLab](https://github.com/isaac-sim/IsaacLab) — 官方框架
  - [rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/envs/mdp/rewards.py) — 内置 reward 函数库
  - [velocity_env_cfg.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py) — 完整 RewardsCfg 示例
  - [velocity/mdp/rewards.py](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/mdp/rewards.py) — 任务级扩展
- [IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) — NVIDIA 官方 envs
- [Walk These Ways (Margolis CoRL 2023)](https://github.com/Improbable-AI/walk-these-ways) — 步态成型与多行为控制
- 论文：Lee et al. 2020 (ANYmal terrain), Hwangbo et al. 2019 (ANYmal agile), Margolis & Agrawal 2023 (Walk These Ways)

每个 pattern 的完整数据（公式、代码、权重、坑）见 `resources/reward_patterns.json`。
