# 机器人配置参考

> 本文档是 Isaac Lab RL 训练常见机器人的配置参考，覆盖 quadruped / biped / manipulator 三大类。
> 与本 skill 的 `dr_advisor` 模块联动：advisor 推荐的 DR 范围按机器人质量等级（small/medium/large）区分，本文档解释为什么这么分。

## 如何使用

- **场景 1（自动）**：调 `dr_advisor.recommend(robot_type="quadruped_medium", ...)` 会返回该机器人的 DR term 推荐范围
- **场景 2（手动）**：开新训练项目时，按本文档找到你的机器人型号，查推荐配置起步

---

## 质量等级划分

本 skill 的 `dr_patterns.json` 把机器人按质量分 5 个 profile：

| Profile | 代表型号 | 质量 (kg) | DR mass_range | 含义 |
|---------|---------|-----------|---------------|------|
| `quadruped_small` | Unitree A1, Go1 | 4.5-12 | ±2 kg | 小型四足，payload 敏感 |
| `quadruped_medium` | Unitree Go2, ANYmal C | 12-25 | ±3 kg | 中型四足，工业级 |
| `quadruped_large` | ANYmal B, Spot | 25-50 | ±5 kg | 大型四足，自带 payload |
| `biped_small` | Cassie, Berkeley Humanoid | 5-15 | ±1 kg | 小型双足 |
| `manipulator_arm` | UR5e, Franka Panda | 3-30 | ±0.5-2 kg (link) | 机械臂，link-level |

**为什么按质量分**：DR 的 `robot_mass` / `center_of_mass_offset` / `motor_strength` 范围都跟机器人物理尺寸相关。给 15kg 的 Go2 随机化 ±5kg mass 等于让它载一个 1/3 体重的额外 payload，训练出来的 policy 会过鲁棒、收敛慢；给 50kg 的 ANYmal B 随机化 ±2kg 又太轻微、达不到 sim-to-real 效果。

---

## Quadruped (Small) — A1 / Go1

| 项目 | 推荐值 | 出处 |
|------|--------|------|
| 质量 | 4.5 kg (A1) / 12 kg (Go1 EDU) | Unitree 官方 |
| 自由度 | 12 (4 legs × 3 joints) | |
| 控制 freq | 50 Hz | legged_gym 默认 |
| `dt` (sim) | 0.005 s (200 Hz) | sim 比 control 快 4 倍 |
| action scale | [-1, 1] → 12 joint positions | 标准做法 |
| `num_envs` | 4096 | legged_gym A1 example |
| `episode_length_s` | 20 s (1000 steps) | |
| `action_clip` | 5.0 (rad/s) | legged_gym 默认 |

**DR 推荐范围**（来自 `dr_patterns.json`）：
- `robot_mass`: ±2 kg
- `center_of_mass_offset`: ±0.05 m
- `motor_strength`: ±10%
- `ground_friction`: 0.3-1.1
- `obs_noise`: 0.001-0.01

**特殊注意**：A1 / Go1 关节力矩有限 (~33.5 Nm)，加 `joint_torques_l2` penalty 时要小心 scale，否则容易主导 reward。

---

## Quadruped (Medium) — Go2 / ANYmal C

| 项目 | 推荐值 | 出处 |
|------|--------|------|
| 质量 | 15 kg (Go2) / 25 kg (ANYmal C) | Unitree / ANYbotics 官方 |
| 自由度 | 12 | |
| 控制 freq | 50-100 Hz | |
| `dt` (sim) | 0.005 s | |
| `num_envs` | 4096 | |
| `episode_length_s` | 20-24 s | |

**DR 推荐范围**：
- `robot_mass`: ±3 kg
- `center_of_mass_offset`: ±0.1 m
- `motor_strength`: ±15%
- `ground_friction`: 0.3-1.1
- `obs_noise`: 0.001-0.02

**特殊注意**：Go2 是当前最常用的研究平台（2024 后），`dr_advisor` 默认推荐 quadruped_medium。Isaac Lab 的 `manager_based/locomotion/velocity` 示例就是用 Go2 URDF。

---

## Quadruped (Large) — ANYmal B / Spot

| 项目 | 推荐值 | 出处 |
|------|--------|------|
| 质量 | 30-50 kg | ANYbotics / Boston Dynamics 官方 |
| 自由度 | 12 | |
| 控制 freq | 40-50 Hz | ANYmal 默认 40 Hz |
| `dt` (sim) | 0.005 s | |
| `num_envs` | 2048-4096 | 大机器人 sim 慢，env 数可降 |
| `episode_length_s` | 20-25 s | |

**DR 推荐范围**：
- `robot_mass`: ±5 kg
- `center_of_mass_offset`: ±0.15 m
- `motor_strength`: ±20%
- `ground_friction`: 0.3-1.1
- `obs_noise`: 0.002-0.02

**特殊注意**：大机器人惯性大，joint_acc penalty 量级要降一档；建议开 `action_delay` DR（5-15ms）模拟控制延迟。

---

## Biped (Small) — Cassie

| 项目 | 推荐值 | 出处 |
|------|--------|------|
| 质量 | 13.7 kg | Agility Robotics 官方 |
| 自由度 | 24 (含 toe joint) | |
| 控制 freq | 50-200 Hz | Cassie 默认 200 Hz |
| `dt` (sim) | 0.0025 s (400 Hz) | 双足 sim freq 高 |
| `num_envs` | 1024-2048 | |
| `episode_length_s` | 10-15 s | 双足易摔，episode 短 |

**DR 推荐范围**：
- `robot_mass`: ±1 kg
- `center_of_mass_offset`: ±0.03 m
- `motor_strength`: ±10%

**特殊注意**：Cassie 是 underactuated（脚踝没有 pitch motor），双足训练 policy collapse 风险高，建议：
- `entropy_coef` 从 0.05 起步（不是 0.01）
- 必加 `terrain_curriculum`，从平地起步
- 终止条件（base_height < 0.5m 或 base tilt > 30°）必须正确

参考实现：Cassie in IsaacLabEnvs、Agility Robotics 公开 gym。

---

## Manipulator (Arm) — UR5e / Franka Panda

| 项目 | 推荐值 | 出处 |
|------|--------|------|
| 质量 | 3-30 kg (link-level) | UR / Franka 官方 |
| 自由度 | 6 (UR5e) / 7 (Franka) | |
| 控制 freq | 20-50 Hz | 机械臂通常慢于四足 |
| `dt` (sim) | 0.01 s (100 Hz) | |
| `num_envs` | 1024-2048 | |
| `episode_length_s` | 5-10 s | |

**DR 推荐范围**：
- `link_mass`: ±0.5 kg（每个 link）
- `joint_damping_friction`: ±20%
- `obs_noise`: 0.005-0.02
- `action_delay`: 5-20 ms

**特殊注意**：
- 机械臂 reward 信号稀疏（reach/grasp 成功才有 reward），必加 `curriculum`
- `gamma` 偏低（0.95-0.98）比 0.99 更稳
- 推荐用 sparse + dense reward 混合：dense = `distance_to_goal`，sparse = `grasp_success`

参考实现：IsaacLab `manager_based/manipulation/reach/`。

---

## 选型决策树

```
你的机器人是什么？
├── 有腿（四足或双足）
│   ├── 4 条腿
│   │   ├── 4-12kg → quadruped_small (A1, Go1)
│   │   ├── 12-25kg → quadruped_medium (Go2, ANYmal C)  ← 最常见
│   │   └── 25-50kg → quadruped_large (ANYmal B, Spot)
│   └── 2 条腿
│       └── 5-15kg → biped_small (Cassie)
└── 机械臂
    └── 3-30kg → manipulator_arm (UR5e, Franka)
```

---

## 各机器人的训练特点对比

| 特性 | Quadruped | Biped | Manipulator |
|------|-----------|-------|-------------|
| Episode 长度 | 长 (20s) | 短 (10s, 易摔) | 短 (5-10s) |
| Reward signal | 稠密 (velocity tracking) | 稠密但易 collapse | 稀疏 (reach/grasp) |
| LR 推荐 | 1e-3 | 5e-4 | 5e-4 |
| Entropy_coef | 0.01 | 0.05 | 0.05 |
| Curriculum 必要性 | 中（rough terrain 才需要） | 高（必加 terrain） | 高（必加 distance） |
| Sim freq | 200 Hz | 400 Hz | 100 Hz |

---

## 参考实现位置

Isaac Lab `manager_based/` 下的官方示例（在 `source/extensions/isaac.lab.tasks/isaac/lab/tasks/manager_based/`）：

- `locomotion/velocity/`：quadruped 行走，含 Anymal/Spot/Cassie 配置
- `manipulation/reach/`：机械臂 reach 任务
- `navigation/`：导航任务

每个示例都有 `__init__.py` 入口 + 多种机器人变体（`<robot>_env_cfg.py`）。开新项目时建议复制最接近的变体再改。

---

## 参考文献

- **Unitree A1 / Go1 / Go2**：unitree 官方文档，质量与 URDF 来自 SDK 包
- **ANYmal B / C**：ANYbotics 公开论文 + IsaacLab 内置 asset
- **Boston Dynamics Spot**：Boston Dynamics 官方 SDK + Spot RL 论文
- **Agility Cassie**：Agility Robotics 学术合作，Cassie RL 论文（Heiden et al. 2021 等）
- **UR5e / Franka Panda**：Isaac Lab `manager_based/manipulation/reach/` 默认 asset
- **legged_gym**：NVIDIA-Omniverse/legged_gym，Anymal/AnymalC/A1 训练基线
- **Walk These Ways**：Bellegarda & Ijspeert 2022，多 gait curriculum 参考
