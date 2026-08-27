# Isaac Lab 训练超参数推荐

> 本文档是 PPO 算法在 Isaac Lab 机器人 RL 任务上的超参数推荐。
> 推荐值来自 legged_gym、IsaacGymEnvs、IsaacLab 官方示例与 PPO 论文，并区分**文献值**（有明确出处）与**经验值**（社区默认）。

## 如何使用

- **场景 1（自动）**：本 skill 的诊断模块（`diagnosis_engine`）在识别失败模式后，会从 `failure_modes.json` 给出针对性超参调整建议（见 `training_failure_modes.md`）
- **场景 2（手动）**：开新训练项目时，按本文档"起步配置"选超参；训练遇问题时查"调参方向"段

---

## PPO 起步配置（locomotion 任务）

下面是四足机器人（A1/Go2 量级，~15kg）平地前进任务的 PPO 起步配置。来自 legged_gym 的 Anymal/AnymalC 基线 + IsaacLab `manager_based` 示例：

| 超参 | 推荐值 | 出处 | 备注 |
|------|--------|------|------|
| `clip_param` | 0.2 | [文献] PPO 原论文 Schulman et al. 2017 | 标准值，绝大多数实现不调 |
| `learning_rate` | 1e-3 → 5e-4（decay） | [经验] legged_gym 默认 | 起步 1e-3，~1000 iter 后降到 5e-4 |
| `num_learning_epochs` | 5 | [经验] legged_gym/IsaacLab 默认 | 范围 4-10；超过 10 易过拟合 |
| `num_mini_batches` | 4 | [经验] legged_gym 默认 | 范围 2-8；增大显存足够时可调 |
| `value_loss_coef` | 1.0 | [文献] PPO 原论文 | 一般不动 |
| `entropy_coef` | 0.01 | [经验] legged_gym/IsaacLab 默认 | 范围 0.001-0.1；本 skill 的诊断模块会建议调这个 |
| `max_grad_norm` | 1.0 | [文献] SB3/legged_gym/IsaacGymEnvs 默认 | grad clip，防梯度爆炸 |
| `target_kl` | 0.01 | [文献] Spinning Up / Schulman blog | KL early stop；超过此值时停止 update |
| `gamma` (discount) | 0.99 | [文献] PPO 标准 | locomotion 一般 0.99，manipulation 可 0.95-0.98 |
| `lam` (GAE) | 0.95 | [文献] GAE paper | 通常不动 |
| `horizon_length` | 24 | [经验] legged_gym 默认 | 每次 rollout 步数 × num_envs |
| `num_envs` | 4096 | [经验] Isaac Lab GPU 训练标配 | 显存不够时降到 2048 或 1024 |

---

## 调参方向（按症状）

### 1. reward 持续上升但行为不收敛

**大概率**：reward 设计有漏洞（reward hacking）或 PPO 更新步太大（KL 单次更新超 target_kl）。

**调参方向**：
- 先确认 `target_kl` 启用（默认应开启），值 0.01
- 降 `learning_rate`：1e-3 → 5e-4 → 1e-4
- 增 `entropy_coef`：0.01 → 0.05（鼓励探索，避免早收敛到局部最优）
- 检查 reward 各项 weight 是否冲突（用 `reward_validator` 静态校验）

### 2. reward 平台不涨（局部最优）

**大概率**：探索不足。

**调参方向**：
- 增 `entropy_coef`：0.01 → 0.05 → 0.1
- 检查 reward 信号是否过稀疏（考虑加 curriculum，见 `curriculum_strategies.md`）
- 增 `num_envs`：更多并行采样拓宽 state 覆盖

### 3. reward 突然崩溃（policy collapse）

**大概率**：policy 更新爆炸 / value function 发散。

**调参方向**：
- 确认 `max_grad_norm=1.0` 启用
- 降 `learning_rate`：1e-3 → 1e-4
- 降 `num_learning_epochs`：5 → 2-3
- 检查 observation 是否有 NaN/Inf（normalize 没生效？）

### 4. grad_norm 持续爆炸（>100）

**大概率**：reward 量级太大 / value loss 主导。

**调参方向**：
- 加 `max_grad_norm=1.0`
- 归一化 reward（RunningMeanStd / VecNormalize）
- 检查 reward 各项量级，把 reward scale 控制 [-10, 10] 区间

### 5. entropy 过早塌缩（< 0.1 in 200 iter）

**大概率**：探索不够 / policy 过于自信。

**调参方向**：
- 增 `entropy_coef`：0.01 → 0.05 → 0.1
- 检查 action space 是否归一化（连续 action 应是 [-1, 1]）
- 降 `num_learning_epochs`：5 → 2-3

---

## 不同任务的差异

### Locomotion（四足/双足行走）

- **lr 偏高**：1e-3 起步，因为 reward signal 强（velocity tracking）
- **horizon 长**：24 步，因为 episode 较长（5s × 50Hz = 250 步）
- **entropy_coef 适中**：0.01，避免过早塌缩但不过度探索

### Manipulation（机械臂操作）

- **lr 偏低**：5e-4 起步，因为 reward signal 稀疏（reach/grasp 成功才有 reward）
- **horizon 短**：8-16 步，episode 短
- **entropy_coef 偏高**：0.05，鼓励探索稀疏 reward 空间
- **gamma 偏低**：0.95-0.98，长 horizon 任务才用 0.99

### Navigation（导航）

- **lr 适中**：5e-4 ~ 1e-3
- **horizon 中等**：16-24 步
- **gamma 偏高**：0.99，因为目标达成有延迟
- **必加 curriculum**：从短距离起步，逐步扩展（见 `curriculum_strategies.md`）

---

## 学习率调度

legged_gym 默认用 piecewise decay：

```python
learning_rate = schedule.PiecewiseLinear(
    boundaries=[1000, 2000],  # iter
    values=[1e-3, 5e-4, 1e-5],
)
```

**经验**：
- 1-1000 iter：1e-3（快速学习）
- 1000-2000 iter：5e-4（精调）
- 2000+ iter：1e-5（稳定收敛）

---

## 常见陷阱

1. **不开 grad clip**：训练 1000+ iter 后 grad_norm 突然爆炸，policy 崩溃
2. **不开 KL early stop**：单次更新步太大，policy 跳出稳定区间
3. **reward 不归一化**：value function 难收敛
4. **observation 不归一化**：相同问题，且对 DR 不鲁棒
5. **entropy_coef=0**：policy 过早塌缩，行为多样性丢失
6. **lr 过高**：典型症状是 reward 上升 → 突降 → 永久低水平
7. **horizon 太短**：每次 update 数据不够，policy 更新噪声大

---

## 参考文献

- **PPO 原论文**：Schulman et al. 2017, "Proximal Policy Optimization Algorithms", arXiv:1707.06347
- **GAE**：Schulman et al. 2015, "High-Dimensional Continuous Control Using Generalized Advantage Estimation", arXiv:1506.02438
- **Spinning Up**：OpenAI Spinning Up 文档，target_kl=0.01 的工程解读
- **legged_gym**：NVIDIA-Omniverse/legged_gym 仓库，Isaac Gym 时代的四足训练基线
- **IsaacLab**：isaac-sim/IsaacLab 仓库 `source/extensions/isaac.lab/tasks/manager_based/` 各任务默认配置
- **Walk These Ways**：Bellegarda & Ijspeert 2022，curriculum 设计参考
