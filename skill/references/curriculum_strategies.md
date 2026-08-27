# Curriculum Learning Strategies for Isaac Lab

> 知识库文档：Isaac Lab 中的课程学习（curriculum learning）设计模式。涵盖 Isaac Lab `CurriculumTerm` API、legged_gym 地形课程、Walk These Ways 步态课程，以及常见陷阱。

---

## 一、为什么需要课程学习

RL 训练的核心困难是**探索效率**：稀疏奖励 + 高维动作空间 → 随机探索几乎不可能命中目标。课程学习通过**逐步增加任务难度**让 agent 在简单环境中先学会基本技能，再迁移到难环境。

**典型场景**：
- 四足从平地 → 简单地形 → 复杂地形（台阶、斜坡、崎岖）
- 机械臂从近距离抓取 → 远距离精确抓取
- 速度跟踪从恒定 0.5 m/s → 随机指令 [-1, 1] m/s

**不需要课程的场景**：
- 任务本身简单（如恒定速度平地行走）
- 奖励密集且单峰（dense + unimodal）
- 训练时间预算充足（curriculum 本身有探索成本）

---

## 二、Isaac Lab CurriculumTerm API

Isaac Lab 在 manager-based env 中通过 `CurriculumTerm` 注册课程函数：

```python
from isaaclab.managers import CurriculumTerm

@configclass
class CurriculumCfg:
    # 课程函数签名: fn(env, env_ids) -> dict[str, float]
    # 返回的 dict 会被记入 tensorboard，便于监控
    terrain_levels = CurriculumTerm(
        func=mdp.terrain_levels,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
```

**关键约定**：
- 函数签名固定：`fn(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> dict[str, float]`
- `env_ids` 是本次需要重置的 env 索引（通常基于 terminate buffer）
- 函数**修改 env 状态**（如调整地形 level、command 范围），返回值仅作日志
- 调用时机：每个 reset 周期，在 env reset 之后、第一次 step 之前

---

## 三、经典课程模式

### 3.1 地形课程（legged_gym 风格）

最经典、最有效的 locomotion 课程。来自 legged_gym / IsaacLab velocity config。

**思路**：
- 地形按难度分成 N 个 level（如 10 级，从平地到 0.4m 台阶）
- 每个 env 维护一个 `terrain_level`，初始为 0
- agent 在当前 level 表现好（成功到达终点）→ level +1
- agent 在当前 level 表现差（摔倒）→ level -1
- 重置时根据 level 选择对应地形

**实现要点**（来自 legged_gym `terrain.py::Terrain`）：

```python
def terrain_levels(env, env_ids, asset_cfg):
    robot = env.scene[asset_cfg.name]
    # 计算每个 env 在本次 episode 的移动距离
    distance = torch.norm(
        robot.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2],
        dim=1,
    )
    # 成功标准：移动超过一半轨道长度
    moved_far = distance > env.scene.terrain.cfg.terrain_length * 0.5
    # 失败标准：摔倒（z 高度过低或 contact 异常）
    fell = env.reset_buf[env_ids] & (~env.time_out_buf[env_ids])

    # 调整 level
    terrain_levels = env.scene.terrain.terrain_levels.clone()
    terrain_levels[moved_far] += 1
    terrain_levels[fell] -= 1
    terrain_levels = torch.clamp(terrain_levels, 0, max_level)

    env.scene.terrain.terrain_levels[env_ids] = terrain_levels[env_ids]
    return {"terrain_levels": terrain_levels.mean().item()}
```

**关键参数**：
- `terrain_length`：单块地形长度（通常 8m）
- `max_level`：最大难度等级（通常 10）
- 成功/失败阈值需根据任务调整

**常见陷阱**：
1. **level 切换时 reset 不彻底**：env 切换 level 后必须重新 spawn 机器人在新地形，否则仍在旧地形
2. **成功标准过松**：移动距离阈值过低 → level 上升太快，agent 没真正学会当前难度
3. **失败标准过严**：timeout 也算失败 → level 永远上不去，陷入死循环
4. **多 env 同步问题**：legged_gym 的 terrain 课程是 per-env 的，但 terrain 块是预生成的。确保 `env_origins` 与 `terrain_levels` 同步更新

### 3.2 指令课程（command curriculum）

逐步扩大 command 范围。来自 IsaacLab velocity_env_cfg.py 的 `commands` 配置。

**思路**：
- 初始：`lin_vel_x ∈ [-0.5, 0.5]` m/s
- 每 N 步评估一次：若平均跟踪误差 < 阈值，扩大范围至 `[-1.0, 1.0]`
- 最终：`lin_vel_x ∈ [-1.5, 1.5]` m/s

**实现**：

```python
def velocity_command_curriculum(env, env_ids):
    # 检查最近 N 步的跟踪误差
    if env.common_step_counter < env.cfg.commands.curriculum_steps:
        return {}
    recent_error = env.command_tracker.get_recent_error(window=100)
    if recent_error < env.cfg.commands.curriculum_threshold:
        # 扩大 command 范围
        env.command_manager.get_term("base_velocity").ranges["lin_vel_x"] = [
            -1.0, 1.0
        ]
    return {"command_range": env.command_manager.get_term("base_velocity").ranges["lin_vel_x"][1]}
```

**陷阱**：
1. 课程函数**只扩大不缩小**——一旦扩大就不收回。若扩大后性能下降，应该有回退机制（但 legged_gym 默认没有）
2. 评估窗口太短 → 噪声大，课程不稳定
3. 一次性扩大太多 → agent 无法适应

### 3.3 步态课程（Walk These Ways 风格）

来自 Margolis & Agrawal CoRL 2023。逐步收紧步态约束。

**思路**：
- 阶段 1（0-200 iters）：无步态约束，agent 自由探索
- 阶段 2（200-1000 iters）：引入 contact schedule，但允许 ±50% 偏差
- 阶段 3（1000+ iters）：收紧到 ±10% 偏差

**实现**：通过 reward weight 的 schedule：

```python
# 在 env 中维护 gait_curriculum_stage
def gait_curriculum(env, env_ids):
    stage = 0
    if env.common_step_counter > 200 * env.num_envs:
        stage = 1
    if env.common_step_counter > 1000 * env.num_envs:
        stage = 2

    # 动态调整 gait reward 的 weight
    weights = {0: 0.0, 1: 0.5, 2: 2.0}
    env.reward_manager.get_term("gait_tracking").weight = weights[stage]
    return {"gait_stage": stage}
```

**适用场景**：需要特定步态（Trot、Bound、Pace）的任务。不适合自由步态探索任务。

### 3.4 观测课程（curriculum on observations）

逐步增加观测维度。较少见但有效。

**思路**：
- 阶段 1：只给本体感受（joint pos/vel）
- 阶段 2：加入高度扫描（height scan）
- 阶段 3：加入前向视觉（depth camera 或 occupancy）

**实现**：通过 `ObservationsCfg` 的动态启用。需要自定义 observation manager。

**陷阱**：
1. 观测维度变化 → policy 网络输入维度变化 → 不能直接加载旧 policy
2. 通常需要重新训练或用 padding 兼容
3. 这是**最复杂**的课程类型，仅在确有必要时使用

### 3.5 障碍物密度课程

机械臂或导航任务常见。逐步增加障碍物数量或密度。

```python
def obstacle_density_curriculum(env, env_ids):
    success_rate = env.task_tracker.get_success_rate(window=200)
    if success_rate > 0.8:
        env.scene.obstacle_manager.density += 0.05
    elif success_rate < 0.3:
        env.scene.obstacle_manager.density -= 0.05
    env.scene.obstacle_manager.density = max(0.0, min(1.0, env.scene.obstacle_manager.density))
    return {"obstacle_density": env.scene.obstacle_manager.density}
```

---

## 四、课程设计的通用原则

### 4.1 阶段评估窗口要够长

**反例**：每 10 步评估一次 → 噪声主导，课程震荡
**正例**：每 100-500 步评估一次，或基于 episode 数（每 50 episode）

### 4.2 难度递增要保守

**反例**：成功一次就升级 → agent 没真正掌握，降级循环
**正例**：连续成功 N 次（如 5 次）才升级

### 4.3 永远保留降级机制

升级容易降级难——agent 在新难度下失败时必须有降级路径。否则陷入"升级 → 失败 → 无法降级 → 卡住"。

### 4.4 课程指标要可观测

课程函数返回的 dict 必须记入 tensorboard。调试课程问题时，这些曲线是第一手证据：
- `terrain_levels`（mean）
- `command_range`
- `success_rate`
- `curriculum_stage`

如果训练时这些曲线不动，说明课程函数没生效或条件没触发。

### 4.5 课程与 reward 不要冲突

**反例**：reward 鼓励快速移动，但课程在 agent 学会快速移动之前就提高难度 → agent 永远在挣扎
**正例**：reward 和课程同步——先让 agent 在简单环境获得正反馈，再提高难度

---

## 五、调试课程问题

### 5.1 课程不推进

**症状**：`terrain_levels` 曲线一直是 0
**排查**：
1. 检查 `CurriculumTerm` 是否注册到 `CurriculumCfg`
2. 检查课程函数是否真的被调用（加 print 或在返回 dict 里放一个常量）
3. 检查成功/失败条件是否触发了 level 变更（可能是阈值设错）
4. 检查 `env_ids` 是否非空——若所有 env 都没 reset，课程函数可能跳过

### 5.2 课程推进过快

**症状**：`terrain_levels` 几个 iteration 就冲到 max
**排查**：
1. 成功标准过松——调高阈值
2. 没有连续成功要求——加 `consecutive_success` 计数
3. 检查是否误把 timeout 当成功

### 5.3 课程震荡

**症状**：`terrain_levels` 上下震荡，agent 在某两个 level 间反复
**排查**：
1. 降级条件太松——降级应该比升级保守
2. 评估窗口太短——加大 window
3. 该 level 难度跨度过大——细分 level

### 5.4 性能随课程推进下降

**症状**：reward 上升后随 level 上升而下降
**排查**：
1. 这是**正常的**——新难度本来就难。区分"暂时下降后恢复"和"持续下降不恢复"
2. 若持续下降：课程推进过快，或 reward 设计不支持新难度的行为
3. 检查是否需要 reward curriculum 同步调整（如在新难度下暂时降低 penalty）

---

## 六、Isaac Lab 特定注意事项

### 6.1 CurriculumTerm 调用时机

`CurriculumTerm` 在 `env.reset()` 之后、第一次 `env.step()` 之前调用。这意味着：
- 课程函数看到的是**已 reset 的状态**
- 课程函数对 env 的修改（如调整 terrain level）会影响下一步的观测
- 不要在课程函数里调用 `env.reset()`——会无限递归

### 6.2 多 env 并行的课程

Isaac Lab 默认多 env 并行（num_envs=4096 等）。课程通常是 per-env 的：
- 每个 env 有自己的 `terrain_level`
- 课程函数收到的 `env_ids` 是本次 reset 的 env 子集
- 修改时只改 `env_ids` 对应的 env，不要全量修改

**陷阱**：直接 `env.terrain_levels = new_levels` 会覆盖所有 env，破坏其他 env 的进度。必须用 `env.terrain_levels[env_ids] = new_levels[env_ids]`。

### 6.3 课程与 DR 的交互

地形课程 + 地形 DR（随机摩擦/质量）会互相影响：
- 高 level 地形 + 高 DR → 过难
- 低 level 地形 + 低 DR → 过易

通常做法：DR 范围随 level 增长（高 level 用更宽 DR）。但这增加复杂度，建议先单独调好课程再叠加 DR。

---

## 七、参考实现与源码

- `IsaacLab/source/isaaclab/isaaclab/envs/mdp/curriculum.py` — 内置课程函数
- `IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config` — 完整 locomotion 课程配置
- `legged_gym/legged_gym/envs/base/legged_robot.py::_reset_terrain` — legged_gym 地形课程
- `Walk-These-Ways/wtw/agent/wtw_agent.py` — 步态课程与 reward schedule

---

## 八、本 skill 中课程的现状

**MVP 范围**：curriculum 设计是 roadmap 项，不在 MVP 中实现。但：
- 本文档提供完整的设计模式与陷阱清单，可在 SKILL.md 的"设计咨询"场景作为 LLM 参考
- 后续可作为 module 3 开发（curriculum_designer.py）
- 示例 `examples/quadruped_locomotion/` 可包含一个简单的 command curriculum 作为演示
