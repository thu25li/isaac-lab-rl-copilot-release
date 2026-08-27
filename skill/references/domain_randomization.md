# Domain Randomization for Isaac Lab

> 知识库文档：sim-to-real 的核心技术——Domain Randomization (DR)。涵盖 Isaac Lab `EventTerm` API、legged_gym DR 配置、物理参数随机化模式、观测噪声注入，以及常见陷阱。

---

## 一、为什么需要 Domain Randomization

**核心问题**：仿真器是真实世界的**近似**。摩擦系数、质量分布、电机延迟等物理参数在仿真中是确定值，但在真机上有不确定性。若 agent 过拟合仿真器特定参数，迁移到真机会失效（sim-to-real gap）。

**DR 的核心思想**：训练时**主动随机化**这些不确定参数，让 policy 学会在**参数分布**上鲁棒，而非过拟合某个点值。

**理论基础**：Ng et al. 2018（sim-to-real with DR）、Peng et al. 2018（dynamics randomization）、Tan et al. 2018（system identification + DR）。

**何时需要 DR**：
- 几乎所有 sim-to-real 任务（除非只评测仿真性能）
- 鲁棒性测试（即使不部署真机，DR 提升泛化能力）
- 多机器人部署（同一 policy 跑不同硬件版本）

**何时不需要 DR**：
- 纯仿真 benchmark（如 Isaac Lab 官方 leaderboard）
- 系统辨识已精确建模的真实系统
- 训练时间预算极紧（DR 会降低样本效率）

---

## 二、Isaac Lab DR API：EventTerm

Isaac Lab 通过 `EventTerm` 在 episode 边界（或定期）注入随机化：

```python
from isaaclab.managers import EventTerm, EventTermCfg
from isaaclab.envs.mdp import events as mdp

@configclass
class EventsCfg:
    # Episode 开始时随机化机器人质量
    robot_mass = EventTermCfg(
        func=mdp.add_robot_mass,
        mode="reset",  # "reset" 或 "interval"
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_range": (-1.0, 1.0),  # 相对偏差，单位 kg
        },
    )

    # 每 5 秒随机化摩擦
    ground_friction = EventTermCfg(
        func=mdp.add_articulation_mass,
        mode="interval",
        interval_time_s=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_range": (-0.5, 0.5),
        },
    )
```

**关键概念**：
- `mode="reset"`：每次 env reset 时触发一次（最常用）
- `mode="interval"`：固定时间间隔触发（用于运行中漂移）
- `interval_time_s`：interval 模式的触发间隔

**与 CurriculumTerm 的区别**：
- `EventTerm`：随机化物理参数，不改任务难度
- `CurriculumTerm`：调整任务难度，不随机化物理

---

## 三、经典 DR 模式

### 3.1 物理参数随机化

来自 legged_gym `legged_robot.py::_randomize_rigid_body_props` 与 IsaacLab events.py。

#### 3.1.1 质量随机化

机器人 payload 变化（背摄像头、电池、传感器）是最常见的 sim-to-real 差异。

```python
# legged_gym 风格：相对偏差
robot_mass = EventTermCfg(
    func=mdp.add_robot_mass,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "mass_range": (-2.0, 2.0),  # ±2kg（机器人 12kg → 10-14kg）
    },
)
```

**典型范围**：
- 小型机器人（Anymal C, 12kg）：±1-3kg
- 大型机器人（Spot, 25kg）：±2-5kg
- 机械臂（UR5e）：±0.5kg（末端负载）

**陷阱**：
1. 范围过大 → 训练不收敛（agent 学不会应对极端情况）
2. 范围过小 → sim-to-real 无效
3. 随机化后**未重新计算动力学**（质量变化但惯量没更新）→ 物理不一致。Isaac Lab 的 `add_robot_mass` 已处理，自定义函数需注意

#### 3.1.2 摩擦系数随机化

地面材质差异（草地、水泥、地毯）是主要不确定性来源。

```python
ground_friction = EventTermCfg(
    func=mdp.add_articulation_mass,
    # 实际用 mdp.randomize_rigid_body_material,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "friction_range": (0.3, 1.2),  # 静摩擦系数
        "num_buckets": 16,  # 离散化桶数（提高效率）
    },
)
```

**典型范围**：
- 室内平地：0.5-1.0
- 户外多地形：0.2-1.5
- 极端情况（冰面、油污）：0.1-2.0

**陷阱**：
1. 摩擦 = 0 → 物理不稳定（机器人无限滑动）
2. 摩擦 > 2 → 不真实，policy 学会"粘"在地上
3. legged_gym 用 `num_buckets` 把连续范围离散化，减少 PhysX material 切换开销

#### 3.1.3 电机强度随机化

电机磨损、电池电压下降、驱动器差异。

```python
motor_strength = EventTermCfg(
    func=mdp.randomize_actuator_parameters,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_range": (0.8, 1.2),  # 相对比例
        "damping_range": (0.8, 1.2),
    },
)
```

**关键**：Isaac Lab 的 actuator 模型（隐式 PD 控制）的 stiffness/damping 直接影响关节响应。随机化这俩等价于随机化电机强度。

**典型范围**：±20%（0.8-1.2 比例）。超过 ±30% 训练困难。

#### 3.1.4 关节阻尼与摩擦

关节内部的物理摩擦（非地面摩擦）。

```python
joint_damping = EventTermCfg(
    func=mdp.randomize_joint_parameters,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "damping_range": (0.0, 0.5),  # N·m·s/rad
        "friction_range": (0.0, 0.05),  # N·m
    },
)
)
```

**陷阱**：damping 过大 → 关节响应过慢；friction 过大 → 关节卡死。

### 3.2 观测噪声注入

模拟传感器噪声（IMU 漂移、编码器量化、电流传感器噪声）。

#### 3.2.1 IMU 噪声

```python
# 在 ObservationsCfg 中
@configclass
class ObsCfg:
    @configclass
    class LinVelCfg(ObsTerm):
        func = mdp.base_lin_vel
        noise = OpenNoiseCfg(n_v=0.05)  # 标准差 0.05 m/s

    @configclass
    class AngVelCfg(ObsTerm):
        func = mdp.base_ang_vel
        noise = OpenNoiseCfg(n_v=0.05)  # rad/s
```

**典型噪声标准差**：
- 线速度：0.05-0.1 m/s
- 角速度：0.05-0.1 rad/s
- 关节位置：0.01 rad
- 关节速度：0.05 rad/s

#### 3.2.2 延迟模拟

真机控制有 5-30ms 延迟（通信 + 计算 + 驱动）。

```python
# Isaac Lab 通过 obs 的 history 实现
@configclass
class ObsCfg:
    joint_pos_history = ObsTerm(
        func=mdp.joint_pos,
        history_length=2,  # 取 2 步前的观测
    )
```

**陷阱**：
1. 延迟过大会让训练不收敛（policy 无法关联动作与效果）
2. 延迟固定 vs 随机：随机延迟更接近真机但更难训练。通常先用固定延迟。

### 3.3 外力扰动

模拟风、碰撞、人为推搡。

```python
external_force = EventTermCfg(
    func=mdp.apply_random_external_force,
    mode="interval",
    interval_time_s=5.0,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "force_range": (0.0, 5.0),  # N
        "direction_range": (0, 2 * math.pi),  # 随机方向
    },
)
```

**典型范围**：
- 测试鲁棒性：0-5N
- 极端扰动测试：0-20N（接近机器人自重的 10%）

### 3.4 地形随机化

不仅是 friction，还包括：
- 地形几何（台阶高度、斜坡角度）
- 地形材质（ restitution）
- 障碍物位置

详见 `references/curriculum_strategies.md` 的地形课程——地形课程与地形 DR 常配合使用。

---

## 四、DR 设计原则

### 4.1 随机化范围要"宽到有效，窄到可学"

**反例**：mass_range=(-10, 10) on 12kg robot → 极端值不真实且训练失败
**正例**：mass_range=(-2, 2) on 12kg robot → 真实且训练可行

**经验法则**：先从窄范围开始，训练能收敛后逐步加宽，直到性能下降。下降点就是有效范围上限。

### 4.2 随机化的参数要相互独立

**反例**：同时随机化 friction 和 motor_strength，且两者范围都很大 → 无法定位 sim-to-real 失败原因
**正例**：分阶段——先单独随机化 friction，验证 OK 后再加 motor_strength

### 4.3 DR 与 curriculum 配合

DR 增加任务难度，curriculum 调整任务难度。两者方向一致时有效，冲突时失败：
- **正例**：低 level 地形 + 窄 DR；高 level 地形 + 宽 DR
- **反例**：低 level 地形 + 宽 DR → 训练不收敛

### 4.4 评估时关闭 DR

训练时开 DR，但**评估时关闭**（或用固定 seed 的 DR）才能横向对比不同方法。

```python
# 在 env cfg 中
class MyEnvCfg:
    def __post_init__(self):
        if self.eval_mode:
            self.events = None  # 关闭 DR
```

---

## 五、DR 调试问题

### 5.1 训练不收敛

**症状**：加了 DR 后 reward 一直上不去
**排查**：
1. 缩小 DR 范围到原来的 10%，看是否收敛
2. 单独测试每个 DR 参数（friction、mass、motor），定位元凶
3. 检查是否同时加了 curriculum——可能两者冲突
4. 增大 num_envs（DR 需要更多样本来覆盖分布）

### 5.2 sim-to-real 仍失败

**症状**：DR 训练收敛，但迁移真机仍失败
**排查**：
1. 真机的物理参数是否在 DR 范围内？用 system identification 测量真机实际参数
2. 观测噪声是否真实？真机 IMU 噪声通常比仿真大
3. 是否遗漏了关键参数（如延迟、电机死区）
4. 检查是否 over-fit 到 DR 分布的某个 mode

### 5.3 性能下降过多

**症状**：加 DR 后性能下降明显（如 reward 从 20 降到 10）
**排查**：
1. 这是**正常的**——DR 牺牲性能换鲁棒性。判断"过多"需要基准
2. 经验：性能下降不超过 30% 是合理范围
3. 若下降超过 50%：DR 范围过宽，或 reward 设计不支持鲁棒行为

---

## 六、Isaac Lab 特定注意事项

### 6.1 EventTerm vs CurriculumTerm

两者都是 `ManagerBasedEnv` 的 term，但用途不同：
- `EventTerm`：**修改 env 状态**（物理参数、观测噪声、外力）
- `CurriculumTerm`：**调整任务难度**（地形 level、command 范围）

误用会导致 env 状态不一致。例：把地形难度放 EventTerm 会失去 curriculum 的渐进性。

### 6.2 PhysX material 切换开销

每次随机化 friction 都会创建新 PhysX material。高频随机化（如每 step）会严重拖慢训练。**解决**：
- 用 `num_buckets` 离散化（legged_gym 默认 16 桶）
- 用 `mode="reset"` 而非 `mode="interval"`（reset 频率低）
- 批量随机化所有 env，而非逐个

### 6.3 DR 与多 env 并行

Isaac Lab 多 env 并行时，DR 通常 per-env：
- 每个 env 有自己的 mass、friction 等
- EventTerm 收到 `env_ids`（reset 的 env 子集），只随机化这些 env
- 随机化函数必须支持 `env_ids` 参数

### 6.4 自定义 DR 函数

若内置函数不够，可自定义：

```python
def my_custom_randomization(env, env_ids, asset_cfg, param_range):
    robot = env.scene[asset_cfg.name]
    # 自定义随机化逻辑
    new_param = torch.rand(len(env_ids), device=env.device) * (param_range[1] - param_range[0]) + param_range[0]
    robot.data.some_param[env_ids] = new_param
    return {"my_param": new_param.mean().item()}
```

**陷阱**：
1. 修改后必须同步到 PhysX（某些参数需要 `robot.update_physx_properties()`）
2. 返回 dict 用于 logging，不返回则无法监控
3. 函数签名必须匹配 `EventTerm` 期望

---

## 七、DR 配置示例（legged_gym 风格）

完整的 locomotion DR 配置参考：

```python
@configclass
class EventsCfg:
    # 物理参数随机化（reset 时）
    robot_mass = EventTermCfg(
        func=mdp.add_robot_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_range": (-2.0, 2.0)},
    )
    ground_friction = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "friction_range": (0.3, 1.2),
            "num_buckets": 16,
        },
    )
    motor_strength = EventTermCfg(
        func=mdp.randomize_actuator_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_range": (0.8, 1.2),
            "damping_range": (0.8, 1.2),
        },
    )
    joint_damping = EventTermCfg(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "damping_range": (0.0, 0.5),
            "friction_range": (0.0, 0.05),
        },
    )

    # 外力扰动（interval）
    external_force = EventTermCfg(
        func=mdp.apply_random_external_force,
        mode="interval",
        interval_time_s=5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "force_range": (0.0, 5.0),
            "direction_range": (0, 2 * math.pi),
        },
    )
```

**对应观测噪声**（在 `ObservationsCfg` 中）：

```python
@configclass
class ObsCfg:
    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=OpenNoiseCfg(n_v=0.05))
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=OpenNoiseCfg(n_v=0.05))
    joint_pos = ObsTerm(func=mdp.joint_pos, noise=OpenNoiseCfg(n_v=0.01))
    joint_vel = ObsTerm(func=mdp.joint_vel, noise=OpenNoiseCfg(n_v=0.05))
```

---

## 八、参考实现与源码

- `IsaacLab/source/isaaclab/isaaclab/envs/mdp/events.py` — 内置 DR 函数
- `IsaacLab/source/isaaclab/isaaclab/envs/mdp/observations.py` — 观测噪声 API
- `legged_gym/legged_gym/envs/base/legged_robot.py::_randomize_rigid_body_props` — legged_gym DR
- `IsaacGymEnvs/isaacgymenvs/tasks/anymal_terrain.py` — AnymalTerrain 完整 DR 配置
- `Walk-These-Ways/wtw/envs/env.py` — Walk These Ways DR（含步态相关参数）

---

## 九、本 skill 中 DR 的现状

**MVP 范围**：DR 顾问是 roadmap 项，不在 MVP 中实现。但：
- 本文档提供完整的 DR 模式与陷阱清单，可在 SKILL.md 的"设计咨询"场景作为 LLM 参考
- 后续可作为 module 4 开发（`dr_advisor.py`：根据 env cfg 推荐合理 DR 范围）
- 示例 `examples/quadruped_locomotion/` 可包含基础 DR 配置作为演示
