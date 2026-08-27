# Quadruped Locomotion Example

> 端到端示例：用 skill 生成 reward → 配置 env → 启动训练 → 诊断失败。契合用户机器狗比赛背景。

---

## 一、这个示例演示什么

1. **模块 1（reward 合成）**：`reward.py` 是由 `scripts/reward_synthesizer.py` 自动生成的，不是手写的
2. **完整 env 配置**：`env.py` 展示如何把生成的 `RewardsCfg` 挂载到 `ManagerBasedRLEnvCfg`
3. **训练启动**：`train.sh` 展示如何用 Isaac Lab runner 启动训练
4. **模块 2（诊断）**：训练失败时，用 `log_analyzer.py` + `diagnosis_engine.py` 定位原因

---

## 二、文件清单

| 文件 | 作用 |
|------|------|
| `reward.py` | **自动生成**的 reward 配置（9 个 pattern，通过静态校验） |
| `env.py` | 完整 env 配置：scene + actions + obs + commands + rewards + terminations |
| `train.sh` | 训练启动脚本 |
| `README.md` | 本文档 |

---

## 三、如何运行

### 3.1 前置条件

- Isaac Lab 已安装（[安装指南](https://isaac-sim.github.io/IsaacLab/)）
- GPU 显存 ≥ 8GB（4096 envs 约需 6GB）
- Python 3.10+

### 3.2 重新生成 reward（可选）

如果想用不同的任务描述重新生成 reward：

```bash
cd isaac-lab-rl-copilot

python scripts/reward_synthesizer.py \
    --task "train quadruped to walk forward at 1.5 m/s, rough terrain" \
    --output examples/quadruped_locomotion/reward.py \
    --validate --explain
```

生成的 reward 会覆盖 `reward.py`。`env.py` 中的 `from .reward import RewardsCfg` 会自动使用新配置。

### 3.3 启动训练

```bash
cd examples/quadruped_locomotion
./train.sh --headless
```

训练日志输出到 `./logs/tensorboard/<timestamp>`。

### 3.4 监控训练

```bash
tensorboard --logdir ./logs/tensorboard
```

关键指标：
- `Train/mean_reward`：应稳步上升
- `Train/mean_episode_length`：应稳定在 ~20s（1000 steps）
- `Loss/grad_norm`：应在 0.1-10 量级
- `Train/entropy`：应缓慢下降，不应骤降

### 3.5 诊断失败（如果训练出问题）

```bash
cd isaac-lab-rl-copilot

# Step 1: 分析日志，提取症状
python scripts/log_analyzer.py \
    --logdir ../examples/quadruped_locomotion/logs/tensorboard/<run> \
    --json > symptoms.json

# Step 2: 诊断引擎匹配失败模式
python scripts/diagnosis_engine.py \
    --symptoms symptoms.json \
    --top 3
```

输出会列出 top-3 候选失败模式，每个含：
- 置信度（基于症状匹配数 + 严重度）
- 根因列表
- 优先级排序的修复建议
- 验证方法

---

## 四、预期结果

### 4.1 健康训练

- **第 100 iteration**：reward 开始上升，episode_length 稳定在 1000
- **第 500 iteration**：reward 达到 ~5-8（速度跟踪生效）
- **第 1000 iteration**：reward 达到 ~8-12，机器人能稳定行走
- **第 2000 iteration**：reward 平台在 ~10-15，步态成型

具体数值取决于 GPU、num_envs、seed 等。关键是**趋势**而非绝对值。

### 4.2 常见失败与诊断

| 症状 | 可能病因 | 修复 |
|------|---------|------|
| reward 飙升但机器人不动 | reward_hacking | 检查 per-term contribution，下调异常主导项 |
| reward 突降不恢复 | policy_collapse | 启用 target_kl early stop，降低 lr |
| grad_norm 爆炸 | gradient_explosion | 启用 grad_clip，归一化 reward |
| entropy 骤降 | entropy_collapse | 加 entropy_coef，降低 lr |
| reward 长期平台 | local_optimum | 加 curriculum，扩大 command 范围 |

详见 `references/training_failure_modes.md`。

---

## 五、扩展方向

### 5.1 加地形课程

替换 `env.py` 中的 `ground` 配置：

```python
from isaaclab.terrains import TerrainGeneratorCfg, FlatTerrainCfg, RoughTerrainCfg

ground: TerrainGeneratorCfg = TerrainGeneratorCfg(
    curriculum_terms=[
        mdp_curriculum.terrain_levels,
    ],
    size=(8.0, 8.0),
    sub_terrains={
        "flat": FlatTerrainCfg(),
        "rough": RoughTerrainCfg(proportions=0.5),
    },
)
```

详见 `references/curriculum_strategies.md`。

### 5.2 加 Domain Randomization

在 `env.py` 中添加 `EventsCfg`：

```python
@configclass
class EventsCfg:
    robot_mass = EventTermCfg(
        func=mdp_events.add_robot_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "mass_range": (-2.0, 2.0)},
    )
    ground_friction = EventTermCfg(
        func=mdp_events.randomize_rigid_body_material,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot"), "friction_range": (0.3, 1.2)},
    )
```

详见 `references/domain_randomization.md`。

### 5.3 用模块 1 生成不同任务的 reward

```bash
# 崎岖地形
python scripts/reward_synthesizer.py \
    --task "train quadruped to walk over rough terrain with stairs" \
    --output reward_rough.py --validate

# 低能耗
python scripts/reward_synthesizer.py \
    --task "train quadruped to walk forward, minimize energy consumption" \
    --include-optional \
    --output reward_energy.py --validate
```

---

## 六、与 Eureka 的对比

本示例的 reward 由 `reward_synthesizer.py` 生成，**不需要运行任何 RL 训练来生成 reward**（与 Eureka 的 16 候选 × 5 轮进化不同）。生成过程：

1. NL 任务描述 → task type 检测（`pattern_matcher.py`）
2. task type → pattern 选择（基于 `TASK_TYPE_PATTERNS` 模板）
3. pattern + 默认权重 → 模板渲染（`locomotion_full_cfg.py.tmpl`）
4. 静态校验（`reward_validator.py`，7 项检查）

整个过程 < 1 秒，输出可直接挂载到 env。Eureka 的进化式生成需要 GPU 训练评估每个候选，耗时数小时。

详见 `references/eureka_comparison.md`。

---

## 七、技术支持

- skill 总览：`README.md`（项目根目录）
- reward 设计：`references/reward_design_patterns.md`
- 失败诊断：`references/training_failure_modes.md`
- 课程设计：`references/curriculum_strategies.md`
- DR 设计：`references/domain_randomization.md`
- 技术演进：`技术演进文档.md`
