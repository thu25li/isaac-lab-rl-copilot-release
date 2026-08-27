# Isaac Lab 概览

> 本文档是 Isaac Lab 的概念性介绍。具体 API 签名与代码示例见 `resources/isaac_lab_api.json`。

## 这是什么

Isaac Lab 是 NVIDIA 基于 Isaac Sim / Omniverse 构建的机器人强化学习框架，是 Isaac Gym 的继任者（2023 年发布）。它提供：

- 基于 PhysX 5 的高保真物理仿真
- GPU 并行仿真（同时跑数千个机器人实例，显著加速 RL 数据采集）
- 基于 Python 的环境定义 API（兼容 Gymnasium 接口）
- 内置 RL 训练流程（支持 RSL-RL、SKRL、Stable Baselines3、rl_games 等库）
- 丰富的机器人 asset（Unitree A1/Go2、ANYmal、Franka、UR 等）

## 核心架构

Isaac Lab 提供两种环境定义方式：

### 1. Manager-based Env

通过 Manager 系统组织环境逻辑：
- **Scene Manager**: 机器人、地形、物体的场景定义
- **Observation Manager**: 观察向量的组织（policy / critic 两组）
- **Action Manager**: action 空间与执行（关节位置/力矩等）
- **Reward Manager**: reward 函数的注册与组合
- **Termination Manager**: 终止条件
- **Curriculum Manager**: 课程学习（动态调整难度）

### 2. Direct RL Env

更直接的方式，继承 `DirectRLEnv`，适合需要细粒度控制的场景。所有逻辑在单一 Python 类中实现。

## Reward 设计哲学

Isaac Lab 的 reward 通过 Reward Manager 管理。核心设计思想：

1. **Reward = 加权求和的多个 term**
2. **每个 term 是一个 Python 函数**：输入 env state，输出 scalar（或 per-env tensor）
3. **权重在 config 中配置**，方便实验，不需要改代码
4. **term 可以引用任意 env 属性**：关节位置/速度、base 线速度/角速度、接触力等

典型 reward 结构（伪代码）：
```
total_reward = w1 * track_lin_vel + w2 * track_ang_vel
             - w3 * orientation_penalty
             - w4 * energy_penalty
             - w5 * action_rate_penalty
             - w6 * termination_penalty
```

详细 API 与示例见 `resources/isaac_lab_api.json` 与 `references/reward_design_patterns.md`。

## 训练流程

```bash
# 训练（示例命令）
./isaaclab.sh -p scripts/train.py --task=Isaac-Velocity-Flat-Unitree-A1-v0

# 播放训练好的 policy
./isaaclab.sh -p scripts/play.py --task=Isaac-Velocity-Flat-Unitree-A1-v0

# 监控（tensorboard）
tensorboard --logdir logs/
```

## 与 Isaac Gym 的区别

| 维度 | Isaac Gym (Preview) | Isaac Lab |
|------|---------------------|-----------|
| 基础 | 独立实现 | 基于 Isaac Sim/Omniverse |
| 物理 | PhysX 4 | PhysX 5 |
| 渲染 | RTX 渲染器 | Omniverse RTX |
| 维护状态 | 已停止更新 | 活跃维护 |
| 配置系统 | 自定义 | Hydra-based |
| Env 风格 | 类 IsaacGymEnvs | Manager-based 或 Direct |

新项目应使用 Isaac Lab。本 skill 针对 Isaac Lab（部分内容向后兼容 Isaac Gym）。

## 常见痛点

1. **环境配置复杂**: config 系统学习曲线陡峭，新人上手慢
2. **Reward 设计靠经验**: 试错成本高，权重难调（本 skill 重点解决）
3. **训练调试困难**: tensorboard 指标解读需要经验，故障定位靠猜（本 skill 重点解决）
4. **Sim-to-real gap**: 仿真训好的 policy 上真机失效，domain randomization 难调
5. **计算资源需求高**: 需要 RTX GPU，迭代周期受限

本 skill 聚焦痛点 2 和 3，痛点 4 和 5 在 roadmap 中。

## 学习资源

- 官方文档: https://isaac-sim.github.io/IsaacLab
- GitHub: https://github.com/isaac-sim/IsaacLab
- 论文: "Isaac Lab: High-Performance GPU-Based Robot Learning Framework" (Mittal et al., 2023)
- 论文: "IsaacGym: High Performance GPU-Based Physics Simulation for Robot Learning" (Makoviychuk et al., 2021)
- NVIDIA Isaac Sim 文档: https://docs.isaacsim.omniverse.nvidia.com/

## 版本说明

本 skill 基于 Isaac Lab v1.x 开发。Isaac Lab 仍在快速迭代，API 可能有变化，使用时请对照官方文档验证。`resources/isaac_lab_api.json` 中记录的 API 签名基于 v1.x，跨版本使用需注意。
