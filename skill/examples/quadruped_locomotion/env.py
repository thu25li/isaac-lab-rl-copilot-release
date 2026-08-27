#!/usr/bin/env python3
"""Quadruped locomotion env config for Isaac Lab.

Demonstrates how to mount the generated reward.py (from reward_synthesizer.py)
onto a ManagerBasedRLEnvCfg. This is a minimal but runnable configuration —
requires Isaac Lab to be installed to actually run training.

Usage:
    # After installing Isaac Lab:
    python -m isaaclab.app.runner --task QuadrupedLocomotion --headless

Reference:
    IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import (
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg,
    CommandTermCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.envs.mdp import actions as mdp_actions
from isaaclab.envs.mdp import observations as mdp_obs
from isaaclab.envs.mdp import commands as mdp_commands
from isaaclab.envs.mdp import terminations as mdp_terminations

# Import the generated reward config
from .reward import RewardsCfg


@configclass
class QuadrupedSceneCfg(InteractiveSceneCfg):
    """Scene with a quadruped robot on flat terrain."""

    # Robot: Unitree A1 (Isaac Lab built-in asset)
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=MISSING,  # Set in __post_init__ via USD asset path
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.42),  # A1 standing height
            joint_pos={
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.6,
            },
        ),
        actuators={
            "legs": ImplicitActuatorCfg(
                joint_names_expr=".*",
                stiffness=20.0,
                damping=0.5,
            ),
        },
    )

    # Flat terrain (for MVP; swap to TerrainGeneratorCfg for rough terrain)
    ground: TerrainGeneratorCfg = TerrainGeneratorCfg(
        curriculum_terms=[],
        size=(8.0, 8.0),
    )


@configclass
class ActionsCfg:
    """Joint position targets (PD control)."""

    joint_pos = ActionTermCfg(
        func=mdp_actions.joint_position_action,
        asset_cfg=SceneEntityCfg("robot", joint_names=".*"),
        scale=0.25,  # Action scaling
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation groups: policy critic share the same obs here."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        base_lin_vel = ObservationTermCfg(func=mdp_obs.base_lin_vel)
        base_ang_vel = ObservationTermCfg(func=mdp_obs.base_ang_vel)
        projected_gravity = ObservationTermCfg(func=mdp_obs.projected_gravity)
        joint_pos = ObservationTermCfg(
            func=mdp_obs.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        joint_vel = ObservationTermCfg(
            func=mdp_obs.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        last_action = ObservationTermCfg(
            func=mdp_obs.last_action,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
        )
        velocity_commands = ObservationTermCfg(
            func=mdp_obs.generated_commands,
            params={"command_name": "base_velocity"},
        )

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Velocity command: [lin_vel_x, lin_vel_y, ang_vel_z]."""

    base_velocity = CommandTermCfg(
        func=mdp_commands.UniformVelocityCommand,
        asset_cfg=SceneEntityCfg("robot"),
        resampling_time_range=(4.0, 8.0),
        rel_styling_envs=0.0,
        ranges=mdp_commands.UniformVelocityCommand.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
        ),
    )


@configclass
class TerminationsCfg:
    """Termination conditions: fall or timeout."""

    time_out = TerminationTermCfg(func=mdp_terminations.time_out)
    base_height_below_threshold = TerminationTermCfg(
        func=mdp_terminations.base_height_below_threshold,
        params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.2},
    )


@configclass
class QuadrupedEnvCfg(ManagerBasedRLEnvCfg):
    """Full env config: scene + actions + obs + commands + rewards + terminations."""

    # 4096 envs is typical for locomotion on a single GPU
    num_envs = 4096
    episode_length_s = 20.0  # 20 seconds per episode
    decimation = 4  # 200Hz physics -> 50Hz control

    # Scene
    scene: QuadrupedSceneCfg = QuadrupedSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )

    # Spaces
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()

    # Rewards — uses the generated RewardsCfg from reward.py
    rewards: RewardsCfg = RewardsCfg()

    # Terminations
    terminations: TerminationsCfg = TerminationsCfg()

    # No curriculum / DR in MVP — see references/curriculum_strategies.md
    # and references/domain_randomization.md for adding these.


# Register the task
from isaaclab_tasks.manager_based.locomotion.velocity.config.quadruped_robots import (
    register_task,
)

register_task("QuadrupedLocomotion", QuadrupedEnvCfg)
