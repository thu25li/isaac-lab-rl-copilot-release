#!/usr/bin/env bash
# Train quadruped locomotion with the generated reward.
#
# Prerequisites:
#   - Isaac Lab installed (https://isaac-sim.github.io/IsaacLab/)
#   - GPU with >=8GB VRAM
#   - This example directory on PYTHONPATH
#
# Usage:
#   ./train.sh              # train with default config
#   ./train.sh --headless   # no GUI (for server/CI)
#   ./train.sh --resume     # resume from checkpoint

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Default args
ARGS="--task QuadrupedLocomotion --num_envs 4096"

# Pass through user args
for arg in "$@"; do
    case "$arg" in
        --headless|--resume|--video)
            ARGS="$ARGS $arg"
            ;;
        *)
            ARGS="$ARGS $arg"
            ;;
    esac
done

echo "Starting training with: $ARGS"
echo "Output logs: ./logs/tensorboard"
echo ""
echo "To monitor: tensorboard --logdir ./logs/tensorboard"
echo ""
echo "To diagnose failures, run the skill's diagnostic:"
echo "  python scripts/log_analyzer.py --logdir ./logs/tensorboard/<run>"
echo "  python scripts/diagnosis_engine.py --symptoms symptoms.json"
echo ""

python -m isaaclab.app.runner $ARGS
