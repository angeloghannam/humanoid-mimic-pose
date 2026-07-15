#!/usr/bin/env bash
#
# One-time dev environment setup for running mjlab tasks via `uv run`.
#
# `uv run` defaults to a project-local `.venv`, which ignores an active conda
# env (uv keys off VIRTUAL_ENV; conda sets CONDA_PREFIX instead). This script
# points uv at the active conda env via a conda activation hook, so
# `uv run train ...` uses the conda env — no duplicate `.venv`.
#
# Usage (with the project's conda env active):
#     conda activate humanoid-mimic-pose
#     bash setup_uv_env.sh
#     conda activate humanoid-mimic-pose   # re-activate to load the hook
#
set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: No conda env active. Run 'conda activate <env>' first." >&2
  exit 1
fi

echo "Configuring uv to use the active conda env: ${CONDA_PREFIX}"

# 1. Install the project (editable) + its deps into the conda env.
pip install -e .

# 2. Add an activation hook so UV_PROJECT_ENVIRONMENT is set automatically
#    whenever this conda env is activated.
hook_dir="${CONDA_PREFIX}/etc/conda/activate.d"
mkdir -p "${hook_dir}"
cat > "${hook_dir}/uv_env.sh" <<'EOF'
# Make `uv run` use this conda env instead of a project-local .venv,
# and don't let uv reshape the conda env against a lockfile.
export UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}"
export UV_NO_SYNC=1
EOF

# 3. Set them for the current shell too (so no re-activation is needed right now).
export UV_PROJECT_ENVIRONMENT="${CONDA_PREFIX}"
export UV_NO_SYNC=1

echo
echo "Done. UV_PROJECT_ENVIRONMENT -> ${CONDA_PREFIX}"
echo "Re-activate the env in other shells (conda activate ...) to pick up the hook."
echo
echo "Launch training with, e.g.:"
echo "  CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Stand-Flat-Unitree-G1 --env.scene.num-envs 256"
