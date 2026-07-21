# humanoid-mimic-pose

## Setup

```
conda env create -f environment.yml
conda activate humanoid-mimic-pose
bash setup_uv_env.sh
```

`setup_uv_env.sh` installs the project and points `uv run` at the conda env
(so it doesn't build a separate `.venv`). Re-activate the env afterwards to
load the hook: `conda activate humanoid-mimic-pose`.

## Train

```
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Stand-Flat-Unitree-G1 --env.scene.num-envs 256
```
