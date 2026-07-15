"""Console-script entry points for mjlab tasks defined in this project.

These wrappers import the project's task package so its tasks (e.g.
``Mjlab-Stand-Flat-Unitree-G1``) are registered in mjlab's registry, then
delegate to mjlab's own train/play CLIs. They intentionally shadow mjlab's
``train``/``play`` scripts so ``uv run train <task> ...`` finds project tasks.
"""


def train() -> None:
    import humanoid_mimic_pose.environments.mjlab.tasks  # noqa: F401  (registers tasks)
    from mjlab.scripts.train import main

    main()


def play() -> None:
    import humanoid_mimic_pose.environments.mjlab.tasks  # noqa: F401  (registers tasks)
    from mjlab.scripts.play import main

    main()