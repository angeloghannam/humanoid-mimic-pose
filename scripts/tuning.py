import joblib
import argparse
import logging

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from humanoid_mimic_pose.fine_tune.objective import objective
from humanoid_mimic_pose.environments.mujoco.unitree_g1 import UnitreeG1
from humanoid_mimic_pose.fine_tune.configs.tuning_config import TuningConfig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter fine tuning")
    parser.add_argument(
        "--config_path",
        type=str,
        required=False,
        help="Tuning config path"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        help="Study name"
    )

    args, _ = parser.parse_known_args()

    study_name = args.study_name

    if args.config_path is not None:
        try:
            config = TuningConfig.from_yaml(args.config_path)
            logging.info(f"Loaded config from {args.config_path}")
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            logging.info("Falling back to default config")
            config = TuningConfig()
    else:
        config = TuningConfig()

    sampler = TPESampler(
        n_startup_trials=config.n_startup_trials, multivariate=True)
    pruner = MedianPruner(n_startup_trials=config.n_startup_trials)

    study = optuna.create_study(
        sampler=sampler, pruner=pruner, direction="maximize", study_name=study_name)
    try:
        study.optimize(lambda trial: objective(trial, config),
                       n_trials=config.n_trials)
    except KeyboardInterrupt:
        pass

    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    if config.save_study:
        joblib.dump(study, f"{study_name}.joblib")
        logging.info(f"Saved study to {study_name}.joblib")
