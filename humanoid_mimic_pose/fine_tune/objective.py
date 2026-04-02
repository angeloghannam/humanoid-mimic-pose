from typing import Any, Dict

import optuna

import torch.nn as nn

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from humanoid_mimic_pose.fine_tune.callbacks.trial_evaluation import TrialEvalCallback
from humanoid_mimic_pose.fine_tune.configs.tuning_config import TuningConfig
from humanoid_mimic_pose.fine_tune.configs.ppo_config import PPOConfig


DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}


def sample_ppo_params(trial: optuna.Trial, config: PPOConfig) -> Dict[str, Any]:
    """Sampler for PPO hyperparameters."""

    n_steps_pow = trial.suggest_int(
        "n_steps_pow", config.n_steps_pow_range[0], config.n_steps_pow_range[-1])

    gamma = trial.suggest_float(
        "gamma", config.gamma_range[0], config.gamma_range[-1])

    learning_rate = trial.suggest_float(
        "learning_rate", config.lr_range[0], config.lr_range[-1], log=True)

    activation_fn = trial.suggest_categorical(
        "activation_fn", config.activation_func)

    n_steps = 2**n_steps_pow
    trial.set_user_attr("n_steps", n_steps)
    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_kwargs": {
            "activation_fn": activation_fn,
        },
    }


def objective(trial: optuna.Trial, config: TuningConfig) -> float:
    vec_env = make_vec_env(config.env, n_envs=config.n_parallel_envs)
    eval_env = make_vec_env(config.env, n_envs=config.n_parallel_envs)

    eval_freq = int(config.n_timesteps_per_trial /
                    config.n_evaluations_per_trial)

    kwargs = DEFAULT_HYPERPARAMS.copy()
    kwargs.update(sample_ppo_params(trial, config.ppo_config))

    model = PPO(env=vec_env, **kwargs)

    eval_callback = TrialEvalCallback(
        eval_env,
        trial,
        n_eval_episodes=config.n_eval_episodes,
        eval_freq=max(eval_freq // config.n_parallel_envs, 1),
        deterministic=True,
    )

    nan_encountered = False
    try:
        model.learn(config.n_timesteps_per_trial, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.best_mean_reward
