from gymnasium import Env
from optuna import Trial
from stable_baselines3.common.callbacks import EvalCallback


EVAL_FREQ = 10000
N_EVAL_EPISODES = 5
DETERMINISTIC = True
VERBOSE = 0


class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env: Env,
        trial: Trial,
        n_eval_episodes: int = N_EVAL_EPISODES,
        eval_freq: int = EVAL_FREQ,
        deterministic: bool = DETERMINISTIC,
        verbose: int = VERBOSE,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()  # evaluate policy
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True
