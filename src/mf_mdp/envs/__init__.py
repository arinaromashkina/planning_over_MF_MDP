"""Controlled stochastic recommendation environments for Phase 5 (Experiment A).

Env-S: small discrete environment with an exact dynamic-programming oracle.
Generative equations are fully documented in docs/PHASE5_EXPERIMENT_PLAN.md and in
EnvSConfig. The environment is defined INDEPENDENTLY of any learned model.
"""

from .discrete_env import EnvS, EnvSConfig, GreedyPolicy, DPPolicy, MCTSPolicy

__all__ = ["EnvS", "EnvSConfig", "GreedyPolicy", "DPPolicy", "MCTSPolicy"]
