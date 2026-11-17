"""H4: interaction models for the MF-MDP response model.

Factorization Machine and MLP classifiers plus grounded/random cross-feature
builders, tested against the logistic baseline on KuaiRand (predictive level)
and Env-C (decision level). See docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md.
"""

from mf_mdp.interactions.cross import (
    BASE_NAMES,
    ENVC_CROSS_NAMES,
    ENVC_HIST_NAMES,
    KUAI_CROSS_NAMES,
    KUAI_HIST_NAMES,
    N_BASE,
    RandomCrosses,
    envc_crosses,
    kuairand_crosses,
)
from mf_mdp.interactions.fm import FactorizationMachine
from mf_mdp.interactions.mlp import MLPClassifier

__all__ = [
    "BASE_NAMES", "ENVC_CROSS_NAMES", "ENVC_HIST_NAMES",
    "KUAI_CROSS_NAMES", "KUAI_HIST_NAMES", "N_BASE",
    "RandomCrosses", "envc_crosses", "kuairand_crosses",
    "FactorizationMachine", "MLPClassifier",
]
