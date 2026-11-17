"""H3: sequential model as an action PROPOSER for planning (candidate-set generation).

See docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md. Torch-dependent pieces live in
`sasrec` and are optional; `pools` and `planning` are NumPy-only.
"""

from mf_mdp.proposal import pools, planning  # noqa: F401
