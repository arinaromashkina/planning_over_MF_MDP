#!/usr/bin/env python
"""Single entry point: reproduce all project results and figures with one command.

    python run_experiment.py            # full run (~3-4 min on the M4)
    python run_experiment.py --fast     # quick smoke run (~40 s)

Pipeline (each step writes JSON to outputs/, figures are built from saved JSONs only):
  1. Env-S planning-correctness checks (greedy vs DP oracle, MCTS convergence)
  2. Env-S response/transition learning + held-out predictive metrics
  3. Env-S ablation: optimistic vs stochastic planners (TRUE-env J, 5 seeds)
  4. Env-S noise grid: error-vs-depth (H3)
  5. Env-C ablation: MF-posterior regime
  6. figures/make_figures.py -> figures/*.png|.pdf
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run_step(name: str, cmd: list[str]) -> None:
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        raise SystemExit(f"step failed: {name} (exit {proc.returncode})")
    print(f"--- {name} done in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="smoke run: fewer rollouts/seeds")
    args = parser.parse_args()

    rollouts = "1500" if args.fast else "4000"
    seeds = "2" if args.fast else "5"
    noise_seeds = "1" if args.fast else "3"

    env = {"MFMDP_OUTPUT_ROOT": str(ROOT / "outputs"), "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home())}
    import os
    full_env = {**os.environ, **env}

    def sh(cmd):
        subprocess.run(cmd, cwd=ROOT, env=full_env, check=True)

    steps = [
        ("Env-S checks (A1)", [PY, "experiments/phase5/run_env_a.py", "--checks", "--n-rollouts", rollouts]),
        ("Env-S learn (A2)", [PY, "experiments/phase5/run_env_a.py", "--learn"]),
        ("Env-S ablation (A3)", [PY, "experiments/phase5/run_env_a.py", "--ablation", "--n-rollouts", rollouts, "--seeds", seeds]),
        ("Env-S noise grid (H3)", [PY, "experiments/phase5/run_env_a.py", "--noise", "--n-rollouts", rollouts, "--seeds", noise_seeds]),
        ("Env-C ablation", [PY, "experiments/phase5/run_env_c.py", "--seeds", seeds]),
        ("Figures", [PY, "figures/make_figures.py"]),
    ]
    t0 = time.time()
    for name, cmd in steps:
        run_step(name, cmd)
    print(f"\nAll done in {(time.time() - t0) / 60:.1f} min. Figures in figures/, results in outputs/.")


if __name__ == "__main__":
    main()
