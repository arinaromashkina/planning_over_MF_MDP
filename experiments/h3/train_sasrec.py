"""Train SASRec on ML-1M LLN prefixes as an action proposer (H3).

Usage:
    .venv/bin/python experiments/h3/train_sasrec.py --seed 0 [--device mps] [--max-epochs 200]

Writes checkpoints/sasrec_ml1m_seed{S}.pt and outputs/h3/sasrec_train_seed{S}.json.
Model-level sanity metrics (next-item Recall@10/NDCG@10 on valid/test, NO consumed
masking — this is the model's own ranking quality, reported separately from its role
as a proposer; see docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md §1.1 falsification).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mf_mdp.data.movielens import build_lln
from mf_mdp.evaluation.metrics import ndcg_at_k, recall_at_k
from mf_mdp.proposal.sasrec import (
    SASRec,
    SASRecConfig,
    build_seq_array,
    fit_sasrec,
    recall_at_k_from_logits,
    score_histories,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if _mps_available() else "cpu")
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=1)
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--limit-users", type=int, default=None, help="debug: cap users")
    args = ap.parse_args()

    out_root = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs"))
    ckpt_root = Path("./checkpoints")
    ckpt_root.mkdir(exist_ok=True)
    (out_root / "h3").mkdir(parents=True, exist_ok=True)

    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    splits = data.splits
    if args.limit_users:
        splits = splits[: args.limit_users]
    n_items = data.n_items
    print(f"users={len(splits)} n_items={n_items}")

    histories = [sp.prefix for sp in splits]
    seqs = build_seq_array(histories, n_items, args.maxlen)
    valid_seqs = build_seq_array(histories, n_items, args.maxlen)

    valid_targets = np.array([int(sp.valid[0]) for sp in splits], dtype=np.int64)
    test_targets = np.array([int(sp.test[0]) for sp in splits], dtype=np.int64)

    cfg = SASRecConfig(
        n_items=n_items,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        maxlen=args.maxlen,
        dropout=args.dropout,
    )
    model = SASRec(cfg)

    def valid_fn(m: SASRec) -> float:
        logits = score_histories(m, valid_seqs, device=args.device)
        return recall_at_k_from_logits(logits, valid_targets, k=10)

    t0 = time.perf_counter()
    model, history = fit_sasrec(
        model,
        seqs,
        valid_fn,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    train_s = time.perf_counter() - t0

    ckpt_path = ckpt_root / f"sasrec_ml1m_seed{args.seed}.pt"
    torch_save(model, ckpt_path)

    logits = score_histories(model, valid_seqs, device=args.device)

    def topk_rows(lg: np.ndarray, k: int) -> list[list[int]]:
        part = np.argpartition(-lg, k - 1, axis=1)[:, :k]
        return [list(row[np.argsort(-lg[i, row], kind="stable")]) for i, row in enumerate(part)]

    metrics = {}
    for name, targets in (("valid", valid_targets), ("test", test_targets)):
        recs = topk_rows(logits, 10)
        metrics[f"{name}_recall10"] = float(np.mean([recall_at_k(r, {int(t)}, 10) for r, t in zip(recs, targets)]))
        metrics[f"{name}_ndcg10"] = float(np.mean([ndcg_at_k(r, {int(t)}, 10) for r, t in zip(recs, targets)]))
    metrics["n_users"] = len(splits)

    result = {
        "config": vars(args),
        "sasrec_cfg": {
            "n_items": cfg.n_items,
            "dim": cfg.dim,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "maxlen": cfg.maxlen,
            "dropout": cfg.dropout,
        },
        "history": history,
        "metrics": metrics,
        "train_time_s": train_s,
        "checkpoint": str(ckpt_path),
    }
    out_path = out_root / "h3" / f"sasrec_train_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"train_time_s={train_s:.1f}; wrote {out_path}")


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def torch_save(model, path: Path) -> None:
    import torch

    torch.save(model.state_dict(), path)


if __name__ == "__main__":
    main()
