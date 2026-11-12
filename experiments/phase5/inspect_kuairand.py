"""Gate E: KuaiRand-1K data inspection (prompt §8).

Documents users/items/interactions, exposure policies (is_rand), timestamp ranges,
feedback fields, watch-ratio semantics, missing values, and session-boundary
assumptions. Read-only; writes outputs/phase5/kuairand_inspection.json.

Usage: python experiments/phase5/inspect_kuairand.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("MFMDP_DATA_ROOT", "./data")) / "KuaiRand-1K" / "data"
OUT = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase5"


def inspect_log(path: Path, name: str) -> dict:
    import csv

    fields = None
    n = 0
    users, videos = set(), set()
    dates = []
    feedback = {"is_click": 0, "is_like": 0, "is_follow": 0, "is_comment": 0,
                "is_forward": 0, "is_hate": 0, "long_view": 0, "is_profile_enter": 0}
    is_rand = 0
    watch_ratios = []
    missing = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            n += 1
            users.add(row["user_id"])
            videos.add(row["video_id"])
            if n % 200000 == 0:
                dates.append((row["date"], row["hourmin"]))
            for k in feedback:
                feedback[k] += int(row[k])
            is_rand += int(row["is_rand"])
            if int(row["is_click"]) == 1 and int(row["duration_ms"]) > 0:
                watch_ratios.append(int(row["play_time_ms"]) / int(row["duration_ms"]))
            for k, v in row.items():
                if v == "" or v is None:
                    missing[k] = missing.get(k, 0) + 1
    wr = np.asarray(watch_ratios) if watch_ratios else np.zeros(1)
    return {
        "name": name,
        "rows": n,
        "fields": fields,
        "n_users": len(users),
        "n_videos": len(videos),
        "date_range": [min(d for d, _ in dates), max(d for d, _ in dates)] if dates else None,
        "is_rand_rows": is_rand,
        "is_rand_frac": round(is_rand / max(n, 1), 4),
        "feedback_rates": {k: round(v / max(n, 1), 4) for k, v in feedback.items()},
        "watch_ratio_on_clicks": {
            "mean": float(wr.mean()), "median": float(np.median(wr)),
            "p90": float(np.quantile(wr, 0.9)), "frac_ge_0.8": float((wr >= 0.8).mean()),
            "frac_ge_1.0": float((wr >= 1.0).mean()),
        },
        "missing_values": missing,
    }


def main() -> None:
    out = {"dataset": "KuaiRand-1K", "files": {}}
    for fname, name in [
        ("log_standard_4_08_to_4_21_1k.csv", "standard_1"),
        ("log_standard_4_22_to_5_08_1k.csv", "standard_2"),
        ("log_random_4_22_to_5_08_1k.csv", "random"),
    ]:
        print(f"inspecting {fname}...", flush=True)
        out["files"][name] = inspect_log(DATA / fname, name)
    # feature file headers + row counts (cheap: read headers only)
    for fname in ("user_features_1k.csv", "video_features_basic_1k.csv"):
        with open(DATA / fname) as f:
            header = f.readline().strip().split(",")
        out.setdefault("feature_files", {})[fname] = {"columns": header[:20], "n_columns": len(header)}
    out["session_boundary_assumption"] = (
        "No explicit session id. Sessions are segmented by (user_id, date) with a "
        "time-gap rule to be validated (e.g., 30min gap between consecutive time_ms); "
        "documented as an assumption, not a dataset fact."
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "kuairand_inspection.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2)[:3000])
    print(f"\nwrote {OUT / 'kuairand_inspection.json'}")


if __name__ == "__main__":
    main()
