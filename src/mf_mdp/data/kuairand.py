"""KuaiRand-1K loader with sessionization and chronological splits.

Exposure semantics (documented): every log row is a CONFIRMED exposure (the video was
served). is_click=0 is exposed-without-click — a valid negative response label (unlike
MovieLens, where non-interaction has unknown exposure). is_rand=1 rows come from the
randomized-exposure policy and support unbiased one-step checks under the
randomization assumption (documented, not assumed silently).

Sessions: no session id exists in the data; sessions are segmented per user by a
30-minute time-gap rule (documented assumption).
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FILES = {
    "standard_1": "log_standard_4_08_to_4_21_1k.csv",
    "standard_2": "log_standard_4_22_to_5_08_1k.csv",
    "random": "log_random_4_22_to_5_08_1k.csv",
}

LOG_FIELDS = ["user_id", "video_id", "date", "hourmin", "time_ms", "is_click", "is_like",
              "is_follow", "is_comment", "is_forward", "is_hate", "long_view",
              "play_time_ms", "duration_ms", "profile_stay_time", "comment_stay_time",
              "is_profile_enter", "is_rand", "tab"]

SESSION_GAP_MS = 30 * 60 * 1000  # 30 minutes (documented assumption)


@dataclass
class KuaiRandLog:
    user_idx: np.ndarray  # contiguous user ids
    video_idx: np.ndarray  # contiguous video ids
    time_ms: np.ndarray
    date: np.ndarray
    hourmin: np.ndarray
    is_click: np.ndarray
    is_like: np.ndarray
    long_view: np.ndarray
    is_hate: np.ndarray
    play_time_ms: np.ndarray
    duration_ms: np.ndarray
    is_rand: np.ndarray
    session_id: np.ndarray  # per-user session counter (gap rule)
    session_pos: np.ndarray  # position within session
    n_users: int
    n_videos: int


def _data_root() -> Path:
    return Path(os.environ.get("MFMDP_DATA_ROOT", "./data")) / "KuaiRand-1K" / "data"


def load_log(name: str, user_map: dict | None = None, video_map: dict | None = None):
    """Parse one log CSV into arrays; builds id maps on first call."""
    path = _data_root() / FILES[name]
    rows = {k: [] for k in ["user_id", "video_id", "date", "hourmin", "time_ms",
                            "is_click", "is_like", "long_view", "is_hate",
                            "play_time_ms", "duration_ms", "is_rand"]}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in rows:
                rows[k].append(int(row[k]))
    if user_map is None:
        user_map = {}
    if video_map is None:
        video_map = {}
    u_raw = np.asarray(rows["user_id"])
    v_raw = np.asarray(rows["video_id"])
    u_idx = np.array([user_map.setdefault(int(u), len(user_map)) for u in u_raw])
    v_idx = np.array([video_map.setdefault(int(v), len(video_map)) for v in v_raw])
    time_ms = np.asarray(rows["time_ms"], dtype=np.int64)

    # sessionization: per user, sort by time_ms, break at >30min gaps
    session_id = np.zeros(len(time_ms), dtype=np.int64)
    session_pos = np.zeros(len(time_ms), dtype=np.int64)
    order = np.lexsort((time_ms, u_idx))
    u_sorted, t_sorted = u_idx[order], time_ms[order]
    sess, pos = 0, 0
    for j in range(len(order)):
        if j == 0 or u_sorted[j] != u_sorted[j - 1]:
            sess, pos = 0, 0
        elif t_sorted[j] - t_sorted[j - 1] > SESSION_GAP_MS:
            sess += 1
            pos = 0
        session_id[order[j]] = sess
        session_pos[order[j]] = pos
        pos += 1

    return KuaiRandLog(
        user_idx=u_idx, video_idx=v_idx, time_ms=time_ms,
        date=np.asarray(rows["date"]), hourmin=np.asarray(rows["hourmin"]),
        is_click=np.asarray(rows["is_click"], dtype=np.int8),
        is_like=np.asarray(rows["is_like"], dtype=np.int8),
        long_view=np.asarray(rows["long_view"], dtype=np.int8),
        is_hate=np.asarray(rows["is_hate"], dtype=np.int8),
        play_time_ms=np.asarray(rows["play_time_ms"], dtype=np.int64),
        duration_ms=np.asarray(rows["duration_ms"], dtype=np.int64),
        is_rand=np.asarray(rows["is_rand"], dtype=np.int8),
        session_id=session_id, session_pos=session_pos,
        n_users=len(user_map), n_videos=len(video_map),
    ), user_map, video_map


def load_video_features(video_map: dict) -> np.ndarray:
    """video_features_basic aligned to video_map order; numeric columns only.

    Columns: video_id, author_id, video_type, upload_dt, upload_type, visible_status,
    video_duration, server_width, server_height, music_id, music_type, tag.
    We keep video_type, video_duration, music_type (numeric); tag/music_id/author are
    high-cardinality and excluded in v1 (documented).
    """
    n = len(video_map)
    feats = np.zeros((n, 3), dtype=np.float64)
    VIDEO_TYPE_CODES = {"NORMAL": 0.0, "AD": 1.0, "UNKNOWN": 2.0}

    def _f(x):
        try:
            return float(x)
        except ValueError:
            return -1.0

    with open(_data_root() / "video_features_basic_1k.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["video_id"])
            if vid in video_map:
                idx = video_map[vid]
                feats[idx, 0] = VIDEO_TYPE_CODES.get(row["video_type"], 2.0)
                feats[idx, 1] = _f(row["video_duration"])
                feats[idx, 2] = _f(row["music_type"])
    return feats


def load_user_features(user_map: dict) -> np.ndarray:
    """Selected numeric user features aligned to user_map order (static attributes)."""
    n = len(user_map)
    cols = ["user_active_degree", "is_lowactive_period", "register_days",
            "follow_user_num", "fans_user_num"]
    feats = np.zeros((n, len(cols)), dtype=np.float64)
    with open(_data_root() / "user_features_1k.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = int(row["user_id"])
            if uid in user_map:
                idx = user_map[uid]
                for j, c in enumerate(cols):
                    try:
                        feats[idx, j] = float(row[c])
                    except (ValueError, TypeError):
                        feats[idx, j] = 0.0
    return feats
