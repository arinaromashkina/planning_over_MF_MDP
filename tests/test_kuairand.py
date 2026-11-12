"""Tests for the KuaiRand loader's sessionization and split-safety helpers."""

import numpy as np

from mf_mdp.data.kuairand import SESSION_GAP_MS


def test_session_gap_rule():
    """Sessions break at >30min gaps per user; positions restart per session."""
    # two users interleaved: u0 has a gap > 30min, u1 has no gap
    u = np.array([0, 1, 0, 0, 1])
    t = np.array([0, 1000, 2000, 2000 + SESSION_GAP_MS + 1, 3000])
    order = np.lexsort((t, u))
    u_s, t_s = u[order], t[order]
    session_id = np.zeros(len(t), dtype=np.int64)
    session_pos = np.zeros(len(t), dtype=np.int64)
    sess, pos = 0, 0
    for j in range(len(order)):
        if j == 0 or u_s[j] != u_s[j - 1]:
            sess, pos = 0, 0
        elif t_s[j] - t_s[j - 1] > SESSION_GAP_MS:
            sess += 1
            pos = 0
        session_id[order[j]] = sess
        session_pos[order[j]] = pos
        pos += 1
    # u0: rows 0 (t=0) and 2 (t=2000) and 3 (t=gap+1): session 0,0 then new session
    assert session_id[0] == 0 and session_pos[0] == 0
    assert session_id[2] == 0 and session_pos[2] == 1
    assert session_id[3] == 1 and session_pos[3] == 0
    # u1: rows 1 and 4, same session, positions 0,1
    assert session_id[1] == 0 and session_pos[1] == 0
    assert session_id[4] == 0 and session_pos[4] == 1


def test_popularity_uses_train_only():
    """Popularity features must come from the train log only (decision-time legal)."""
    train_v = np.array([0, 0, 1, 2])
    pop = np.bincount(train_v, minlength=5)
    assert pop.tolist() == [2, 1, 1, 0, 0]
