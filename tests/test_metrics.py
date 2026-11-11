from mf_mdp.evaluation.metrics import ndcg_at_k, recall_at_k


def test_recall_at_k():
    relevant = {1, 3, 5}
    recs = [5, 2, 3, 4, 1]
    assert recall_at_k(recs, relevant, 5) == 3 / 3
    assert recall_at_k(recs, relevant, 2) == 1 / 2
    assert recall_at_k(recs, {9}, 10) == 0.0


def test_ndcg_at_k():
    relevant = {0}
    recs = [0, 1, 2]
    assert ndcg_at_k(recs, relevant, 3) == 1.0
    recs = [1, 0, 2]
    assert ndcg_at_k(recs, relevant, 3) < 1.0
