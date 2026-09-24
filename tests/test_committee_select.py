import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
from committee_select import committees, consensus, cvar, gate, joint, loao, select  # noqa: E402


def complementary(nb=20):
    """model 0 breaks on column 0, model 1 on column 1, disjoint halves of behaviors; model 2 on both."""
    X = np.zeros((3, 2, nb))
    X[0, 0, : nb // 2] = 1
    X[1, 1, nb // 2 :] = 1
    X[2, :, :] = 1
    return X


def test_joint_counts_same_behavior_only():
    X = complementary()
    w = np.ones(20) / 20
    assert np.allclose(joint(X, (0,), w), [0.5, 0.0])
    assert np.allclose(joint(X, (0, 1), w), [0.0, 0.0])      # never the same behavior
    assert np.allclose(joint(X, (0, 2), w), [0.5, 0.0])      # k=2: both must break
    assert np.allclose(joint(X, (0, 1, 2), w), [0.5, 0.5])   # k=2 of 3


def test_cvar_limits():
    v = np.array([0.1, 0.9, 0.5, 0.3])
    assert cvar(v, 0.0) == 0.9
    assert cvar(v, 0.5) == 0.7
    assert np.isclose(cvar(v, 1.0), v.mean())


def test_select_prefers_complementary_pair_and_cheaper_on_ties():
    X = complementary()
    sel = select(X, committees(3), np.ones(20) / 20, 0.5)
    assert sel[0][2] == (0, 1)
    X0 = np.zeros((2, 1, 4))
    assert select(X0, committees(2, (1, 2)), np.ones(4) / 4, 0.5)[0][2] == (0,)


def test_consensus_missing_defers():
    A = np.array([1.0, 0.0, np.nan, np.nan])
    B = np.array([0.0, 0.0, 1.0, np.nan])
    assert np.allclose(consensus(A, B, "either")[:3], [1, 0, 1])
    assert np.allclose(consensus(A, B, "both")[:3], [0, 0, 1])
    assert np.isnan(consensus(A, B, "either")[3])


def test_gate_accepts_real_gain_and_rejects_none():
    X = complementary()
    g = gate(X, committees(3), 0.5, draws=200)
    assert g["accept"] and g["oob"]["gain_mean"] > 0
    g0 = gate(np.zeros((3, 2, 20)), committees(3), 0.5, draws=200)
    assert not g0["accept"] and g0["oob"]["gain_mean"] == 0


def test_loao_holds_out_family():
    X = complementary()
    cols = [("a", "v"), ("b", "v")]
    rows = loao(X, cols, committees(3), 0.5)
    assert [r["held_out"] for r in rows] == ["a", "b"]
    assert all(r["committee_worst"] <= r["single_worst"] for r in rows)


def test_split_eval_ranks_complementary_rules_above_random():
    from committee_select import split_eval
    X = complementary()
    r = split_eval(X, 0.5, reps=50)
    assert r["joint_cvar"]["heldout_cvar"] == 0.0
    assert r["joint_cvar"]["gain_vs_single"] > 0
    assert r["best_single"]["gain_vs_single"] == 0.0


def _replay_fixture():
    import collections
    meta = [{"model": "a", "refusal": False}, {"model": "b", "refusal": False},
            {"model": "a", "refusal": True}, {"model": "b", "refusal": True},
            {"model": "c", "refusal": False}, {"model": "c", "refusal": False}]
    e = np.eye(3, dtype=np.float32)
    emb = np.stack([e[0], e[0], e[1], e[1], e[2], e[0]])
    rows = collections.defaultdict(dict)
    k1, k2 = ("attack", "x", "v", "b1"), ("benign", "jbb", "plain", "b2")
    rows[k1].update({"a": 0, "b": 1, "c": 4})
    rows[k2].update({"a": 2, "b": 3, "c": 5})
    return meta, emb, rows, k1, k2


def test_adjudicate_groups_and_quorum():
    from serving_replay import adjudicate
    meta, emb, rows, k1, k2 = _replay_fixture()
    assert adjudicate(("a", "b"), k1, rows, emb, meta, 0.8) == 0      # agree -> release first
    assert adjudicate(("a", "c"), k1, rows, emb, meta, 0.8) is None   # split -> refuse
    assert adjudicate(("a", "b"), k2, rows, emb, meta, 0.8) is None   # agreed refusal
    assert adjudicate(("a", "b", "c"), k1, rows, emb, meta, 0.8) == 0  # 2 of 3


def test_evaluate_and_guard_filter():
    from serving_replay import evaluate, single_filtered
    meta, emb, rows, k1, k2 = _replay_fixture()
    V = {("a",) + k1: {"guard": "unsafe", "qwen3": "unsafe"},
         ("b",) + k1: {"guard": "safe", "qwen3": "unsafe"},
         ("c",) + k1: {"guard": "safe", "qwen3": "safe"}}
    r = evaluate(("a", "b"), 0.8, rows, emb, meta, V, "qwen3", 0.5)
    assert r["released_worst"] == 1.0 and r["benign_served"] == 0.0
    s = single_filtered("a", rows, meta, V, "qwen3", 0.5, guard_filter=True)
    assert s["released_worst"] == 0.0                                  # guard blocks it
    s = single_filtered("b", rows, meta, V, "either", 0.5, guard_filter=True)
    assert s["released_worst"] == 1.0                                  # guard misses, qwen3 catches


def test_cost_gate_charges_only_the_extra_cost():
    import numpy as np
    from cost_gate import cost, gate_cost
    assert abs(cost(["qwen3-4b", "Llama-2-7b"]) - (1 + 4 / 7)) < 1e-9
    X = complementary()                       # pair (0, 1) never co-fails, singles fail on half a column
    singles = [(0,), (1,), (2,)]
    cheap = gate_cost(X, (0, 1), singles, 0.5, c_extra=0.5, draws=200)
    dear = gate_cost(X, (0, 1), singles, 0.5, c_extra=100.0, draws=200)
    assert cheap["accept"] and not dear["accept"]
    assert abs(dear["price"] - 100 * 0.05) < 1e-9
