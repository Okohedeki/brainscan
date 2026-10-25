"""Decoding: probes on recorded agent runs, cross-validated by run, with shuffled-label nulls and
transcript baselines. Synthetic runs with a planted signal check that the probe finds the signal's
layer and beats the within-task null, and that a probe which can only read the task is called out."""

import json
import math

import numpy as np
import pytest
import torch

from brainscan.recorder import decoding
from brainscan.recorder.store import RunStore

SPEC = "test-model"
L, D = 4, 24


def test_auc_counts_ties_and_handles_one_class():
    y = np.array([1, 1, 0, 0, 1])
    s = np.array([0.9, 0.5, 0.5, 0.1, 0.7])
    brute = np.mean([(a > b) + 0.5 * (a == b) for a in s[y == 1] for b in s[y == 0]])
    assert decoding.auc(y, s) == pytest.approx(brute)
    assert math.isnan(decoding.auc(np.ones(3), np.arange(3)))
    w, n = decoding.within_task_auc(np.array([1, 0, 1, 1]), np.array([2, 1, 0, 1]), np.array(["a", "a", "b", "b"]))
    assert (w, n) == (1.0, 1)  # task b has one outcome only


def test_folds_keep_a_run_together():
    groups = np.repeat([f"r{i}" for i in range(10)], 3)
    strata = np.repeat(["x"] * 5 + ["y"] * 5, 3)
    f = decoding.folds_by_run(groups, strata, 5, np.random.default_rng(0))
    assert all(len(set(f[groups == g])) == 1 for g in set(groups))
    assert sorted(np.bincount(f)) == [6] * 5


def test_the_dual_ridge_matches_the_primal():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 8))
    y = rng.integers(0, 2, 30).astype(float)
    tr, te = np.arange(20), np.arange(20, 30)
    lam = 3.0
    got = decoding.Dual(Z).fit_predict(tr, te, y, lam)
    Zc = Z[tr] - Z[tr].mean(0)
    K = Zc @ Zc.T
    w = np.linalg.solve(Zc.T @ Zc + lam * np.trace(K) / len(tr) * np.eye(8), Zc.T @ (y[tr] - y[tr].mean()))
    assert np.allclose(got, (Z[te] - Z[tr].mean(0)) @ w)


def write_runs(store, outcome, signal=0.0, runs_per_task=8, seed=0):
    """Agent runs on three tasks. Each step's residual is noise plus the task's own offset at every
    layer, plus (at layer 2, in runs that fail) a planted direction scaled by `signal`."""
    rng = np.random.default_rng(seed)
    offsets = {t: rng.normal(size=D) * 2 for t in ("alpha", "beta", "gamma")}
    planted = rng.normal(size=D)
    planted /= np.linalg.norm(planted)
    for task in offsets:
        for i in range(runs_per_task):
            rid = f"{task}-{i}"
            fails = outcome(task, i)
            n = int(rng.integers(3, 7))
            store.write_meta(rid, {"id": rid, "created": i, "model": {"spec": SPEC}, "requests": n,
                                   "harness": {"task": task, "study": "s"}, "outcome": {"success": not fails}})
            events = [{"t": 0, "type": "user_message", "data": {}, "source": "recorder", "req": 1}]
            for k in range(1, n + 1):
                x = rng.normal(size=(L, D)) + offsets[task]
                if fails:
                    x[2] += signal * planted
                t = 10 * k
                store.write_residuals(rid, k, {f"t{t}": torch.tensor(x, dtype=torch.float16)},
                                      {"marks": {"pre_gen": t}})
                events.append({"t": t + 1, "type": "tool_call", "data": {"name": "run_tests", "arguments": "{}"},
                               "source": "recorder", "req": k})
            store.append_jsonl(rid, "events", events)
    return planted


def test_a_planted_signal_is_found_within_tasks(tmp_path):
    store = RunStore(tmp_path, 1)
    planted = write_runs(store, lambda task, i: i % 2 == 0, signal=3.0)
    r = decoding.train(store, SPEC, "fails", study="s", perms=9, seed=0)
    assert r["runs"] == 24 and r["positive_runs"] == 12
    assert r["probe"]["auc"] > 0.85 and r["probe"]["within_task"]["auc"] > 0.85
    assert max(r["layers"], key=lambda x: x["auc"])["layer"] == 2
    assert r["probe"]["layers_chosen"].get(2, 0) >= 4
    assert r["controls"]["shuffled_within_task"]["p"] <= 0.1
    assert r["baselines"]["transcript"]["within_task"]["auc"] < 0.75  # nothing about the outcome in the transcript
    layer, unit, samples = decoding.probe_direction(r)
    assert layer == 2 and abs(float(unit @ planted)) > 0.5 and float(unit @ planted) > 0  # points toward failing
    assert "_probe" not in decoding.public(r)


def test_a_task_detector_is_called_out(tmp_path):
    store = RunStore(tmp_path, 1)
    write_runs(store, lambda task, i: task == "alpha")  # failing is just being task alpha
    r = decoding.train(store, SPEC, "fails", study="s", perms=5, seed=0)
    assert r["probe"]["auc"] > 0.9  # it tells the tasks apart...
    assert math.isnan(r["probe"]["within_task"]["auc"])  # ...which says nothing within a task
    assert r["controls"]["shuffled_within_task"]["p"] == 1.0
    assert any("confounded" in n for n in r["notes"])
    out = decoding.public(r)
    assert out["probe"]["within_task"]["auc"] is None
    json.dumps(out, allow_nan=False)  # what the API sends
    from brainscan.studies import study1
    md = study1.markdown([out], store.list_runs())
    assert "| Probe (layer and strength chosen in each training fold) |" in md and "| – |" in md
    with pytest.raises(ValueError):
        decoding.train(store, SPEC, "nope")
    with pytest.raises(ValueError):
        decoding.train(store, "other-model", "fails")


def test_train_and_save_through_the_api(tmp_path):
    import threading
    import time
    from types import SimpleNamespace

    from fastapi.testclient import TestClient

    from brainscan import server
    from brainscan.recorder.directions import DirectionBank

    store = RunStore(tmp_path / "data", 1)
    write_runs(store, lambda task, i: i % 2 == 0, signal=3.0)
    subject = SimpleNamespace(spec=SPEC, slug="test-model", n_layers=L, d_model=D, device="cpu")
    rec = SimpleNamespace(store=store, s=subject, lock=threading.RLock(),
                          directions=DirectionBank(tmp_path / "directions", subject, None))
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        summary = client.get("/brainscan/decoding").json()
        assert summary["runs"] == 24 and summary["studies"] == ["s"] and "fails" in summary["targets"]
        assert client.post("/brainscan/decoding/train", json={"target": "nope"}).status_code == 400
        job = client.post("/brainscan/decoding/train", json={"target": "fails", "study": "s", "perms": 3}).json()
        for _ in range(600):
            job = client.get(f"/brainscan/decoding/jobs/{job['id']}").json()
            if job["status"] != "running":
                break
            time.sleep(0.1)
        assert job["status"] == "done", job.get("error")
        assert job["report"]["probe"]["auc"] > 0.85 and "_probe" not in job["report"]
        assert client.post(f"/brainscan/decoding/jobs/{job['id']}/save", json={"name": "bad name"}).status_code == 400
        dirs = client.post(f"/brainscan/decoding/jobs/{job['id']}/save", json={"name": "fails_probe"}).json()
        d = next(x for x in dirs if x["name"] == "fails_probe")
        assert d["kind"] == "probe" and d["layer"] == 2 and d["decoding"]["target"] == "fails"
        assert client.get("/brainscan/decoding/jobs/nope").status_code == 404
    finally:
        server.state.clear()
        server.state.update(saved)
