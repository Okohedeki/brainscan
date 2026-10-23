"""Actions: forget a fact. Target ranks the units behind an answer, try and measure knock them out
on typed prompts, and keeping the result as a weight edit gives what was measured.

Runs a real CPU recorder on the tiny saved Llama in fp32, so a knockout at run time and the same
units zeroed in the weights agree to float precision.
"""

from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

from brainscan import server
from brainscan.config import Settings
from brainscan.recorder import actions, workbench
from brainscan.recorder.model import Subject
from brainscan.recorder.session import GenParams, Recorder
from brainscan.recorder.store import RunStore

MODEL_DIR = Path.home() / ".cache" / "brainscan" / "models" / "tiny-llama-test"

pytestmark = pytest.mark.skipif(not MODEL_DIR.exists(), reason=f"needs the tiny CPU Llama at {MODEL_DIR}")


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("actions")
    subject = Subject(str(MODEL_DIR), dtype=torch.float32, device="cpu")
    settings = Settings(model=str(MODEL_DIR), device="cpu", anatomy=True, anatomy_clusters=8,
                        graph_decode=False, data_dir=root)
    r = Recorder(subject, RunStore(root / "data", settings.disk_budget_gb), settings, lambda m: None)
    r.generate(GenParams(messages=[{"role": "user", "content": "Name a color."}], max_tokens=8, temperature=0,
                         run_hint="act-run"))
    return r


@pytest.fixture()
def clean(rec):
    yield
    rec.revert_weight_edits()
    for eid in list(rec.edits.edits):
        rec.edits.remove(eid)
    for c in workbench.list_checks(rec):
        workbench.remove_check(rec, c["id"])


def run_fact(rec):
    """The word the model wrote right after the prompt of the recorded run."""
    run = rec.get_run("act-run")
    t = max(r["t"] for r in run.tape if r["ph"] != "gen")
    target = next(r["tok"] for r in run.tape if r["t"] == t + 1)
    return run, t, {"run": run.id, "t": t, "target": target}


def plain_prob(rec, ids, target):
    with rec.tracer._model():
        return torch.softmax(rec.tracer.last_logits(ids), -1)[target].item()


def test_target_ranks_units_and_matches_the_run_knockout(rec, clean):
    run, t, body = run_fact(rec)
    res = actions.forget_target(rec, body)
    directs = [u["direct"] for u in res["units"]]
    assert directs == sorted(directs, reverse=True) and all(d > 0 for d in directs)
    assert len(res["units"]) <= actions.CANDIDATES
    ex = rec.tracer.explain(run.id, t, target=str(body["target"]))
    assert res["fact"]["prob"] == pytest.approx(ex["target"]["prob"], abs=1e-5)
    ks = [p["k"] for p in res["curve"]]
    assert ks == sorted(ks) and ks[0] == 1
    # The typed-prompt path and the run's own knockout agree on the same units.
    for p in res["curve"][:3]:
        units = [u["g"] for u in res["units"][:p["k"]]]
        ko = rec.tracer.knockout(run.id, t, units, target=str(body["target"]))
        assert p["prob"] == pytest.approx(ko["after"]["prob"], abs=1e-5)
        assert p["rank"] == ko["after"]["rank"]
    assert res["suggested"] in ks
    if res["flipped"]:
        assert next(p for p in res["curve"] if p["k"] == res["suggested"])["gone"]
    assert res["fact"]["first"] == (res["fact"]["rank"] == 1)


def test_typed_fact_reads_like_a_check(rec, clean):
    f = actions.fact(rec, {"prompt": "Roses are ", "answer": "red"})
    assert f.lead == "Roses are" and f.target == workbench.answer_token(rec.s, "Roses are", "red")
    res = actions.forget_target(rec, {"prompt": "Roses are", "answer": "red"})
    with rec.lock:
        check = workbench._next_token(rec.s, workbench.prompt_ids(rec.s, "Roses are"), f.target)
    assert res["fact"]["prob"] == pytest.approx(check["prob"], abs=1e-5)
    with pytest.raises(ValueError):
        actions.fact(rec, {"prompt": "Roses are", "answer": ""})


def test_try_answers_with_and_without_the_units(rec, clean):
    _, _, body = run_fact(rec)
    units = [u["g"] for u in actions.forget_target(rec, body)["units"][:4]]
    res = actions.forget_try(rec, body | {"units": units, "leads": "Violets are", "questions": ["Name a color."]})
    assert [r["kind"] for r in res["rows"]] == ["lead", "lead", "question"]
    ids = workbench.prompt_ids(rec.s, "Violets are")
    with rec.tracer._model():
        assert res["rows"][1]["original"] == actions._continue(rec.s, ids, actions.LEAD_TOKENS)
        with rec.tracer.knocked_out(units):
            assert res["rows"][1]["edited"] == actions._continue(rec.s, ids, actions.LEAD_TOKENS)
    # No hook is left behind: the model reads exactly as before.
    f = actions.fact(rec, body)
    assert plain_prob(rec, f.ids, f.target) == pytest.approx(actions.forget_target(rec, body)["fact"]["prob"], abs=1e-6)
    with pytest.raises(ValueError):
        actions.forget_try(rec, body | {"units": []})
    with pytest.raises(ValueError):
        actions.forget_try(rec, body | {"units": list(range(65))})


def test_random_controls_match_the_set(rec):
    an = rec.anatomy
    units = [g for g in range(an.n_units) if an.units[g]["kind"] == "head"][:2] + \
            [g for g in range(an.n_units) if an.units[g]["kind"] == "cluster"][:3]
    a, b = actions._random_units(an, units, 0), actions._random_units(an, units, 1)
    assert a == actions._random_units(an, units, 0) and a != b
    for pick in (a, b):
        assert not set(pick) & set(units) and len(pick) == len(units)
        assert sum(an.locate(g)[1] == "head" for g in pick) == 2


def test_measure_then_keep_gives_what_was_measured(rec, clean):
    _, _, body = run_fact(rec)
    units = [u["g"] for u in actions.forget_target(rec, body)["units"][:4]]
    workbench.add_check(rec, {"prompt": "The sky is", "expect": "blue"})
    res = actions.forget_measure(rec, body | {"units": units, "leads": ["Violets are"], "keep": "Grass is → green"})
    assert len(res["leads"]) == 2 and [k["source"] for k in res["keep"]] == ["typed", "check"]
    assert res["drift"]["kl_per_token"] >= 0 and res["drift"]["tokens"] > 0
    assert res["control"] == {"seeds": actions.CONTROL_SEEDS, "units": 4}
    assert res["verdict"]["status"] in {"works", "partial", "costly", "kept", "unspecific"}
    with pytest.raises(ValueError):
        actions.forget_measure(rec, body | {"units": units, "keep": "no arrow here"})
    # Keep: the same units zeroed in the weights read exactly as the knockout did.
    rec.add_weight_edit(**rec.anatomy.lesion_units(units, note="forget"))
    rec.apply_weight_edits()
    f = actions.fact(rec, body)
    assert plain_prob(rec, f.ids, f.target) == pytest.approx(res["leads"][0]["after"]["prob"], abs=1e-5)
    grass = workbench.prompt_ids(rec.s, "Grass is")
    assert plain_prob(rec, grass, workbench.answer_token(rec.s, "Grass is", "green")) == \
        pytest.approx(res["keep"][0]["after"]["prob"], abs=1e-5)


def test_gone():
    assert actions.gone({"rank": 1, "prob": 0.9}, {"rank": 2, "prob": 0.4})
    assert not actions.gone({"rank": 1, "prob": 0.9}, {"rank": 1, "prob": 0.3})
    # An answer that wasn't first must lose most of its probability.
    assert not actions.gone({"rank": 2, "prob": 0.2}, {"rank": 5, "prob": 0.05})
    assert actions.gone({"rank": 2, "prob": 0.2}, {"rank": 40, "prob": 0.01})


def test_verdicts():
    lead = lambda before, after, control=0.0: {"before": {"rank": before, "prob": 0.9}, "after": {"rank": after, "prob": 0.1},
                                                "gone": after > 1, "control_gone": control}
    keep_ok = [{"before": {"rank": 1, "prob": 0.8}, "after": {"rank": 1, "prob": 0.79}}]
    keep_broken = [{"before": {"rank": 1, "prob": 0.8}, "after": {"rank": 3, "prob": 0.2}}]
    calm = {"kl_per_token": 0.001}
    assert actions._verdict([lead(1, 1)], keep_ok, calm)["status"] == "kept"
    assert actions._verdict([lead(1, 2, 1.0)], keep_ok, calm)["status"] == "unspecific"
    assert actions._verdict([lead(1, 2), lead(1, 2)], keep_ok, calm)["status"] == "works"
    assert actions._verdict([lead(1, 2), lead(1, 1)], keep_ok, calm)["status"] == "partial"
    assert actions._verdict([lead(1, 2)], keep_broken, calm)["status"] == "costly"
    assert actions._verdict([lead(1, 2)], keep_ok, {"kl_per_token": 0.5})["status"] == "costly"


def test_forget_endpoints(rec, clean):
    saved = dict(server.state)
    server.state.update(recorder=rec, status="ready", step="ready", error=None)
    try:
        client = TestClient(server.app)
        _, _, body = run_fact(rec)
        r = client.post("/brainscan/actions/forget/target", json=body)
        assert r.status_code == 200
        units = [u["g"] for u in r.json()["units"][:2]]
        r = client.post("/brainscan/actions/forget/try", json=body | {"units": units})
        assert r.status_code == 200 and r.json()["rows"][0]["kind"] == "lead"
        r = client.post("/brainscan/actions/forget/measure", json=body | {"units": units, "checks": False})
        assert r.status_code == 200 and r.json()["verdict"]["status"]
        assert client.post("/brainscan/actions/forget/nope", json=body).status_code == 404
        assert client.post("/brainscan/actions/forget/target", json={"prompt": ""}).status_code == 400
        assert client.post("/brainscan/actions/forget/target", json={"run": "nope", "t": 0, "target": 1}).status_code == 404
        assert client.post("/brainscan/actions/forget/try", json=body | {"units": [10 ** 6]}).status_code == 404
    finally:
        server.state.clear()
        server.state.update(saved)
