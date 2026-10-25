"""Decoding: linear probes trained on recorded agent runs, with the controls that say whether a
probe reads anything the transcript doesn't already show.

A sample is one agent step: the residual stream at every layer just before the model starts its
reply (the "pre_gen" decision residual the recorder saves for each request), together with what
the transcript showed by then. A target labels whole runs and is read at every step, so each
forecast can be scored by how many steps before the end of the run it was made (its lead):

  fails      the run ends with its tests failing
  gives_up   the run ends failing because the agent stopped calling tools on its own (not because
             it ran out of steps)

The probe is ridge regression on standardized residuals (a linear probe), fitted in its dual form
so the hundreds of fits behind the controls stay cheap. Everything is cross-validated by run: no
run has steps on both sides of a split. Each fold's held-out scores are ranked within the fold
before they are pooled, so folds with differently scaled models compare fairly. Reported, with 95%
bootstrap intervals over runs:

  probe              out-of-fold AUC. The layer and the ridge strength are chosen inside each
                     training fold (nested), so the headline isn't picked after seeing the test runs.
                     Also the AUC within each task that has both outcomes, averaged: a probe that only
                     tells the tasks apart scores 0.5 there.
  layers             out-of-fold AUC at every layer (strength chosen inside each fold)
  shuffled labels    the whole nested pipeline, with the probe's own splits, on run labels shuffled
                     across runs and on labels shuffled within each task (the null for "it only reads
                     which task this is"), many times; p counts the shuffles that score at least the
                     probe's AUC, plus one, over the shuffles plus one
  position only      step number and tokens so far
  transcript only    the task, step number, tool calls, tool errors, test runs, failing test runs,
                     whether the last run failed, file edits, edits to test files, repeated tool
                     calls and tokens so far: what a monitor could read without the model's internals
  lead               AUC at each number of steps before the run ended, for the probe and for the
                     transcript baseline, and the gap between them

Global standardization of the residuals uses every sample's features but never a label.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter

import numpy as np
from safetensors import safe_open

TARGETS = {
    "fails": "the run ends with its tests failing",
    "gives_up": "the run ends failing because the agent stopped calling tools on its own",
}
GEN_EVENTS = {"tool_call", "final_answer", "cutoff", "cancelled", "think_end", "monitor"}
LAMS = (0.3, 3.0, 30.0)  # ridge strengths, relative to the mean diagonal of the training kernel
K_OUTER, K_INNER = 5, 4
BOOT = 1000
TRANSCRIPT = ("step", "tokens", "tool_calls", "errors", "test_runs", "failing_runs", "last_failing", "edits",
              "test_edits", "repeats")
POSITION = ("step", "tokens")


# ---------------------------------------------------------------- the samples
def candidate_runs(store, model_spec: str, study: str | None = None) -> list[dict]:
    """Agent runs of this model with an outcome (and, with study, that study's tag)."""
    out = []
    for meta in store.list_runs():
        h = meta.get("harness") or {}
        if (meta.get("model") or {}).get("spec") != model_spec or not meta.get("outcome") or not h.get("task"):
            continue
        if study and h.get("study") != study:
            continue
        out.append(meta)
    return out


def transcript_features(events: list[dict], k: int, t: int) -> dict:
    """What the transcript showed when request k's prompt had been read. Events of earlier requests
    count; of request k itself only those read from its prompt (not its own reply, nor what the
    harness reported after running the reply's tool calls)."""
    seen = [e for e in events if e.get("req", 0) < k
            or (e.get("req") == k and e.get("source") != "harness" and e["type"] not in GEN_EVENTS)]
    calls = [(e["data"].get("name"), e["data"].get("arguments")) for e in seen if e["type"] == "tool_call"]
    tests = [e["data"] for e in seen if e["type"] == "test_result"]
    bad = [d for d in tests if (d.get("failed") or 0) + (d.get("errors") or 0)]
    edits = [e["data"] for e in seen if e["type"] == "file_edit"]
    return {"step": k, "tokens": math.log1p(t), "tool_calls": len(calls), "repeats": len(calls) - len(set(calls)),
            "errors": sum(e["type"] == "error" for e in seen), "test_runs": len(tests), "failing_runs": len(bad),
            "last_failing": float(bool(tests) and tests[-1] in bad), "edits": len(edits),
            "test_edits": sum(bool(d.get("is_test")) for d in edits)}


def run_labels(meta: dict, events: list[dict]) -> dict:
    n = meta.get("requests", 0)
    fails = not meta["outcome"].get("success")
    stopped = any(e["type"] == "final_answer" and e.get("req") == n for e in events)
    return {"fails": fails, "gives_up": fails and stopped}


def load_steps(store, metas: list[dict]) -> tuple[np.ndarray | None, list[dict]]:
    """X (steps, layers, d) float16 and one row per step."""
    X, rows = [], []
    for meta in metas:
        rid, n = meta["id"], meta.get("requests", 0)
        events = store.read_jsonl(rid, "events")
        labels = run_labels(meta, events)
        for k in range(1, n + 1):
            path = store.runs_dir / rid / "residuals" / f"req{k:04d}.safetensors"
            if not path.exists():
                continue
            with safe_open(str(path), "np") as f:
                t = json.loads((f.metadata() or {}).get("marks", "{}")).get("pre_gen")
                if t is None or f"t{t}" not in f.keys():
                    continue
                X.append(f.get_tensor(f"t{t}").astype(np.float16))
            rows.append({"run": rid, "task": meta["harness"]["task"], "k": k, "steps": n, "lead": n - k, "t": t,
                         **labels, **transcript_features(events, k, t)})
    if X and len({x.shape for x in X}) > 1:
        raise ValueError("the runs' residuals have different shapes; were they recorded with different models?")
    return (np.stack(X) if X else None), rows


# ---------------------------------------------------------------- statistics
def _ranks(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged."""
    order = np.argsort(a, kind="mergesort")
    s = a[order]
    edges = np.flatnonzero(np.r_[True, s[1:] != s[:-1], True])
    r = np.empty(len(a))
    for i in range(len(edges) - 1):
        r[order[edges[i]:edges[i + 1]]] = (edges[i] + edges[i + 1] + 1) / 2
    return r


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if not npos or not nneg:
        return float("nan")
    r = _ranks(np.asarray(s, float))
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def within_task_auc(y, s, tasks) -> tuple[float, int]:
    """The AUC inside each task that has both outcomes, averaged (and how many tasks had both)."""
    vals = [auc(y[tasks == task], s[tasks == task]) for task in np.unique(tasks)]
    vals = [v for v in vals if not math.isnan(v)]
    return (float(np.mean(vals)) if vals else float("nan")), len(vals)


def boot(groups: np.ndarray, stat, rng, n: int = BOOT) -> tuple[float, float]:
    """95% interval of stat(index) over runs resampled with replacement."""
    runs = np.unique(groups)
    idx = {r: np.flatnonzero(groups == r) for r in runs}
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[r] for r in rng.choice(runs, len(runs), replace=True)])
        v = stat(ii)
        if not math.isnan(v):
            vals.append(v)
    if len(vals) < n // 10:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def folds_by_run(groups: np.ndarray, strata: np.ndarray, k: int, rng) -> np.ndarray:
    """A fold for every sample; all steps of a run share one, and each stratum's runs are dealt
    across the folds in turn."""
    runs = {}
    for g, s in zip(groups, strata):
        runs.setdefault(g, s)
    fold_of, i = {}, int(rng.integers(k))
    for s in sorted(set(runs.values())):
        members = [g for g, x in runs.items() if x == s]
        rng.shuffle(members)
        for g in members:
            fold_of[g] = i % k
            i += 1
    return np.array([fold_of[g] for g in groups])


# ---------------------------------------------------------------- the probe
class Dual:
    """Ridge in its dual form on one feature matrix, for any split of its rows: the kernel is
    computed once, and each fit solves a system the size of the training set."""

    def __init__(self, Z: np.ndarray):
        Z = Z.astype(np.float64)
        self.G = Z @ Z.T

    def _fit(self, tr, y, lam):
        Gss = self.G[np.ix_(tr, tr)]
        r = Gss.mean(1)
        m = r.mean()
        K = Gss - r[:, None] - r[None, :] + m
        lam = lam * max(np.trace(K) / len(tr), 1e-9)
        a = np.linalg.solve(K + lam * np.eye(len(tr)), y[tr] - y[tr].mean())
        return a, r, m

    def fit_predict(self, tr, te, y, lam):
        a, r, m = self._fit(tr, y, lam)
        Gts = self.G[np.ix_(te, tr)]
        return (Gts - Gts.mean(1)[:, None] - r[None, :] + m) @ a


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = X.astype(np.float32)
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return (X - mu) / sd, mu, sd


def _fold_rank(scores, te):
    """Held-out scores as within-fold ranks in (0, 1]."""
    return _ranks(scores) / len(te)


def nested(feats: list[Dual], y, groups, strata, seed: int, lams=LAMS) -> tuple[np.ndarray, list]:
    """Out-of-fold scores, the feature set and strength chosen inside each training fold."""
    rng = np.random.default_rng(seed)
    out = np.zeros(len(y))
    chosen = []
    outer = folds_by_run(groups, strata, K_OUTER, rng)
    for f in range(K_OUTER):
        te, tr = np.flatnonzero(outer == f), np.flatnonzero(outer != f)
        if not len(te):
            continue
        inner = folds_by_run(groups[tr], strata[tr], K_INNER, rng)
        best = None
        for li, D in enumerate(feats):
            for lam in lams:
                oof = np.zeros(len(tr))
                for g in range(K_INNER):
                    ite, itr = tr[inner == g], tr[inner != g]
                    if len(ite) and len(np.unique(y[itr])) == 2:
                        oof[inner == g] = _fold_rank(D.fit_predict(itr, ite, y, lam), ite)
                a = auc(y[tr], oof)
                a = -1.0 if math.isnan(a) else a
                if best is None or a > best[0]:
                    best = (a, li, lam)
        _, li, lam = best
        if len(np.unique(y[tr])) == 2:
            out[te] = _fold_rank(feats[li].fit_predict(tr, te, y, lam), te)
        else:
            out[te] = 0.5
        chosen.append((li, lam))
    return out, chosen


def _shuffle(y_run: dict, runs_by_stratum: dict, rng) -> dict:
    out = {}
    for members in runs_by_stratum.values():
        labels = [y_run[g] for g in members]
        rng.shuffle(labels)
        out.update(zip(members, labels))
    return out


def train(store, model_spec: str, target: str = "fails", study: str | None = None, perms: int = 20,
          seed: int = 0, progress=None) -> dict:
    """Train and evaluate a probe for `target` on this model's agent runs. Returns the report and,
    under "_probe", what saving it as a direction needs."""
    if target not in TARGETS:
        raise ValueError(f"target must be one of {sorted(TARGETS)}")
    say = progress or (lambda *_: None)
    t0 = time.time()
    metas = candidate_runs(store, model_spec, study)
    say("loading residuals", 0.02)
    X, rows = load_steps(store, metas)
    if X is None or not rows:
        raise ValueError("no agent runs of this model with an outcome and saved residuals")
    y = np.array([r[target] for r in rows], float)
    groups = np.array([r["run"] for r in rows])
    tasks = np.array([r["task"] for r in rows])
    run_y = {r["run"]: r[target] for r in rows}
    n_pos_runs = sum(run_y.values())
    if n_pos_runs < 2 or len(run_y) - n_pos_runs < 2:
        raise ValueError(f"need at least 2 runs with each outcome for {target!r}; have {n_pos_runs} of {len(run_y)}")
    strata = np.array([f"{r['task']}|{int(r[target])}" for r in rows])
    rng = np.random.default_rng(seed)
    n_layers = X.shape[1]

    say("building kernels", 0.05)
    duals, scale = [], []
    for layer in range(n_layers):
        Z, mu, sd = standardize(X[:, layer])
        duals.append(Dual(Z))
        scale.append((mu, sd))
    task_names = sorted(set(tasks))
    def table(names):
        F = np.array([[r[c] for c in names] for r in rows], float)
        if names is TRANSCRIPT:
            F = np.hstack([F, (tasks[:, None] == np.array(task_names)[None]).astype(float)])
        return Dual(standardize(F)[0])
    pos_dual, tr_dual = table(POSITION), table(TRANSCRIPT)

    say("probe: nested cross-validation", 0.12)
    probe, chosen = nested(duals, y, groups, strata, seed)
    say("baselines", 0.3)
    position, _ = nested([pos_dual], y, groups, strata, seed)
    transcript, _ = nested([tr_dual], y, groups, strata, seed)

    def summary(s):
        a = auc(y, s)
        lo, hi = boot(groups, lambda ii: auc(y[ii], s[ii]), rng)
        w, n_tasks = within_task_auc(y, s, tasks)
        wlo, whi = boot(groups, lambda ii: within_task_auc(y[ii], s[ii], tasks[ii])[0], rng)
        return {"auc": round(a, 4), "lo": round(lo, 4), "hi": round(hi, 4),
                "within_task": {"auc": round(w, 4), "lo": round(wlo, 4), "hi": round(whi, 4), "tasks": n_tasks}}

    report = {"target": target, "describes": TARGETS[target], "model": model_spec, "study": study,
              "runs": len(run_y), "positive_runs": n_pos_runs, "steps": len(rows), "layers_total": n_layers,
              "tasks": [{"task": task, "runs": len({r["run"] for r in rows if r["task"] == task}),
                         "positive_runs": len({r["run"] for r in rows if r["task"] == task and r[target]}),
                         "steps": sum(r["task"] == task for r in rows)} for task in task_names]}
    report["probe"] = summary(probe) | {"layers_chosen": dict(Counter(int(li) for li, _ in chosen)),
                                        "strengths_chosen": dict(Counter(float(lam) for _, lam in chosen))}
    report["baselines"] = {"position": summary(position), "transcript": summary(transcript)}
    lo, hi = boot(groups, lambda ii: auc(y[ii], probe[ii]) - auc(y[ii], transcript[ii]), rng)
    report["probe_minus_transcript"] = {"diff": round(auc(y, probe) - auc(y, transcript), 4),
                                        "lo": round(lo, 4), "hi": round(hi, 4)}

    say("layer by layer", 0.36)
    layers = []
    for layer, D in enumerate(duals):
        s, _ = nested([D], y, groups, strata, seed)
        a = auc(y, s)
        l_lo, l_hi = boot(groups, lambda ii: auc(y[ii], s[ii]), rng, n=200)
        layers.append({"layer": layer, "auc": round(a, 4), "lo": round(l_lo, 4), "hi": round(l_hi, 4)})
        say("layer by layer", 0.36 + 0.24 * (layer + 1) / n_layers)
    report["layers"] = layers

    leads = np.array([r["lead"] for r in rows])
    by_lead = []
    for L in range(0, int(leads.max()) + 1):
        m = leads == L
        if len(np.unique(y[m])) < 2 or m.sum() < 6:
            continue
        sub = np.flatnonzero(m)
        pa, ta = auc(y[m], probe[m]), auc(y[m], transcript[m])
        p_lo, p_hi = boot(groups[m], lambda ii: auc(y[sub][ii], probe[sub][ii]), rng, n=500)
        t_lo, t_hi = boot(groups[m], lambda ii: auc(y[sub][ii], transcript[sub][ii]), rng, n=500)
        d_lo, d_hi = boot(groups[m], lambda ii: auc(y[sub][ii], probe[sub][ii]) - auc(y[sub][ii], transcript[sub][ii]),
                          rng, n=500)
        by_lead.append({"lead": L, "steps": int(m.sum()), "positive": int(y[m].sum()),
                        "probe": {"auc": round(pa, 4), "lo": round(p_lo, 4), "hi": round(p_hi, 4)},
                        "transcript": {"auc": round(ta, 4), "lo": round(t_lo, 4), "hi": round(t_hi, 4)},
                        "diff": {"diff": round(pa - ta, 4), "lo": round(d_lo, 4), "hi": round(d_hi, 4)}})
    report["lead"] = by_lead
    report["lead_time"] = {
        "above_chance": max((b["lead"] for b in by_lead if b["probe"]["lo"] > 0.5), default=None),
        "above_transcript": max((b["lead"] for b in by_lead if b["diff"]["lo"] > 0), default=None)}

    # The nulls: the same nested pipeline on shuffled run labels.
    runs = sorted(run_y)
    by_all = {"all": runs}
    by_task = {}
    for g in runs:
        by_task.setdefault(next(r["task"] for r in rows if r["run"] == g), []).append(g)
    observed = auc(y, probe)
    nulls = {}
    for i, (name, strata_runs) in enumerate((("shuffled", by_all), ("shuffled_within_task", by_task))):
        vals = []
        for p in range(perms):
            yp_run = _shuffle(run_y, strata_runs, rng)
            yp = np.array([yp_run[g] for g in groups], float)
            if len(np.unique(yp)) < 2:
                continue
            sp = np.array([f"{task}|{int(v)}" for task, v in zip(tasks, yp)])
            s, _ = nested(duals, yp, groups, sp, seed)  # the same splits as the probe's: only the labels change
            vals.append(auc(yp, s))
            say(f"shuffled labels, {'within tasks' if i else 'across runs'} ({p + 1} of {perms})",
                0.6 + 0.38 * (i * perms + p + 1) / (2 * max(perms, 1)))
        vals = [v for v in vals if not math.isnan(v)]
        nulls[name] = ({"n": len(vals), "mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4),
                        "p95": round(float(np.percentile(vals, 95)), 4),
                        "p": round((1 + sum(v >= observed for v in vals)) / (1 + len(vals)), 4),
                        "selectivity": round(observed - float(np.mean(vals)), 4)} if vals else {"n": 0})
    report["controls"] = nulls
    report["notes"] = notes(report)
    report["seconds"] = round(time.time() - t0, 1)

    # What saving the probe as a direction needs: the layer chosen most often, refit on every run.
    layer = Counter(int(li) for li, _ in chosen).most_common(1)[0][0]
    lam = Counter(float(lam) for _, lam in chosen).most_common(1)[0][0]
    report["_probe"] = {"layer": layer, "lam": lam, "y": y, "X": X[:, layer], "scale": scale[layer]}
    say("done", 1.0)
    return report


def notes(r: dict) -> list[str]:
    """Plain-language reading of a report, cautious about what it can't show."""
    out = []
    p, t = r["probe"], r["baselines"]["transcript"]
    ctl = r["controls"]
    if r["runs"] < 30:
        out.append(f"Only {r['runs']} runs: intervals are wide, and a few runs can move the AUC a lot.")
    mixed = [x["task"] for x in r["tasks"] if 0 < x["positive_runs"] < x["runs"]]
    if not mixed:
        out.append("No task has both outcomes, so outcome and task are confounded: the probe can't be told apart "
                   "from a task detector. The within-task AUC is undefined.")
    elif len(mixed) < len(r["tasks"]):
        out.append(f"Outcomes vary within {len(mixed)} of {len(r['tasks'])} tasks ({', '.join(mixed)}); the within-task "
                   "AUC uses only those.")
    w = ctl.get("shuffled_within_task", {})
    if w.get("n") and w["p"] > 0.05:
        out.append("The probe doesn't beat labels shuffled within each task: what it reads may be which task this is.")
    d = r["probe_minus_transcript"]
    if d["lo"] > 0:
        out.append("The probe beats the transcript-only baseline overall.")
    elif d["hi"] < 0:
        out.append("The transcript-only baseline beats the probe overall.")
    else:
        out.append("The probe and the transcript-only baseline aren't distinguishable overall.")
    if p["within_task"]["auc"] == p["within_task"]["auc"] and t["within_task"]["auc"] == t["within_task"]["auc"]:
        out.append(f"Within tasks: probe {p['within_task']['auc']:.2f}, transcript {t['within_task']['auc']:.2f}.")
    return out


def probe_direction(report: dict):
    """The saved probe as a unit vector in the residual stream at its layer (pointing toward the
    target), and the samples to z-score it on."""
    pr = report["_probe"]
    Z, mu, sd = standardize(pr["X"])
    D = Dual(Z)
    tr = np.arange(len(pr["y"]))
    a, _, _ = D._fit(tr, pr["y"], pr["lam"])
    w = (Z - Z.mean(0)).T.astype(np.float64) @ a  # primal weights on standardized features
    raw = w / pr["scale"][1]
    norm = np.linalg.norm(raw)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("the probe has no direction to save")
    return pr["layer"], (raw / norm).astype(np.float32), pr["X"].astype(np.float32)


def _plain(x):
    """JSON-safe: NumPy numbers as Python ones, and an undefined number (NaN) as null."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def public(report: dict) -> dict:
    """The report without what saving the probe needs, ready for JSON."""
    return _plain({k: v for k, v in report.items() if not k.startswith("_")})
