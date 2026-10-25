"""Study 1: the moment of giving up. Does a probe on the model's internals forecast that an agent
run will fail, some steps before it ends, better than what the transcript already shows?

    scripts/run.sh -m brainscan.studies.study1 collect --repeats 8   # record agent runs (needs the recorder)
    scripts/run.sh -m brainscan.studies.study1 analyze               # train and evaluate probes (CPU only)

collect runs the mini agent on the tasks in brainscan/harness/tasks, one repeat of every task at a
time, tagging each run "study1" in its run_meta. analyze reads those runs from the store, trains a
probe for each target with recorder/decoding.py (nested cross-validation by run, shuffled-label
nulls across and within tasks, position-only and transcript-only baselines, AUC by steps before
the end) and writes reports/study1.md with the numbers in reports/study1.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from ..config import PROJECT_ROOT, Settings
from ..recorder import decoding
from ..recorder.store import RunStore

STUDY = "study1"
TASKS = ["fix_fizzbuzz", "toml_lite", "roman_numerals", "wrap_text", "semver_compare", "impossible_clamp"]


def collect(args) -> None:
    from ..harness.mini_agent import run_task

    tasks = args.tasks.split(",") if args.tasks else TASKS
    done = []
    for rep in range(args.repeats):
        for task in tasks:
            t0 = time.time()
            try:
                r = run_task(task, args.base_url, args.max_steps, args.temperature, None, args.thinking,
                             args.max_tokens, STUDY)
            except Exception as exc:  # one broken run shouldn't end the collection
                r = {"task": task, "error": f"{type(exc).__name__}: {exc}"}
            r["seconds"] = round(time.time() - t0)
            done.append(r)
            print(f"[study1] {rep + 1}/{args.repeats} {task}: "
                  f"{r.get('error') or ('PASS' if r.get('success') else 'FAIL')} in {r['seconds']}s", flush=True)
    print(json.dumps(done, indent=2))


def _n(v, spec: str = ".2f") -> str:
    """A number, or a dash when it's undefined (null in the report)."""
    return "–" if v is None or v != v else format(v, spec)


def _fmt(x: dict, key: str = "auc") -> str:
    v, lo, hi = x.get(key), x.get("lo"), x.get("hi")
    if v is None or v != v:
        return "–"
    return f"{v:.2f} [{_n(lo)}, {_n(hi)}]" if lo is not None and hi is not None else f"{v:.2f}"


def markdown(reports: list[dict], runs: list[dict]) -> str:
    out = ["# Study 1: does a probe forecast an agent's failure before it happens?", ""]
    model = reports[0]["model"] if reports else "?"
    by_task = Counter((m["harness"]["task"], bool(m["outcome"].get("success"))) for m in runs)
    out += [f"Model: `{model}`. Runs: {len(runs)} agent runs of the mini agent tagged `{STUDY}`, recorded "
            f"{time.strftime('%Y-%m-%d', time.localtime(min(m.get('created', 0) for m in runs)))} to "
            f"{time.strftime('%Y-%m-%d', time.localtime(max(m.get('created', 0) for m in runs)))}.", "",
            "**Question.** At each step of an agent run, the residual stream just before the model replies is read "
            "by a linear probe trained to say whether the run will end failing. Does it see failure coming, and "
            "earlier or better than a model of what the transcript shows (the task, steps, tool errors, failing "
            "test runs, edits to tests, repeated calls)?", "",
            "| Task | Runs | Passed | Failed |", "|---|---|---|---|"]
    for task in sorted({t for t, _ in by_task}):
        p, f = by_task[(task, True)], by_task[(task, False)]
        out.append(f"| `{task}` | {p + f} | {p} | {f} |")
    for r in reports:
        p, b, c = r["probe"], r["baselines"], r["controls"]
        out += ["", f"## Target: {r['target']} ({r['describes']})", "",
                f"{r['runs']} runs ({r['positive_runs']} positive), {r['steps']} steps. AUCs are out of fold with 95% "
                "bootstrap intervals over runs.", "",
                "| Readout | AUC | Within tasks |", "|---|---|---|",
                f"| Probe (layer and strength chosen in each training fold) | {_fmt(p)} | {_fmt(p['within_task'])} |",
                f"| Transcript only | {_fmt(b['transcript'])} | {_fmt(b['transcript']['within_task'])} |",
                f"| Position only | {_fmt(b['position'])} | {_fmt(b['position']['within_task'])} |", "",
                f"Probe minus transcript: {_n(r['probe_minus_transcript']['diff'], '+.2f')} "
                f"[{_n(r['probe_minus_transcript']['lo'], '+.2f')}, {_n(r['probe_minus_transcript']['hi'], '+.2f')}]. "
                f"Layers chosen: {', '.join(f'L{k} ×{v}' for k, v in sorted(p['layers_chosen'].items()))}.", "",
                "| Null (same pipeline, shuffled run labels) | Shuffles | Mean AUC | 95th pct | p | Selectivity |",
                "|---|---|---|---|---|---|"]
        for name, label in (("shuffled", "across all runs"), ("shuffled_within_task", "within each task")):
            x = c.get(name, {})
            out.append(f"| {label} | {x.get('n', 0)} | {_n(x.get('mean'))} | {_n(x.get('p95'))} | "
                       f"{_n(x.get('p'), '.3f')} | {_n(x.get('selectivity'), '+.2f')} |" if x.get("n") else
                       f"| {label} | 0 | – | – | – | – |")
        out += ["", "**By steps before the end of the run** (0 is the last step):", "",
                "| Steps before end | Steps | Positive | Probe | Transcript | Probe − transcript |", "|---|---|---|---|---|---|"]
        for x in r["lead"]:
            d = x["diff"]
            out.append(f"| {x['lead']} | {x['steps']} | {x['positive']} | {_fmt(x['probe'])} | {_fmt(x['transcript'])} | "
                       f"{_n(d['diff'], '+.2f')} [{_n(d['lo'], '+.2f')}, {_n(d['hi'], '+.2f')}] |")
        lt = r["lead_time"]
        out += ["", f"Lead time: the probe's interval stays above chance up to **{lt['above_chance']}** steps before the end"
                f" ({'never' if lt['above_chance'] is None else 'counting from the last step'}); it beats the transcript "
                f"baseline up to **{lt['above_transcript']}** steps before the end.", "",
                "**Probe AUC by layer** (strength chosen in each training fold):", "",
                "| Layer | " + " | ".join(str(x["layer"]) for x in r["layers"][::4]) + " |",
                "|---|" + "---|" * len(r["layers"][::4]),
                "| AUC | " + " | ".join(_n(x["auc"]) for x in r["layers"][::4]) + " |", "",
                "**Reading it.**", ""] + [f"- {n}" for n in r["notes"]]
    out += ["", "## Method", "",
            "- **Samples.** One per agent step: the residual stream at every layer at the last prompt token, just "
            "before the model writes its reply (the recorder's `pre_gen` decision residual), with the transcript "
            "features of everything read by then. Labels are per run and read at every step.",
            "- **Probe.** Ridge regression on residuals standardized over all samples (label-free), fitted in its dual "
            "form. Five outer folds by run (no run on both sides), stratified by task and outcome; the layer and ridge "
            "strength are chosen by four inner folds inside each training fold. Held-out scores are ranked within "
            "their fold before pooling.",
            "- **Controls.** The same nested pipeline on run labels shuffled across runs and within each task. The "
            "within-task shuffle is the null for a probe that only recognizes the task. Position-only (step, tokens) "
            "and transcript-only baselines use the same cross-validation.",
            "- **Intervals.** 95% bootstrap over runs (whole runs resampled).",
            "- **Limits.** Small numbers of runs give wide intervals. One model, one harness, six tasks. The probe "
            "reads a single position per step. A forecast is a correlation inside the model's state, not evidence "
            "about what the model experiences."]
    return "\n".join(out) + "\n"


def analyze(args) -> None:
    settings = Settings()
    store = RunStore(Path(args.data) if args.data else settings.data_dir, settings.disk_budget_gb)
    tagged = [m for m in store.list_runs() if (m.get("harness") or {}).get("study") == STUDY and m.get("outcome")]
    if not tagged:
        raise SystemExit(f"no runs tagged {STUDY} with an outcome; run `collect` first")
    model = args.model or Counter(m["model"]["spec"] for m in tagged).most_common(1)[0][0]
    runs = [m for m in tagged if m["model"]["spec"] == model]
    reports = []
    for target in args.targets.split(","):
        try:
            r = decoding.train(store, model, target, STUDY, args.perms, seed=args.seed,
                               progress=lambda step, f: print(f"[study1] {target}: {step} {f:.0%}", flush=True))
        except ValueError as exc:
            print(f"[study1] {target}: skipped ({exc})", flush=True)
            continue
        reports.append(decoding.public(r))
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "study1.json").write_text(json.dumps({"model": model, "runs": [m["id"] for m in runs], "reports": reports},
                                                indent=2))
    (out / "study1.md").write_text(markdown(reports, runs))
    print(f"[study1] wrote {out / 'study1.md'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="record agent runs through the recorder's endpoint")
    c.add_argument("--repeats", type=int, default=8)
    c.add_argument("--tasks", default="", help=f"comma-separated (default: {','.join(TASKS)})")
    c.add_argument("--base-url", default="http://localhost:8765/v1")
    c.add_argument("--max-steps", type=int, default=12)
    c.add_argument("--temperature", type=float, default=0.8)
    c.add_argument("--thinking", choices=["default", "on", "off"], default="off")
    c.add_argument("--max-tokens", type=int, default=2048)
    a = sub.add_parser("analyze", help="train and evaluate the probes, write reports/study1.md")
    a.add_argument("--targets", default="fails,gives_up")
    a.add_argument("--perms", type=int, default=100)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--model", default="", help="model spec (default: the one most of the runs used)")
    a.add_argument("--data", default="", help="data folder (default: the recorder's)")
    args = parser.parse_args()
    collect(args) if args.cmd == "collect" else analyze(args)


if __name__ == "__main__":
    main()
