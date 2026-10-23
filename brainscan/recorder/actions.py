"""Actions: things to do to the model, each tried and measured before it is kept.

Forget a fact. The fact is a lead-in and the answer the model gives after it ("The capital of
Australia is" -> "Canberra"), typed, or a word the model wrote in a recorded run. Four steps:

  target   rank the units (attention heads, MLP neuron clusters) by their direct effect on the
           answer at the lead-in's last token, then knock out the top 1, 2, 4 ... 32 to see how
           many it takes before the model no longer gives the answer. The smallest such set is
           suggested. "No longer gives it": the answer loses first place; for an answer that wasn't
           first to begin with, it keeps under a tenth of its probability (and the UI says the
           model wasn't stating the fact after that lead-in).
  try      greedy answers with and without the units for every way of asking: lead-ins continued
           as text, and questions asked as chat (thinking off).
  measure  the answer's probability before and after on every lead-in; the same number of random
           units removed instead, as a control (if random units remove the fact too, it wasn't
           found in these units); facts to keep, and the saved behavior checks, before and after;
           and drift on neutral reference text (KL and perplexity). A plain verdict sums it up.
  keep     is the existing grouped lesion (anatomy.lesion_units through the edit stack). Knocking
           out a unit is the same as zeroing its output weights, so what was measured is what is
           kept.

Everything runs on the weights as they are now (with any applied edits), and holds the model so no
request runs while units are knocked out.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import numpy as np
import torch

from . import workbench

CURVE = (1, 2, 4, 8, 16, 32)   # set sizes tried when locating a fact
CANDIDATES = 32                # units ranked for removal
CONTROL_SEEDS = 3              # random unit sets per measurement
MAX_LEADS, MAX_QUESTIONS, MAX_KEEP = 8, 4, 16
LEAD_TOKENS, QUESTION_TOKENS = 20, 48
BROKEN_DROP = 0.2              # a fact to keep "broke" if it lost first place or this much probability
DRIFT_NOTICEABLE = 0.05        # KL per token on neutral text, in nats
FADED = 0.1                    # an answer that wasn't first counts as gone below this share of its probability


@dataclass
class Fact:
    ids: list[int]
    target: int
    answer: str          # as the user wrote it, or the token's text
    lead: str            # the lead-in as shown
    question: str | None
    source: dict


def _lines(value, limit: int, what: str) -> list[str]:
    items = value.splitlines() if isinstance(value, str) else list(value or [])
    out = []
    for x in items:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    if len(out) > limit:
        raise ValueError(f"at most {limit} {what}")
    return out


def _pair(line: str) -> tuple[str, str]:
    for sep in ("→", "->", "=>", "|"):
        if sep in line:
            a, b = line.split(sep, 1)
            if a.strip() and b.strip():
                return a.strip(), b.strip()
    raise ValueError(f"write a fact to keep as 'lead-in → answer': {line!r}")


def fact(recorder, body: dict) -> Fact:
    """The fact to forget: {prompt, answer} typed, or {run, t, target}: the word written after tape
    slot t of a recorded run."""
    s, tr = recorder.s, recorder.tracer
    if body.get("run") not in (None, ""):
        run = tr._run(str(body["run"]))
        t, tid = int(body["t"]), int(body["target"])
        ids, _, _ = tr.context(run, t)
        try:
            question = workbench.conversation(run)[-1]["content"]
        except ValueError:
            question = None
        return Fact(ids, tid, s.tokenizer.decode([tid]).strip() or s.token_str(tid),
                    _clean(s.tokenizer.decode(ids[-24:], skip_special_tokens=False)), question,
                    {"run": run.id, "t": t, "target": tid})
    prompt, answer = str(body.get("prompt") or "").rstrip(), str(body.get("answer") or "").strip()
    if not prompt.strip() or not answer:
        raise ValueError("name the fact: a lead-in and the answer the model gives after it")
    return Fact(workbench.prompt_ids(s, prompt), workbench.answer_token(s, prompt, answer), answer, prompt, None,
                {"prompt": prompt, "answer": answer})


def _read(s, logits: torch.Tensor, tid: int) -> dict:
    probs = torch.softmax(logits, -1)
    top = int(logits.argmax())
    return {"prob": round(float(probs[tid]), 5), "rank": int((logits > logits[tid]).sum()) + 1,
            "top": s.token_str(top), "top_prob": round(float(probs[top]), 5)}


def gone(before: dict, after: dict) -> bool:
    """The model no longer gives the answer: it lost first place, or, if it wasn't first to begin
    with, most of its probability."""
    if before["rank"] == 1:
        return after["rank"] > 1
    return after["prob"] < before["prob"] * FADED


def _unit(an, g: int, direct: float) -> dict:
    lm, kind, i = an.locate(g)
    return {"g": int(g), "name": f"L{lm.index} {'h' if kind == 'head' else 'c'}{i}", "kind": kind,
            "layer": lm.index, "direct": round(float(direct), 4)}


def _leads(s, f: Fact, extra) -> list[dict]:
    """The fact's own lead-in, then the other ways of leading into the same answer."""
    out = [{"text": f.lead, "ids": f.ids, "target": f.target}]
    for line in _lines(extra, MAX_LEADS, "other lead-ins"):
        line = line.rstrip()
        out.append({"text": line, "ids": workbench.prompt_ids(s, line),
                    "target": workbench.answer_token(s, line, f.answer)})
    return out


# ---- target -------------------------------------------------------------------------------------

def forget_target(recorder, body: dict) -> dict:
    """Rank the units that push the answer up, and how many it takes to knock it off first place."""
    s, tr = recorder.s, recorder.tracer
    f = fact(recorder, body)
    t0 = time.perf_counter()
    cap = tr.capture_ids(f.ids)
    contrib, _, _ = tr.attribute(cap, f.target)
    c = contrib.cpu().numpy()
    ranked = [int(g) for g in np.argsort(-c) if c[g] > 0][:CANDIDATES]
    if not ranked:
        raise ValueError("no unit pushes this answer up, so there is nothing to remove")
    before = _read(s, cap.logits, f.target)
    curve = []
    with tr._model():
        for k in CURVE:
            if k > len(ranked):
                break
            after = _read(s, tr.last_logits(f.ids, ranked[:k]), f.target)
            curve.append({"k": k, "gone": gone(before, after)} | after)
    flip = next((p["k"] for p in curve if p["gone"]), None)
    probs = torch.softmax(cap.logits, -1)
    top_p, top_i = probs.topk(5)
    return {
        "fact": {"lead": f.lead, "answer": f.answer, "token": s.token_str(f.target), "question": f.question,
                 "source": f.source, **before, "first": before["rank"] == 1,
                 "alternatives": [{"text": s.token_str(int(i)), "prob": round(float(p), 5)}
                                  for p, i in zip(top_p.tolist(), top_i.tolist())]},
        "units": [_unit(tr.an, g, c[g]) for g in ranked],
        "curve": curve, "suggested": flip or curve[-1]["k"], "flipped": flip is not None,
        "weights_edited": bool(recorder.edits.applied),
        "seconds": round(time.perf_counter() - t0, 2),
    }


# ---- try ----------------------------------------------------------------------------------------

def _continue(s, ids: list[int], n: int) -> str:
    return s.tokenizer.decode(workbench._greedy_ids(s, ids, n), skip_special_tokens=True).strip()


def forget_try(recorder, body: dict) -> dict:
    """Greedy answers with and without the units: every lead-in continued, every question asked."""
    s, tr = recorder.s, recorder.tracer
    f = fact(recorder, body)
    units = tr.check_units(body.get("units"))
    leads = _leads(s, f, body.get("leads"))
    questions = _lines(body.get("questions"), MAX_QUESTIONS, "questions")
    says = lambda text: f.answer.lower() in text.lower()
    t0, rows = time.perf_counter(), []
    with tr._model():
        for kind, text, ids, n in ([("lead", ld["text"], ld["ids"], LEAD_TOKENS) for ld in leads] +
                                   [("question", q, workbench.prompt_ids(s, q, chat=True), QUESTION_TOKENS) for q in questions]):
            original = _continue(s, ids, n)
            with tr.knocked_out(units):
                edited = _continue(s, ids, n)
            rows.append({"kind": kind, "prompt": text, "original": original, "edited": edited,
                         "said_before": says(original), "says_now": says(edited)})
    return {"rows": rows, "answer": f.answer, "units": len(units), "seconds": round(time.perf_counter() - t0, 2)}


# ---- measure ------------------------------------------------------------------------------------

def _random_units(an, units: list[int], seed: int) -> list[int]:
    """As many heads and clusters as `units` has, drawn at random from the rest of the model."""
    chosen = set(units)
    heads = [g for lm in an.layers for g in range(lm.offset, lm.offset + lm.n_heads) if g not in chosen]
    clusters = [g for lm in an.layers for g in range(lm.offset + lm.n_heads, lm.offset + lm.n_units) if g not in chosen]
    n_heads = sum(an.locate(g)[1] == "head" for g in units)
    rng = np.random.default_rng(seed)
    pick = list(rng.choice(heads, min(n_heads, len(heads)), replace=False)) if n_heads else []
    n_cl = len(units) - n_heads
    pick += list(rng.choice(clusters, min(n_cl, len(clusters)), replace=False)) if n_cl else []
    return sorted(int(g) for g in pick)


def _clean(label: str) -> str:
    """A check's label without chat-template markers."""
    label = re.sub(r"<\|[^|>]*\|>\s*(?:system|user|assistant)?|</?think>", " ", label)
    return re.sub(r"\s+", " ", label).strip(" …")


def _keep_rows(recorder, keep, include_checks: bool) -> list[dict]:
    s, tr = recorder.s, recorder.tracer
    rows = []
    for line in _lines(keep, MAX_KEEP, "facts to keep"):
        lead, answer = _pair(line)
        rows.append({"label": f"{lead} → {answer}", "source": "typed", "ids": workbench.prompt_ids(s, lead),
                     "target": workbench.answer_token(s, lead, answer)})
    if include_checks:
        for c in workbench.list_checks(recorder):
            row = {"label": _clean(c["label"]), "source": "check", "target": c["target"]}
            try:
                row["ids"] = workbench._check_ids(s, c) if c["kind"] == "text" else tr.context(tr._run(c["run"]), c["t"])[0]
            except (KeyError, ValueError) as exc:
                row["error"] = str(exc)
            rows.append(row)
    return rows


@torch.no_grad()
def _drift(recorder, units: list[int]) -> dict:
    """KL(intact || without the units) per token, and perplexity, on neutral reference text."""
    s, tr = recorder.s, recorder.tracer
    kl = nll_o = nll_e = 0.0
    n = 0
    for text in workbench.REFERENCE_TEXTS:
        x = torch.tensor([workbench._reference_ids(s, text)], device=s.device)
        lp_o = torch.log_softmax(s.model(input_ids=x, use_cache=False).logits[0, :-1].float(), -1)
        with tr.knocked_out(units):
            lp_e = torch.log_softmax(s.model(input_ids=x, use_cache=False).logits[0, :-1].float(), -1)
        nxt = x[0, 1:]
        kl += float((lp_o.exp() * (lp_o - lp_e)).sum())
        nll_o -= float(lp_o.gather(-1, nxt[:, None]).sum())
        nll_e -= float(lp_e.gather(-1, nxt[:, None]).sum())
        n += len(nxt)
        del lp_o, lp_e
    return {"kl_per_token": round(kl / max(n, 1), 5), "perplexity_before": round(math.exp(nll_o / max(n, 1)), 3),
            "perplexity_after": round(math.exp(nll_e / max(n, 1)), 3), "tokens": n}


def _verdict(leads: list[dict], keep: list[dict], drift: dict) -> dict:
    n, n_gone = len(leads), sum(r["gone"] for r in leads)
    control = sum(r["control_gone"] for r in leads) / max(n, 1)
    broken = [k for k in keep if k.get("before") and (
        (k["before"]["rank"] == 1 and k["after"]["rank"] > 1) or k["before"]["prob"] - k["after"]["prob"] > BROKEN_DROP)]
    kl = drift["kl_per_token"]
    if not n_gone:
        return {"status": "kept", "text": "The model still gives the answer. Remove more units, or pick a different fact."}
    if control >= 0.5:
        return {"status": "unspecific", "text": "Random units of the same size remove it too, so these units aren't "
                                                "where the fact lives. Try fewer units."}
    issues = []
    if n_gone < n:
        issues.append(f"it's gone after {n_gone} of {n} lead-ins")
    if broken:
        issues.append(f"{len(broken)} fact{'s' if len(broken) != 1 else ''} to keep broke")
    if kl > DRIFT_NOTICEABLE:
        issues.append(f"neutral text drifts (KL {kl:.3f} per token)")
    if issues:
        return {"status": "partial" if n_gone < n and not broken else "costly",
                "text": "Partly: " + "; ".join(issues) + "."}
    return {"status": "works", "text": "Works: the answer is gone after every lead-in, the facts to keep survive, "
                                       "random units of the same size leave it alone, and neutral text barely moves."}


def forget_measure(recorder, body: dict) -> dict:
    s, tr = recorder.s, recorder.tracer
    f = fact(recorder, body)
    units = tr.check_units(body.get("units"))
    leads = _leads(s, f, body.get("leads"))
    for ld in leads:
        ld["source"] = "lead"
    # A saved check that asks for the same answer is another way of asking for the fact, not a fact
    # to keep; and the same context with the same answer is measured once.
    seen, keep = {(tuple(ld["ids"]), ld["target"]) for ld in leads}, []
    for k in _keep_rows(recorder, body.get("keep"), bool(body.get("checks", True))):
        key = (tuple(k["ids"]), k["target"]) if k.get("ids") else None
        if key in seen:
            continue
        if key and k["source"] == "check" and k["target"] == f.target:
            leads.append({"text": k["label"], "ids": k["ids"], "target": k["target"], "source": "check"})
        else:
            keep.append(k)
        if key:
            seen.add(key)
    controls = [_random_units(tr.an, units, seed) for seed in range(CONTROL_SEEDS)]
    t0 = time.perf_counter()
    with tr._model():
        lead_rows = []
        for ld in leads:
            before = _read(s, tr.last_logits(ld["ids"]), ld["target"])
            after = _read(s, tr.last_logits(ld["ids"], units), ld["target"])
            ctrl = [_read(s, tr.last_logits(ld["ids"], cu), ld["target"]) for cu in controls]
            lead_rows.append({"prompt": ld["text"], "source": ld["source"], "token": s.token_str(ld["target"]),
                              "before": before, "after": after, "gone": gone(before, after),
                              "control": {"prob": round(sum(c["prob"] for c in ctrl) / len(ctrl), 5),
                                          "rank": int(np.median([c["rank"] for c in ctrl]))},
                              "control_gone": sum(gone(before, c) for c in ctrl) / len(ctrl)})
        keep_rows = []
        for k in keep:
            row = {"label": k["label"], "source": k["source"], "token": s.token_str(k["target"])}
            if k.get("error"):
                row["error"] = k["error"]
            else:
                row["before"] = _read(s, tr.last_logits(k["ids"]), k["target"])
                row["after"] = _read(s, tr.last_logits(k["ids"], units), k["target"])
            keep_rows.append(row)
        drift = _drift(recorder, units)
    return {
        "leads": lead_rows, "keep": keep_rows, "drift": drift,
        "control": {"seeds": CONTROL_SEEDS, "units": len(units)},
        "verdict": _verdict(lead_rows, keep_rows, drift),
        "units": [{k: v for k, v in _unit(tr.an, g, 0.0).items() if k != "direct"} for g in units],
        "weights_edited": bool(recorder.edits.applied),
        "seconds": round(time.perf_counter() - t0, 2),
    }
