"""Tracing: explain what the model did at one token of a recorded run.

For token t of a run, the tracer re-reads the run's context up to t (the exact tokens the model saw)
and captures, at t's position:

  attribution  how much each unit (attention head, MLP neuron cluster) and each MLP neuron pushed the
               logit of the next token up or down. A unit's write into the residual stream is
               projected onto the next token's unembedding row through the final norm, with the norm's
               scale frozen at the value it had. The final residual is the embedding plus every
               block's write, so the contributions add up to the logit, up to the norm's epsilon and any
               output biases (reported as "rest").
  neurons      each MLP neuron's activation at t and its share of that attribution.
  attention    for softmax-attention heads, how the head spread its attention over earlier tokens at t.
               Linear-attention heads (gated delta net) keep a recurrent state instead of an attention
               matrix, so they have none.

Tracing uses the current weights with runtime steering paused, and says when the run was recorded
under different weight edits or with steering on.
"""

from __future__ import annotations

import contextlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

CACHE_SIZE = 6
TOP_UNITS = 24


class Busy(Exception):
    """The model is generating; tracing needs it to itself."""


@dataclass
class Capture:
    pos: int
    ids: list[int]
    head_in: dict[int, torch.Tensor] = field(default_factory=dict)  # layer -> (heads*hd,) output-projection input
    mlp_in: dict[int, torch.Tensor] = field(default_factory=dict)   # layer -> (neurons,) down-projection input
    attn: dict[int, torch.Tensor] = field(default_factory=dict)     # layer -> (heads, positions) last-query weights
    embed: torch.Tensor | None = None                                # (d,) embedding written at this position
    final: torch.Tensor | None = None                                # (d,) input to the final norm
    logits: torch.Tensor | None = None                               # (vocab,) fp32
    seconds: float = 0.0


class Tracer:
    def __init__(self, recorder):
        self.rec, self.s, self.an = recorder, recorder.s, recorder.anatomy
        self._cache: OrderedDict = OrderedDict()
        self._gain: torch.Tensor | None = None
        self._capturing = False

    # ---- context ---------------------------------------------------------------------

    def _run(self, run_id: str):
        run = self.rec.get_run(run_id)
        if run is None:
            raise KeyError(f"no run {run_id}")
        spec = (run.meta.get("model") or {}).get("spec")
        if spec and spec != self.s.spec:
            raise ValueError(f"run was recorded with {spec}; the loaded model is {self.s.spec}")
        return run

    @staticmethod
    def context(run, t: int) -> tuple[list[int], int, int]:
        """(token ids at positions 0..pos, pos, request) the model saw when it read tape slot t.

        A request records only the positions past the prefix it shares with the run's previous
        request; earlier positions come from the requests that recorded them, which saw the same
        tokens there."""
        if not 0 <= t < len(run.tape):
            raise KeyError(f"no token {t}")
        r = run.tape[t]
        req, pos = r["req"], r["pos"]
        ctx: dict[int, int] = {}
        for rec in run.tape:
            if rec["req"] <= req:
                ctx[rec["pos"]] = rec["tok"]
        missing = [p for p in range(pos + 1) if p not in ctx]
        if missing:
            raise ValueError(f"the run's tape lacks positions {missing[:5]} of this token's context")
        return [ctx[p] for p in range(pos + 1)], pos, req

    @staticmethod
    def next_token(run, req: int, pos: int) -> int | None:
        return next((r["tok"] for r in run.tape if r["req"] == req and r["pos"] == pos + 1), None)

    # ---- the capture forward ------------------------------------------------------------

    @contextlib.contextmanager
    def _model(self):
        if not self.rec.lock.acquire(timeout=1.0):
            raise Busy("the model is generating; trace again when the request finishes")
        try:
            with torch.no_grad(), self.rec.tap.paused():
                yield
        finally:
            self.rec.release_memory(below_gb=2.0)
            self.rec.lock.release()

    def _hooks(self, cap: Capture) -> list:
        handles = []

        def last_input(store, key):
            def fn(module, args):
                if self._capturing:
                    store[key] = args[0][0, -1].detach().float()
            return fn

        for lm in self.an.layers:
            if lm.heads:
                handles.append(lm.heads[0].register_forward_pre_hook(last_input(cap.head_in, lm.index)))
            if lm.down is not None:
                handles.append(lm.down.register_forward_pre_hook(last_input(cap.mlp_in, lm.index)))

        def embed_hook(module, args, output):
            if self._capturing:
                cap.embed = output[0, -1].detach().float()

        def final_hook(module, args):
            if self._capturing:
                cap.final = args[0][0, -1].detach().float()

        handles.append(self.s.model.get_input_embeddings().register_forward_hook(embed_hook))
        handles.append(self.s.norm.register_forward_pre_hook(final_hook))
        return handles

    @contextlib.contextmanager
    def _attention_capture(self, cap: Capture):
        """Route softmax attention through a wrapper that keeps the last query's weights."""
        impl = self.s.model.config._attn_implementation
        local = ALL_ATTENTION_FUNCTIONS._local_mapping
        before = local.get(impl)
        original = ALL_ATTENTION_FUNCTIONS[impl]

        def attend(module, query, key, value, attention_mask, **kwargs):
            if self._capturing and getattr(module, "layer_idx", None) is not None:
                q = query[0, :, -1, :].float()                     # (heads, hd)
                k = key[0].float()                                   # (kv heads, positions, hd)
                if k.shape[0] != q.shape[0]:
                    k = k.repeat_interleave(q.shape[0] // k.shape[0], dim=0)
                scale = kwargs.get("scaling") or getattr(module, "scaling", q.shape[-1] ** -0.5)
                cap.attn[module.layer_idx] = torch.softmax(torch.einsum("hd,hnd->hn", q, k) * scale, dim=-1)
            return original(module, query, key, value, attention_mask, **kwargs)

        local[impl] = attend
        try:
            yield
        finally:
            if before is None:
                local.pop(impl, None)
            else:
                local[impl] = before

    def capture(self, run, t: int) -> Capture:
        ids, pos, _ = self.context(run, t)
        key = (run.id, t, self.rec.edits.version, len(ids))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        cap = Capture(pos=pos, ids=ids)
        t0 = time.perf_counter()
        with self._model():
            handles = self._hooks(cap)
            try:
                with self._attention_capture(cap):
                    kv, chunk = None, self.s.prefill_chunk
                    for cs in range(0, len(ids), chunk):
                        self._capturing = cs + chunk >= len(ids)
                        out = self.s.model(input_ids=torch.tensor([ids[cs:cs + chunk]], device=self.s.device),
                                           past_key_values=kv, use_cache=True, logits_to_keep=1)
                        kv = out.past_key_values
                    cap.logits = out.logits[0, -1].float()
                    del out, kv
            finally:
                self._capturing = False
                for h in handles:
                    h.remove()
        cap.seconds = round(time.perf_counter() - t0, 3)
        self._cache[key] = cap
        while len(self._cache) > CACHE_SIZE:
            self._cache.popitem(last=False)
        return cap

    # ---- attribution --------------------------------------------------------------------

    def _final_gain(self) -> torch.Tensor:
        """Per-dimension gain of the final norm (what it multiplies a unit-RMS vector by)."""
        if self._gain is None:
            norm, d = self.s.norm, self.s.d_model
            if isinstance(norm, nn.LayerNorm):
                self._gain = norm.weight.detach().float()
            else:
                with torch.no_grad():
                    try:  # fp32 in keeps the gain at full precision; RMS norms upcast internally
                        self._gain = norm(torch.ones(1, d, device=self.s.device)).float()[0]
                    except RuntimeError:
                        p = next(norm.parameters())
                        self._gain = norm(torch.ones(1, d, device=self.s.device, dtype=p.dtype)).float()[0]
        return self._gain

    def _direction(self, cap: Capture, target: int) -> torch.Tensor:
        """The residual-space direction whose dot product with the final residual is the target's logit,
        with the final norm's scale frozen at this position."""
        norm = self.s.norm
        x = cap.final
        eps = float(getattr(norm, "eps", getattr(norm, "variance_epsilon", 1e-6)))
        if isinstance(norm, nn.LayerNorm):
            scale = (x - x.mean()).pow(2).mean().add(eps).rsqrt()
        else:
            scale = x.pow(2).mean().add(eps).rsqrt()
        return self._final_gain() * self.s.lm_head.weight[target].float() * scale

    def _resolve_target(self, run, req: int, pos: int, cap: Capture, target) -> int:
        if target not in (None, ""):
            text = str(target)
            if text.lstrip("-").isdigit():
                tid = int(text)
            else:
                enc = self.s.tokenizer.encode(text, add_special_tokens=False)
                if not enc:
                    raise ValueError(f"{text!r} does not tokenize to anything")
                tid = enc[0]
            if not 0 <= tid < self.s.vocab_size:
                raise ValueError(f"token id {tid} is outside the vocabulary")
            return tid
        nxt = self.next_token(run, req, pos)
        return nxt if nxt is not None else int(cap.logits.argmax())

    @torch.no_grad()
    def _neuron_contrib(self, lm, a: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return a * (u @ lm.down.weight.float())

    @torch.no_grad()
    def explain(self, run_id: str, t: int, target=None) -> dict:
        run = self._run(run_id)
        _, pos, req = self.context(run, t)
        cap = self.capture(run, t)
        tid = self._resolve_target(run, req, pos, cap, target)
        u = self._direction(cap, tid)
        U = self.an.n_units
        contrib = torch.zeros(U, device=self.s.device)
        layer_sum = []
        for lm in self.an.layers:
            heads = clusters = 0.0
            if lm.heads and lm.index in cap.head_in:
                o, H, hd = lm.heads
                z = cap.head_in[lm.index].view(H, hd)
                w = (u @ o.weight.float()).view(H, hd)
                c = (z * w).sum(-1)
                contrib[lm.offset:lm.offset + H] = c
                heads = float(c.sum())
            if lm.down is not None and lm.index in cap.mlp_in and lm.assign is not None:
                cn = self._neuron_contrib(lm, cap.mlp_in[lm.index], u)
                cc = torch.zeros(lm.n_clusters, device=cn.device).index_add_(0, lm.assign, cn)
                contrib[lm.offset + lm.n_heads:lm.offset + lm.n_units] = cc
                clusters = float(cc.sum())
            layer_sum.append([round(heads, 4), round(clusters, 4)])
        embed = float(cap.embed @ u) if cap.embed is not None else 0.0
        bias = self.s.lm_head.bias
        logit = float(cap.logits[tid])
        total = embed + float(contrib.sum()) + (float(bias[tid]) if bias is not None else 0.0)
        probs = torch.softmax(cap.logits, -1)
        top_p, top_i = probs.topk(5)
        c = contrib.cpu().numpy()
        order = np.argsort(-np.abs(c))[:TOP_UNITS]
        return {
            "run": run.id, "t": t, "pos": pos, "req": req, "token": self.s.token_str(cap.ids[-1]),
            "target": {"id": tid, "text": self.s.token_str(tid), "logit": round(logit, 4),
                       "prob": round(float(probs[tid]), 5),
                       "rank": int((cap.logits > cap.logits[tid]).sum()) + 1},
            "alternatives": [{"id": int(i), "text": self.s.token_str(int(i)), "prob": round(float(p), 5)}
                             for p, i in zip(top_p.tolist(), top_i.tolist())],
            "contrib": [round(float(x), 4) for x in c],
            "layers": layer_sum, "embed": round(embed, 4), "logit": round(logit, 4),
            "rest": round(logit - total, 4),
            "top": [[int(g), round(float(c[g]), 4)] for g in order],
            "context_tokens": len(cap.ids), "seconds": cap.seconds,
            **self._conditions(run, req),
        }

    def _conditions(self, run, req: int) -> dict:
        recorded = next((e["edits"] for e in run.meta.get("weight_edits", []) if e.get("req") == req), [])
        now = self.rec.edits.active()
        steered = any(e.get("req") == req for e in run.meta.get("interventions", []))
        return {"weights_match": [e["id"] for e in recorded] == [e["id"] for e in now],
                "weights_edited": bool(now), "steering_was_on": steered}

    # ---- deeper internals ---------------------------------------------------------------

    @torch.no_grad()
    def neurons(self, run_id: str, t: int, g: int, target=None) -> dict:
        run = self._run(run_id)
        lm, kind, i = self.an.locate(int(g))
        if kind != "cluster":
            raise ValueError("neurons are listed for MLP clusters; open a cluster unit")
        _, pos, req = self.context(run, t)
        cap = self.capture(run, t)
        tid = self._resolve_target(run, req, pos, cap, target)
        idx = (lm.assign == i).nonzero().flatten()
        a = cap.mlp_in[lm.index][idx]
        cn = self._neuron_contrib(lm, cap.mlp_in[lm.index], self._direction(cap, tid))[idx]
        wn = lm.wnorm[idx] if lm.wnorm is not None else lm.down.weight.float()[:, idx].norm(dim=0)
        rows = sorted(zip(idx.tolist(), a.tolist(), cn.tolist(), wn.tolist()), key=lambda r: -abs(r[2]))
        # Gated MLP activations are almost never exactly zero, so "active" counts the neurons that
        # carry 90% of the cluster's write (sum of |activation| x output-weight norm).
        write = torch.sort(a.float().abs() * wn.float(), descending=True).values.cumsum(0)
        carry = int((write < 0.9 * write[-1]).sum()) + 1 if len(write) and write[-1] > 0 else 0
        return {"unit": int(g), "layer": lm.index, "cluster": i, "token": self.s.token_str(cap.ids[-1]),
                "target": {"id": tid, "text": self.s.token_str(tid)},
                "contrib": round(float(cn.sum()), 4), "active": carry,
                "neurons": [{"n": n, "act": round(x, 4), "write": round(abs(x) * w, 4), "contrib": round(c, 4)}
                            for n, x, c, w in rows]}

    @torch.no_grad()
    def attention(self, run_id: str, t: int, g: int) -> dict:
        run = self._run(run_id)
        lm, kind, h = self.an.locate(int(g))
        if kind != "head":
            raise ValueError("attention patterns belong to heads; open a head unit")
        _, pos, req = self.context(run, t)
        cap = self.capture(run, t)
        base = {"unit": int(g), "layer": lm.index, "head": h, "layer_type": lm.type}
        if lm.index not in cap.attn:
            reason = ("Linear-attention heads keep a running state instead of attending to each earlier token, "
                      "so there is no attention pattern to show." if lm.type == "linear_attention"
                      else "This layer's attention was not captured.")
            return base | {"available": False, "reason": reason}
        w = cap.attn[lm.index][h].cpu().numpy()
        t_of = {r["pos"]: r["t"] for r in run.tape if r["req"] <= req}
        top = np.argsort(-w)[:12]
        return base | {"available": True, "pos": pos,
                       "weights": [round(float(x), 5) for x in w],
                       "tokens": [self.s.token_str(i) for i in cap.ids],
                       "t_of": [t_of.get(p, -1) for p in range(len(cap.ids))],
                       "top": [[int(p), round(float(w[p]), 4)] for p in top]}
