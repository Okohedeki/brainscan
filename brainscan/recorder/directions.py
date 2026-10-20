"""A bank of named directions in the residual stream, read live during every run.

Each direction records exactly how it was built (texts, layer, method) so the UI can
label a readout by its construction instead of by a loose name. Projections are
z-scored against the tokens used to build the direction. A matched random direction
ships alongside as a control band.
"""

from __future__ import annotations

import json
import random
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

DEFAULTS = Path(__file__).with_name("default_directions.json")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass
class Direction:
    name: str
    layer: int
    unit: torch.Tensor  # float32, on the model device
    mean: float
    std: float
    meta: dict


class DirectionBank:
    def __init__(self, root: Path, subject, tap):
        self.root = Path(root) / subject.slug
        self.root.mkdir(parents=True, exist_ok=True)
        self.s, self.tap = subject, tap
        self.dirs: dict[str, Direction] = {}
        self._analyses: OrderedDict[str, dict] = OrderedDict()
        for meta_file in sorted(self.root.glob("*.json")):
            meta = json.loads(meta_file.read_text())
            unit = load_file(str(meta_file.with_suffix(".safetensors")))["unit"].to(subject.device)
            self.dirs[meta["name"]] = Direction(meta["name"], meta["layer"], unit, meta["mean"], meta["std"], meta)

    @torch.no_grad()
    def _token_acts(self, text: str, layer: int) -> torch.Tensor:
        ids = self.s.tokenizer(text, add_special_tokens=False).input_ids
        bos = self.s.tokenizer.bos_token_id
        if bos is not None:  # some converted tokenizers don't add BOS themselves
            ids = [bos] + ids
        if len(ids) < 2:
            raise ValueError(f"text {text!r} has no tokens after the first position")
        with self.tap.paused():
            self.s.model(input_ids=torch.tensor([ids], device=self.s.device), use_cache=False)
        return self.tap.resid[layer][0, 1:].float()  # drop the first (attention-sink) position

    def _prompt_ids(self, text: str, chat: bool) -> list[int]:
        tok = self.s.tokenizer
        if chat and tok.chat_template:
            # Thinking off where supported, so the last token is where the model starts answering.
            rendered = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                               tokenize=False, enable_thinking=False)
            return tok(rendered, add_special_tokens=False).input_ids
        ids = tok(text, add_special_tokens=False).input_ids
        return ([tok.bos_token_id] if tok.bos_token_id is not None else []) + ids

    @torch.no_grad()
    def layer_acts(self, text: str, position: str = "last", chat: bool = True) -> torch.Tensor:
        """Residual at every layer for one text, shape (layers, d_model), float32 on CPU.
        position 'last' reads the final token; 'mean' averages all but the first."""
        ids = self._prompt_ids(text, chat)
        if len(ids) < 2:
            raise ValueError(f"text {text!r} is too short")
        with self.tap.paused():
            self.s.model(input_ids=torch.tensor([ids], device=self.s.device), use_cache=False)
        rows = [r[0, -1] if position == "last" else r[0, 1:].mean(0) for r in self.tap.resid]
        return torch.stack(rows).float().cpu()

    def analyze(self, positive: list[str], negative: list[str], position: str = "last", chat: bool = True) -> dict:
        """Diff-of-means direction at every layer, scored on held-out examples.

        A quarter of each set (at least one example) is held out when both sets have four or
        more examples. Separation d is the gap between held-out class means divided by the
        within-class spread of the training projections (stable even with one held-out example
        per set); accuracy is a midpoint-threshold classifier on the held-out examples.
        """
        positive = [t for t in (x.strip() for x in positive) if t]
        negative = [t for t in (x.strip() for x in negative) if t]
        if len(positive) < 2 or len(negative) < 2:
            raise ValueError("each set needs at least two examples")
        if position not in ("last", "mean"):
            raise ValueError("position must be 'last' or 'mean'")
        pos = torch.stack([self.layer_acts(t, position, chat) for t in positive])  # (n, L, d)
        neg = torch.stack([self.layer_acts(t, position, chat) for t in negative])
        held_out = len(positive) >= 4 and len(negative) >= 4
        rng = random.Random(0)

        def split(n):
            idx = list(range(n))
            rng.shuffle(idx)
            k = max(1, n // 4) if held_out else 0
            return (idx[k:], idx[:k]) if held_out else (idx, idx)

        (p_tr, p_te), (n_tr, n_te) = split(len(positive)), split(len(negative))
        layers = []
        for layer in range(pos.shape[1]):
            diff = pos[p_tr, layer].mean(0) - neg[n_tr, layer].mean(0)
            unit = diff / diff.norm().clamp_min(1e-8)
            tp, tn = pos[p_tr, layer] @ unit, neg[n_tr, layer] @ unit
            spread = torch.sqrt((tp.var(unbiased=False) + tn.var(unbiased=False)) / 2)
            spread = spread.clamp_min(1e-3 * float((tp.mean() - tn.mean()).abs()) + 1e-6)
            pp, nn_ = pos[p_te, layer] @ unit, neg[n_te, layer] @ unit
            mid = (tp.mean() + tn.mean()) / 2
            acc = ((pp > mid).float().sum() + (nn_ <= mid).float().sum()) / (len(pp) + len(nn_))
            layers.append({"layer": layer, "d": round(float((pp.mean() - nn_.mean()) / spread), 3),
                           "acc": round(float(acc), 3)})
        aid = uuid.uuid4().hex[:10]
        self._analyses[aid] = {"pos": pos, "neg": neg, "position": position, "chat": chat,
                               "positive": positive, "negative": negative, "layers": layers, "held_out": held_out}
        while len(self._analyses) > 4:
            self._analyses.popitem(last=False)
        best = max(layers, key=lambda x: x["d"])["layer"]
        return {"id": aid, "layers": layers, "best": best, "held_out": held_out,
                "n_positive": len(positive), "n_negative": len(negative)}

    def save_analysis(self, aid: str, layer: int, name: str, description: str = "") -> Direction:
        if aid not in self._analyses:
            raise ValueError("analysis expired; run it again")
        if not NAME_RE.match(name):
            raise ValueError("name: letters, digits, '.', '_' or '-', up to 64 characters")
        a = self._analyses[aid]
        if not 0 <= layer < a["pos"].shape[1]:
            raise ValueError(f"layer must be in 0..{a['pos'].shape[1] - 1}")
        diff = a["pos"][:, layer].mean(0) - a["neg"][:, layer].mean(0)
        unit = (diff / diff.norm()).to(self.s.device)
        score = a["layers"][layer]
        where = f"the {'last token' if a['position'] == 'last' else 'mean over tokens'}"
        meta = {"kind": "contrast", "description": description,
                "construction": f"difference of means at {where}{' of chat-formatted prompts' if a['chat'] else ''}, "
                                f"layer {layer}, {len(a['positive'])} vs {len(a['negative'])} examples; "
                                f"{'held-out' if a['held_out'] else 'in-sample'} separation d={score['d']}, "
                                f"accuracy {score['acc']:.0%}; z-scored on the same examples",
                "positive": a["positive"], "negative": a["negative"], "position": a["position"],
                "chat": a["chat"], "separation": score}
        samples = torch.cat([a["pos"][:, layer], a["neg"][:, layer]]).to(self.s.device)
        return self._save(name, layer, unit, samples, meta)

    def delete(self, name: str) -> None:
        if name not in self.dirs:
            raise KeyError(name)
        for suffix in (".json", ".safetensors"):
            (self.root / f"{name}{suffix}").unlink(missing_ok=True)
        del self.dirs[name]

    def _save(self, name: str, layer: int, unit: torch.Tensor, tokens: torch.Tensor, meta: dict) -> Direction:
        proj = tokens @ unit
        if not torch.isfinite(proj).all():
            raise ValueError(f"direction {name!r} produced non-finite projections")
        meta = {**meta, "name": name, "layer": layer, "mean": float(proj.mean()),
                "std": float(proj.std().clamp_min(1e-6)), "model": self.s.spec, "created": time.time()}
        save_file({"unit": unit.cpu().contiguous()}, str(self.root / f"{name}.safetensors"))
        (self.root / f"{name}.json").write_text(json.dumps(meta, indent=2))
        d = Direction(name, layer, unit, meta["mean"], meta["std"], meta)
        self.dirs[name] = d
        return d

    def build_contrast(self, name: str, layer: int, positive: list[str], negative: list[str],
                       description: str) -> Direction:
        pos = [self._token_acts(t, layer) for t in positive]
        neg = [self._token_acts(t, layer) for t in negative]
        diff = torch.stack([a.mean(0) for a in pos]).mean(0) - torch.stack([a.mean(0) for a in neg]).mean(0)
        unit = diff / diff.norm()
        meta = {"kind": "contrast", "description": description,
                "construction": f"difference of means, layer {layer}, {len(positive)} vs {len(negative)} texts, "
                                "z-scored on the same texts",
                "positive": positive, "negative": negative}
        return self._save(name, layer, unit, torch.cat(pos + neg), meta)

    def build_random(self, name: str, layer: int, reference: list[str], seed: int, description: str) -> Direction:
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(self.s.d_model, generator=g).to(self.s.device)
        unit = v / v.norm()
        tokens = torch.cat([self._token_acts(t, layer) for t in reference])
        meta = {"kind": "random_control", "description": description, "seed": seed,
                "construction": f"random unit vector (seed {seed}), layer {layer}, z-scored on the same texts "
                                "as the contrast directions"}
        return self._save(name, layer, unit, tokens, meta)

    def ensure_defaults(self) -> None:
        spec = json.loads(DEFAULTS.read_text())
        reference: list[str] = []
        for d in spec["contrast"]:
            reference += d["positive"] + d["negative"]
        for d in spec["contrast"]:
            if d["name"] not in self.dirs:
                layer = round(d["layer_fraction"] * (self.s.n_layers - 1))
                self.build_contrast(d["name"], layer, d["positive"], d["negative"], d["description"])
        r = spec["random"]
        if r["name"] not in self.dirs:
            layer = round(r["layer_fraction"] * (self.s.n_layers - 1))
            self.build_random(r["name"], layer, reference, r["seed"], r["description"])

    def project(self, resid: torch.Tensor) -> dict[str, np.ndarray]:
        """z-scored projections for residuals of shape (layers, tokens, d)."""
        return {name: ((resid[d.layer].float() @ d.unit - d.mean) / d.std).cpu().numpy()
                for name, d in self.dirs.items()}

    def describe(self) -> list[dict]:
        return [{k: v for k, v in d.meta.items() if k not in ("positive", "negative")}
                | {"n_positive": len(d.meta.get("positive", [])), "n_negative": len(d.meta.get("negative", []))}
                for d in self.dirs.values()]
