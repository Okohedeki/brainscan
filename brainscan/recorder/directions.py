"""A bank of named directions in the residual stream, read live during every run.

Each direction records exactly how it was built (texts, layer, method) so the UI can
label a readout by its construction instead of by a loose name. Projections are
z-scored against the tokens used to build the direction. A matched random direction
ships alongside as a control band.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

DEFAULTS = Path(__file__).with_name("default_directions.json")


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
