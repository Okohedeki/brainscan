"""Load the subject model and expose the parts the recorder hooks into.

`ollama:<tag>` loads weights straight from the local Ollama store: the GGUF blob is
dequantized once into a bf16 Hugging Face checkpoint under ~/.cache/brainscan. The
quantization noise stays in the weights, so those recordings are exploratory and
labelled as such; causal experiments should use original bf16 checkpoints.
"""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import save_model
from transformers import AutoModelForCausalLM, AutoTokenizer

CACHE = Path(os.environ.get("BRAINSCAN_CACHE", Path.home() / ".cache" / "brainscan"))
STOP_TOKENS = ["<|eot_id|>", "<|eom_id|>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>",
               "<end_of_turn>", "<|return|>", "<|call|>"]


def ollama_roots() -> list[Path]:
    roots = [Path(os.environ["OLLAMA_MODELS"])] if os.environ.get("OLLAMA_MODELS") else []
    roots += [Path(p) for p in glob.glob("/mnt/c/Users/*/.ollama/models")]
    roots.append(Path.home() / ".ollama" / "models")
    return [r for r in roots if r.exists()]


def resolve_ollama(tag: str) -> tuple[Path, dict]:
    """Return (GGUF blob path, Ollama config) for a tag like 'llama3.1:8b'."""
    name, _, version = tag.partition(":")
    name = name if "/" in name else f"library/{name}"
    for root in ollama_roots():
        manifest = root / "manifests" / "registry.ollama.ai" / name / (version or "latest")
        if not manifest.exists():
            continue
        data = json.loads(manifest.read_text())
        blob = next((root / "blobs" / layer["digest"].replace(":", "-") for layer in data.get("layers", [])
                     if layer["mediaType"] == "application/vnd.ollama.image.model"), None)
        config = {}
        if data.get("config", {}).get("digest"):
            config_path = root / "blobs" / data["config"]["digest"].replace(":", "-")
            if config_path.exists():
                config = json.loads(config_path.read_text())
        if blob:
            return blob, config
    raise FileNotFoundError(f"Ollama model {tag!r} not found under {ollama_roots()}")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")


def display_name(spec: str) -> str:
    """'org/Name' for Hugging Face ids and cache snapshots, the tag for Ollama models."""
    if spec.startswith("ollama:"):
        return spec.split(":", 1)[1]
    cached = re.search(r"models--([^/\\]+)--([^/\\]+)", spec)
    if cached:
        return f"{cached.group(1)}/{cached.group(2)}"
    return spec.rstrip("/").split("/")[-1] if Path(spec).exists() else spec


def prepare(spec: str, dtype=torch.bfloat16) -> tuple[str, dict]:
    """Make sure the model exists as a Hugging Face checkpoint; return (path, provenance)."""
    if not spec.startswith("ollama:"):
        source = "local" if Path(spec).exists() else "huggingface"
        return spec, {"spec": spec, "source": source, "quantization": None, "exploratory": False}
    tag = spec.split(":", 1)[1]
    local = CACHE / "models" / f"ollama-{_slug(tag)}"
    prov_file = local / "brainscan_provenance.json"
    if prov_file.exists():
        return str(local), json.loads(prov_file.read_text())
    blob, config = resolve_ollama(tag)
    provenance = {
        "spec": spec, "source": "ollama", "blob": str(blob),
        "quantization": config.get("file_type"), "family": config.get("model_family"),
        "exploratory": True,
        "note": "Weights dequantized from an Ollama GGUF; quantization noise remains. "
                "Use original bf16 weights for causal claims.",
    }
    gguf_dir = CACHE / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    link = gguf_dir / f"{_slug(tag)}.gguf"
    if not link.exists():
        link.symlink_to(blob)
    print(f"[brainscan] dequantizing {blob} -> {local} (one time)", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(gguf_dir, gguf_file=link.name)
    model = AutoModelForCausalLM.from_pretrained(gguf_dir, gguf_file=link.name, dtype=dtype)
    # The weights are already dequantized, but transformers still tags the model as
    # GGUF-quantized and refuses save_pretrained, so write config and tensors directly.
    local.mkdir(parents=True, exist_ok=True)
    if hasattr(model.config, "quantization_config"):
        delattr(model.config, "quantization_config")
    model.config.save_pretrained(local)
    model.generation_config.save_pretrained(local)
    save_model(model, str(local / "model.safetensors"))
    tokenizer.save_pretrained(local)
    prov_file.write_text(json.dumps(provenance, indent=2))
    del model
    return str(local), provenance


def _first_attr(obj, paths: list[str]):
    for path in paths:
        cur = obj
        try:
            for part in path.split("."):
                cur = getattr(cur, part)
            return cur
        except AttributeError:
            continue
    raise AttributeError(f"none of {paths} found on {type(obj).__name__}")


def _check_vram(path: str, device: str, headroom_gb: float = 2.5) -> None:
    """Refuse to load when the weights won't fit. On Windows the driver would otherwise spill
    into system memory and every forward pass would crawl."""
    weights = sum(f.stat().st_size for f in Path(path).glob("*.safetensors")) if Path(path).is_dir() else 0
    if not weights or not device.startswith("cuda") or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    need = weights + headroom_gb * 1024 ** 3
    if free < need:
        raise RuntimeError(
            f"not enough free GPU memory: {free / 1024 ** 3:.1f} GB free, about {need / 1024 ** 3:.1f} GB needed. "
            "Close other GPU apps (image generators, local LLM servers) or choose a smaller model.")


class Subject:
    """The model under study plus the handles the recorder needs."""

    def __init__(self, spec: str, dtype=torch.bfloat16, device: str = "cuda"):
        path, self.provenance = prepare(spec, dtype)
        _check_vram(path, device)
        self.spec = spec
        self.name = os.environ.get("BRAINSCAN_MODEL_NAME") or display_name(spec)
        self.slug = _slug(self.name.split("/")[-1])
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, dtype=dtype, device_map=device, attn_implementation="sdpa")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.layers = _first_attr(self.model, ["model.layers", "model.language_model.layers",
                                               "language_model.model.layers", "transformer.h"])
        self.norm = _first_attr(self.model, ["model.norm", "model.language_model.norm",
                                             "language_model.model.norm", "transformer.ln_f"])
        self.lm_head = self.model.get_output_embeddings()
        self.n_layers = len(self.layers)
        config = self.model.config.get_text_config()
        self.d_model = config.hidden_size
        self.vocab_size = self.lm_head.weight.shape[0]
        self.dtype = str(dtype).replace("torch.", "")
        # Linear-attention / SSM layers keep recurrent state that can't be cropped back to a
        # shared prefix, so those models re-read the prompt on every request.
        self.layer_types = list(getattr(config, "layer_types", None) or [])
        self.croppable_cache = all(t in ("full_attention", "sliding_attention") for t in self.layer_types)
        # Rows per logit-lens chunk, sized so one fp32 (rows x vocab) buffer stays near 400 MB,
        # and a prefill chunk small enough that its logits fit beside the weights.
        self.lens_chunk = max(64, int(1e8 // self.vocab_size))
        self.prefill_chunk = 512 if self.vocab_size < 200_000 else 256
        self.stop_ids = self._stop_ids()
        # Token placed before raw texts (direction building, reference text): BOS where the model
        # has one, otherwise its document separator. It absorbs the first-position attention sink.
        vocab = self.tokenizer.get_vocab()
        self.sink_id = (self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None
                        else vocab.get("<|endoftext|>", self.tokenizer.eos_token_id))
        n_tok = len(self.tokenizer)
        strings = self.tokenizer.batch_decode([[i] for i in range(n_tok)])
        self._token_strs = strings + [f"<{i}>" for i in range(n_tok, self.vocab_size)]

    def _stop_ids(self) -> set[int]:
        ids: set[int] = set()
        eos = self.model.generation_config.eos_token_id
        ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
        if self.tokenizer.eos_token_id is not None:
            ids.add(self.tokenizer.eos_token_id)
        vocab = self.tokenizer.get_vocab()
        ids.update(vocab[t] for t in STOP_TOKENS if t in vocab)
        return ids

    def token_str(self, token_id: int) -> str:
        return self._token_strs[token_id] if 0 <= token_id < len(self._token_strs) else f"<{token_id}>"

    def info(self) -> dict:
        return {"name": self.name, "spec": self.spec, "n_layers": self.n_layers, "d_model": self.d_model,
                "vocab_size": self.vocab_size, "dtype": self.dtype, "provenance": self.provenance,
                "architecture": type(self.model).__name__, "layer_types": self.layer_types,
                "croppable_cache": self.croppable_cache}


if __name__ == "__main__":
    path, prov = prepare(sys.argv[1] if len(sys.argv) > 1 else "ollama:llama3.1:8b")
    print(path)
    print(json.dumps(prov, indent=2))
    tok = AutoTokenizer.from_pretrained(path)
    print("chat template present:", bool(tok.chat_template))
