"""Recorder settings, read from BRAINSCAN_* environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str) -> str:
    return os.environ.get(f"BRAINSCAN_{name}", default)


@dataclass
class Settings:
    # "ollama:<tag>" loads a model from the local Ollama store (dequantized from GGUF);
    # anything else is a Hugging Face repo id or local directory.
    model: str = field(default_factory=lambda: _env("MODEL", "ollama:llama3.1:8b"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA", str(PROJECT_ROOT / "runs"))))
    disk_budget_gb: float = field(default_factory=lambda: float(_env("DISK_BUDGET_GB", "50")))
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("PORT", "8765")))
    max_context: int = field(default_factory=lambda: int(_env("MAX_CONTEXT", "16384")))
    # Logit lens over prompt and tool-result tokens, not only generated ones.
    prefill_lens: bool = field(default_factory=lambda: _env("PREFILL_LENS", "1") == "1")
    # Full residual stacks at decision points (pre-generation, first token, tool-call start, last token).
    decision_residuals: bool = field(default_factory=lambda: _env("DECISION_RESIDUALS", "1") == "1")
