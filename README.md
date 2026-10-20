# Brainscan

An open lab for watching the inside of an open-weights model while it works through real
multi-step agent tasks. The recorder sits behind an OpenAI-compatible endpoint, so any agent
harness can use it as a drop-in model. Every token's internals are recorded and lined up with
what the agent does: tool calls, tool results, errors, test outcomes, file edits, verdicts.
The scanner UI shows it live.

Readouts are functional measurements (logit-lens probabilities, projections onto contrast
vectors built from example texts). Each is labelled by how it was built. Nothing here measures
experience.

## What works today (milestone 1 slice)

- **Recorder** (`brainscan/recorder/`): Hugging Face model with forward hooks on every decoder
  layer, a token-by-token generation loop with KV-prefix reuse across agent turns, and
  OpenAI-compatible `/v1/chat/completions` (streaming and tool calls; Llama-JSON and Hermes
  `<tool_call>` formats).
- **Per-token record** for prompt, tool-result and generated tokens: probability and surprisal,
  entropy, top-5 alternatives, logit lens at every layer (probability of the actual token and
  each layer's top guess), residual norm per layer, and z-scored projections onto a direction bank.
  Full residual stacks are saved only at decision points (state before generation, first token,
  tool-call start, last token).
- **Event alignment**: events are parsed from the transcript itself (user turns, tool results,
  errors, pytest summaries, the model's own tool calls). Harnesses add what the transcript can't
  show (`run_meta`, `file_edit`, `verdict`) via `POST /brainscan/events`.
- **Intervention API**: add, ablate or clamp a direction at any layer, for prompt tokens,
  generated tokens or both. A random direction is available as a control. Requests made with
  steering on are marked in the run's metadata.
- **Direction bank**: two demo contrast vectors (`error_vs_success`, `frustrated_vs_calm`) plus a
  matched `random_control`, built at about two-thirds depth on first start. They are uncalibrated
  demos for the UI, not validated probes.
- **Storage**: `runs/<run>/` holds Parquet token tables, JSONL events and transcript entries, and
  safetensors decision residuals, under a disk budget.
- **Scanner UI** at `http://localhost:8765/`: transcript, a shared token timeline (event lane,
  logit-lens heatmap by layer, surprisal and entropy, projections against the random control and
  a |z| ≤ 2 band), a token inspector, and steering controls.
- **Mini agent harness** with three tasks: `fix_fizzbuzz`, `toml_lite`, and `impossible_clamp`
  (contradictory tests, for studying giving up and test-editing).
- **Reasoning models** (Qwen3.5): thinking tokens are recorded as their own role and shown as
  collapsible thinking in the transcript; `--thinking on|off` on the harness, or
  `chat_template_kwargs` per request.

## Workbench: edit the weights

The Workbench tab changes the model's parameters, the way "abliterated" models are made, and
measures what that does. Runtime steering (Scanner tab) edits activations during a run; the
workbench edits the weights themselves.

1. **Find a direction.** Paste two sets of examples. Brainscan reads the residual stream for each
   at every layer, takes the difference of the set means, and charts how well each layer's
   direction separates held-out examples (k-fold cross-validated, with the spread across folds).
   Pick a layer and save the direction to the bank. Directions are always measured on the
   original weights. With ten or so examples per set the best layer is a rough guide; use
   30 or more for a steady choice.
2. **Edit weights.** Stack edits, then apply them:
   - *Remove a direction*: orthogonalize the weights that write to the residual stream (token
     embeddings, attention and MLP outputs) or read from it (attention and MLP inputs,
     unembedding), at chosen layers and strength. Embeddings plus all attention and MLP outputs
     at strength 1 is the directional ablation of Arditi et al. (2024).
   - *Scale a component*: a layer's attention or MLP block, one attention head, or one MLP neuron
     (factor 0 is a lesion).
   Add a random-direction edit with the same settings as a control. Applying keeps a CPU copy of
   every touched weight, so restore is exact, and every recording made while edits are applied is
   tagged with them (`weight_edits` in the run metadata; "weights edited" in the header).
3. **Compare.** Runs prompts on the original and edited weights side by side (greedy, steering
   off), with a refusal-phrase heuristic and side-effect checks on neutral text (KL divergence and
   perplexity).
4. **Export.** Saves the edited weights as a Hugging Face checkpoint with `brainscan_edits.json`
   recording the base model and every edit.

To try the whole stack without a GPU, build a tiny random model and serve it on CPU:

```bash
wsl -e bash scripts/run.sh scripts/make_tiny_model.py
```

```bash
wsl -e env BRAINSCAN_MODEL=/home/$USER/.cache/brainscan/models/tiny-llama-test BRAINSCAN_DEVICE=cpu BRAINSCAN_DATA=/tmp/brainscan-smoke BRAINSCAN_PORT=8766 bash scripts/run.sh -m brainscan.server
```

```bash
wsl -e bash scripts/run.sh scripts/smoke_workbench.py
```

## Run it

Everything runs inside WSL2 (Linux-first CUDA stack). One-time setup:

```bash
wsl -e bash scripts/setup_wsl.sh
```

Start the recorder (first start with an Ollama model converts it once, about 3 minutes):

```bash
wsl -e bash scripts/run.sh -m brainscan.server
```

Open `http://localhost:8765/`, then drive it with the bundled agent or any harness pointed at
`http://localhost:8765/v1`:

```bash
wsl -e bash scripts/run.sh -m brainscan.harness.mini_agent --task fix_fizzbuzz
```

Hardware report and disk estimates:

```bash
wsl -e bash scripts/run.sh -m brainscan.hardware
```

### Settings (environment variables)

Machine-specific defaults go in `brainscan.env` (git-ignored; copy `brainscan.env.example`).
`scripts/run.sh` reads it, and variables already set in the environment win.

| Variable | Default | Meaning |
|---|---|---|
| `BRAINSCAN_MODEL` | `ollama:llama3.1:8b` | `ollama:<tag>` (dequantized from the local Ollama GGUF), a Hugging Face repo id, or a local checkpoint folder |
| `BRAINSCAN_MODEL_NAME` | from the path | Display name, e.g. `Qwen/Qwen3.5-9B` |
| `BRAINSCAN_DEVICE` | `cuda` | `cpu` for testing without a GPU |
| `BRAINSCAN_THINKING` | `default` | Reasoning models: `on`, `off`, or the chat template's default |
| `BRAINSCAN_DATA` | `./runs` | Where recordings go |
| `BRAINSCAN_DISK_BUDGET_GB` | `50` | Stop writing residuals beyond this |
| `BRAINSCAN_PORT` | `8765` | Server port |
| `BRAINSCAN_MAX_CONTEXT` | `16384` | Prompt plus completion limit |
| `BRAINSCAN_PREFILL_LENS` | `1` | Logit lens on prompt and tool-result tokens too |
| `BRAINSCAN_DECISION_RESIDUALS` | `1` | Save full residual stacks at decision points |

## Known limits

- **Ollama weights are 4-bit.** `ollama:` models carry quantization noise even after
  dequantization, and published SAEs and transcoders were trained on bf16 activations. The UI
  marks such runs as exploratory. Use original bf16 checkpoints (e.g. `Qwen/Qwen3-8B`) for any
  causal claim.
- **Plain hooks, not nnsight.** The recorder uses raw PyTorch forward hooks (the same mechanism
  nnsight wraps) because the streaming server needs a hand-written decode loop with KV reuse.
  nnsight remains the tool for offline analysis.
- **A full GPU slows down instead of failing on Windows/WSL.** The driver spills into system
  RAM, so a recorder that runs out of VRAM crawls rather than raising an error. The recorder
  checks free memory at load time, releases cached memory after workbench operations and
  when free memory runs low, and uses expandable allocator segments. Other GPU apps
  (image generators, local LLM servers) still need to be closed for 9B-class models.
- **Qwen3.5 decodes at about 13 tokens/s** on a 4090. The `causal_conv1d` kernel isn't installed,
  so its linear-attention layers use a slower PyTorch fallback; the recording itself adds about
  5 ms per token.
- **Weight edits on some architectures are approximate or unsupported.** Gemma 2/3 pass block
  outputs through an extra norm before the residual add, so removing a direction from their
  output weights doesn't fully remove it from the stream. Mixture-of-experts MLPs (GPT-OSS) are
  not edited yet. Head scaling needs standard attention (`o_proj`); Qwen3.5's linear-attention
  layers have none.
- **Not built yet:** SAE features, attention summaries, MoE routing capture, and nnsight-based
  analysis modules. One request runs at a time.

## Layout

```
brainscan/recorder/   model loading, hooks, chat rendering, events, storage, session, directions,
                      weight edits (edits.py) and workbench checks (workbench.py)
brainscan/server.py   FastAPI app: /v1 endpoint, /brainscan API, live WebSocket, UI
brainscan/harness/    mini coding agent and tasks
brainscan/hardware.py Phase 0 hardware report
ui/                   scanner (plain HTML, CSS and canvas JavaScript; no build step)
reports/              research reports
tests/                unit tests (pytest)
```
