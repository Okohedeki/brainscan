# Brainscan

A tracing and tuning platform for open-weights models. The recorder sits behind an
OpenAI-compatible endpoint, so any agent harness can use it as a drop-in model. Every token's
internals are recorded and lined up with what the agent does: tool calls, tool results, errors,
test outcomes, file edits, verdicts. You can then trace any token back through the model's
attention heads and neurons, and change the weights and measure what changed.

The UI opens on **Actions**: what you want to do to the model. Each action is tried and measured
before anything is kept, and the traces are evidence you open when a result needs explaining.
The first is **Make it forget a fact** (see [Actions](#actions-forget-a-fact)). The cards for
other jobs open the tool that does them today.

To look around instead, three tabs show the model at work:

1. **Watch**: the run live: transcript, a per-token timeline (logit lens, surprisal,
   projections onto directions), a token inspector, and runtime steering. Ask the model directly
   from the composer under the transcript; no harness needed. The model name in the top bar
   switches to any other open-weights model on the machine.
2. **Trace**: the conversation next to a graph of the model's attention heads and MLP neuron
   clusters. Click any word to see why the model wrote it: where its logit came from layer by
   layer, when the model first had it as its top guess, the units that pushed it up or down,
   and knockout tests that remove units and re-run. Find paths through the graph, and open any
   unit down to its neurons and attention pattern at a token.
3. **Tune**: find directions from examples, edit the weights (remove a direction, lesion a head
   or neuron), compare original and edited side by side, and export the variant.

Press `?` anywhere in the UI for a guide to every tool, with a "Show me" link to each.

Readouts are functional measurements (logit-lens probabilities, projections onto contrast
vectors built from example texts). Each is labelled by how it was built. Nothing here measures
experience.

## What works today (milestone 1 slice)

- **Recorder** (`brainscan/recorder/`): Hugging Face model with forward hooks on every decoder
  layer, a token-by-token generation loop with KV-prefix reuse across agent turns, and
  OpenAI-compatible `/v1/chat/completions` (streaming and tool calls; Llama-JSON and Hermes
  `<tool_call>` formats).
- **Stopping a generation**: `POST /brainscan/cancel` stops the request that is generating;
  with `{"run": "<id>"}` it stops only a request of that run, including one still waiting for the
  model. The request stops between two tokens (or between prefill chunks of a long prompt) and
  finishes like any other: the tokens generated so far are recorded (tape, transcript, unit
  activity), a `cancelled` event marks where it stopped, and its client gets them back with
  `finish_reason: "cancelled"`. OpenAI's API has no such value; most SDKs pass it through. A
  stopped reply is returned as text, never as tool calls a harness might run. A client that
  disconnects mid-request (a closed tab, a killed harness) stops its request the same way. A
  request stopped before it reached the model records nothing and returns `cancelled` with
  `"brainscan": {"run": null}`.
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
- **Watch tab** at `http://localhost:8765/`: transcript, a shared token timeline (event lane,
  logit-lens heatmap by layer, surprisal and entropy, projections against the random control and
  a |z| ≤ 2 band), a token inspector, and steering controls.
- **Mini agent harness** with three tasks: `fix_fizzbuzz`, `toml_lite`, and `impossible_clamp`
  (contradictory tests, for studying giving up and test-editing).
- **Reasoning models** (Qwen3.5): thinking tokens are recorded as their own role and shown as
  collapsible thinking in the transcript; `--thinking on|off` on the harness, or
  `chat_template_kwargs` per request.

## Actions: forget a fact

A fact is a lead-in and the answer the model gives after it ("The capital of Australia is" →
"Canberra"), typed, or picked from a chat: click a word in Trace and press **Forget this fact**.
The flow has five steps (`brainscan/recorder/actions.py`, `POST /brainscan/actions/forget/{step}`):

1. **Goal**: the fact, other lead-ins and questions to test it with, and facts it should keep
   (`lead-in → answer`, one per line). Saved behavior checks are measured too.
2. **Target**: units are ranked by their direct effect on the answer at the lead-in's last token,
   and the top 1, 2, 4 … 32 are knocked out together. The smallest set that knocks the answer off
   first place is suggested; the slider picks any number.
3. **Try**: greedy answers with and without the units, for every lead-in and question. Nothing
   in the weights changes.
4. **Measure**: the answer's probability before and after on every lead-in; the same number of
   random heads and clusters removed instead, three times (if random units remove the fact too,
   it wasn't found in these units); the facts to keep and saved checks before and after; and drift
   on neutral reference text (KL and perplexity). A verdict sums it up: works, partly, kept (still
   answers) or unspecific.
5. **Keep**: the units' output weights are zeroed as one edit in the stack and applied now, and
   the facts to keep (and the forgotten fact) can be saved as behavior checks. Knocking out a unit
   at run time is the same as zeroing its output weights, so what was measured is what is kept.
   Restore the original weights, or export the edited model, from **Your model** on the Actions
   home and from Tune.

Everything is measured on the weights as they are, including any edits already applied.

## Tune: edit the weights

The Tune tab changes the model's parameters, the way "abliterated" models are made, and
measures what that does. Runtime steering (Watch tab) edits activations during a run; Tune edits
the weights themselves.

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
3. **Compare.** *Re-run this chat* asks the open conversation's last question again, greedy, on
   the original weights (edits lifted, not recorded) and the edited weights (recorded as
   `<run>-edited`), shows both answers and the first token where they part, and *Trace the
   difference* opens that word in Trace compared with the original weights. Your own prompts can
   still be run side by side, with a refusal-phrase heuristic and side-effect checks on neutral
   text (KL divergence and perplexity).
4. **Check behavior.** Saved checks of what the model should say next: from any traced word
   (*Add as a check* in its panel; the exact context of that chat) or typed ("The capital of
   France is" → "Paris", as raw text or a chat question). With edits applied, each shows its
   probability on the original weights → now, with rank and the new top guess, so an edit's
   intended effect and its side effects show up together. Checks are saved per model.
5. **Export.** Saves the edited weights as a Hugging Face checkpoint with `brainscan_edits.json`
   recording the base model and every edit.

Tune is also reachable from Trace. *Make permanent* on a knockout result adds one edit that
removes all its units (heads and whole neuron clusters, kind `units`) and applies it, and the
**Edits** panel (`E`) in the Trace toolbar holds the stack, *Re-run this chat* and the checks, so
the loop find → test → edit → verify stays on the graph. On Qwen3.5-9B, making the five-unit
" Canberra" knockout permanent turns the model's own answer to "What is the capital of
Australia?" into "Sydney"; the checks show it also weakens " Paris" after "The capital of France
is" (0.21 → 0.08), a side effect the single-token view would miss.

## Trace: the model as a live graph

The Trace tab draws the model as a node graph on a zoomable canvas. Layers run left to right
along the residual stream; each layer's attention heads sit above it and its MLP neurons, grouped
into clusters by the direction of their output weights (64 per layer by default), sit below.

- **Activity.** Every weight takes part in every token, so a node shows how much its unit *wrote*
  into the residual stream at the current token: the exact norm of each head's write, and each
  cluster's write mass (the summed size of its neurons' writes). Nodes are colored against their
  own average over the run on a log scale; a unit counts as hit at 3σ above it. Units hit in the
  last few tokens keep a fading ring.
- **Edges.** Curved edges join units whose weights compose: a later unit's input weights (after
  the norm in front of its block) read an earlier unit's top output directions, scored against a
  random direction (1× = chance; Elhage et al., 2021). They are computed once per model from the
  weights and cached. "Circuit" shows the strongest edges between units active at this token;
  selecting a node shows its strongest inputs and outputs. They describe the weights, not a
  measured causal effect.
- **Zoom.** Scroll to zoom and drag to pan: dots become labelled nodes, then cards with a
  thumbnail of each unit's output weights. A minimap, a jump-to palette (Ctrl/Cmd+K: "L12 c7",
  "L3 h5", a token, or a parameter name) and keyboard shortcuts (F fit, Space replay, arrows step
  tokens) help move around.
- **A unit.** Clicking a node opens a drawer: its output and input weights as a heatmap you can
  zoom down to single weights, the value histogram and singular values, the tokens its main write
  direction promotes and suppresses through the unembedding, its strongest connections, its most
  active tokens in the run, its neurons, and a button that adds a lesion of it to the edit stack
  in Tune. The token embedding and unembedding nodes open their matrices the same way.
- **Timeline.** The dock follows a live run (units hit per token, or the selected unit's activity)
  and can replay a recorded one token by token.

### Tracing a token

Tracing re-runs the run's context up to a token on the current weights (a couple of seconds for a
few hundred tokens on Qwen3.5-9B; results are cached) and reads every unit exactly.

- **Ask and watch.** The composer (under the Watch transcript, and at the bottom of the Trace
  tab's conversation) sends a chat request to the recorder's own `/v1` endpoint, so it is
  recorded like any agent run: its tokens stream into the transcript, the timeline and the graph
  as they are generated. A chat continues in the same run until **New chat**; options are
  thinking on/off, max tokens and temperature (0 = greedy).
- **Stop.** While the composer's request runs, its Send button is **■ Stop** (or press Esc in the
  box): the reply ends at the next token and the composer reports how many tokens were kept. The
  Trace dock shows **■ Stop** whenever any request is generating, an agent harness's included.
  Either way the stopped request is recorded up to where it stopped (see *Stopping a generation*
  above), and the next message continues the run.
- **The conversation in Trace.** The run's text, word by word, grouped into turns, with words the
  model found surprising underlined (by −log₂ p). Clicking a word traces the position before it
  with that word as the target, so you see why the model wrote it; the readout and the graph
  follow. `T` shows or hides the text.
- **Why this token** (`W`, or **Why?** in the dock). The target's probability, rank and logit
  and the runners-up; *where the logit came from*: each layer's attention heads and MLP clusters
  by direct effect, with the running total from the embedding to the final logit; *when the
  model knew*: the logit lens for that token at every layer and the first layer where it is the
  top guess; the units that pushed it up or pulled it down (click one to open it).
- **Knockout tests.** Remove units and re-run the context with their writes zeroed at every
  position: the token's probability before → after, its rank and the new top guesses, with the
  units' total effect next to their direct effect. Run it on the top five units, your selection,
  the units on the shown paths, or one unit from its drawer. On Qwen3.5-9B, knocking out the five
  units that pushed " Canberra" up for "What is the capital of Australia?" turns the answer to
  " Sydney" (p 0.999 → 0.26; Sydney 0.71); their direct effect was +12.4 logits and the rest of
  the network made up 40% of it. A knockout is the same as lesioning those units' output weights,
  without touching the weights.

- **Attribution** (`A`). Colors each unit by its direct effect on the next token's logit: its
  write at this token, read through the final norm (with its scale held at the recorded value)
  and the target's unembedding row. The units plus the embedding add up to the logit. The dock
  shows the target, its probability and the runner-up tokens; click one to explain it instead.
  This is the direct effect only; effects a unit has through later layers are not counted.
- **Selection.** Shift-click adds a unit, shift-drag selects a box (alt-drag removes), Esc clears.
- **Paths** (`P`). A beam search over the composition edges: the strongest routes out of the
  selection, into it, between selected units, or the circuit behind the next token (the units
  with the largest positive attribution and the active units feeding them). Paths can be weighted
  by activity at the current token. Select all or none, frame them, add their units to the
  selection, or copy them as JSON.
- **At this token.** A unit's drawer shows its attribution and, for a full-attention head, where
  it attended from this token (click a token to jump there); for an MLP cluster, every neuron's
  activation and effect on the next token, and a grid of its neurons inside the graph node.
  Linear-attention heads keep a running state rather than attending to earlier tokens, so they
  have no attention pattern.
- **Before and after an edit** (`vs original`). With weight edits applied in Tune, the dock offers
  a comparison: the same token is traced again with the edits lifted (the forward pass and the
  weights the attribution is read through), the graph colors each unit by edited minus original,
  the dock shows the target's probability before → after, a unit's drawer shows both values, and
  the path explorer's next-token mode traces the circuit behind the change. The model is held
  for the whole read, so no request runs on the unedited weights, and the edits go back exactly.
  The same `weights=original` option works on the `explain`, `neurons` and `attention` endpoints.

Activity costs one byte per unit per token (2,944 units, about 3 KB per token, for Qwen3.5-9B) and
is stored under `runs/<run>/anatomy/`. The first start on a model clusters the neurons and scores
the edges once, then caches both.

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

### Demo videos

`scripts/demo/` records product videos of the running UI: a headless Chrome is driven over the
DevTools protocol through scripted scenes (an on-screen cursor, captions and title cards; captions
quote the numbers the app shows), and ffmpeg joins the frames into 1080p MP4s. It runs on the
Windows side with Node 22+, against a recorder serving Qwen3.5-9B. The scenes change the
recorder's state (a chat, a weight edit, checks, steering), so record on a scratch setup:

```bash
node scripts/demo/recorder.mjs C:/Temp/brainscan-demos
```

Name scenes after the folder to record only those (`01-overview` … `09-directions-and-steering`).
`CHROME`, `FFMPEG` and `BRAINSCAN_URL` override the defaults.

### Switching models

Click the model name in the top bar to see every open-weights model already on this machine and
load another one without restarting. The list comes from Brainscan's model cache
(`~/.cache/brainscan/models`), the Hugging Face hub cache, any folders in `BRAINSCAN_MODEL_DIRS`,
and the local Ollama store. Each entry shows its size, layer count and the GPU memory the loader
needs against what would be free once the current model is unloaded; models that won't fit, that
transformers can't load, or that have no tokenizer are marked and can't be picked.

- Loading unloads the current model first. Steering and applied weight edits are dropped (files on
  disk never change), and `/v1` returns 503 until the new model is ready. If the new model fails to
  load, the previous one is loaded again and the picker shows why.
- An Ollama tag is dequantized to bf16 once on first load (several minutes; it writes the
  checkpoint under `~/.cache/brainscan/models`) and is marked exploratory.
- The first load of a model also builds its unit map and edges for the Trace tab; both are cached
  per model, like its direction bank. Runs keep the model they were recorded with.
- Brainscan never downloads a model. To add one from Hugging Face, download it yourself, e.g.
  `hf download Org/Name --local-dir ~/.cache/brainscan/models/Org--Name` inside WSL, then reopen
  the list. `GET /brainscan/models` and `POST /brainscan/model {"spec": ...}` do the same over HTTP.
- A restart loads `BRAINSCAN_MODEL` again.

### Settings (environment variables)

Machine-specific defaults go in `brainscan.env` (git-ignored; copy `brainscan.env.example`).
`scripts/run.sh` reads it, and variables already set in the environment win.

| Variable | Default | Meaning |
|---|---|---|
| `BRAINSCAN_MODEL` | `ollama:llama3.1:8b` | `ollama:<tag>` (dequantized from the local Ollama GGUF), a Hugging Face repo id, or a local checkpoint folder |
| `BRAINSCAN_MODEL_NAME` | from the path | Display name for `BRAINSCAN_MODEL`, e.g. `Qwen/Qwen3.5-9B` (an `Org--Name` folder is already shown as `Org/Name`) |
| `BRAINSCAN_MODEL_DIRS` | none | Extra folders the model picker searches for checkpoints, separated by `:` (e.g. `/mnt/h/models`) |
| `BRAINSCAN_DEVICE` | `cuda` | `cpu` for testing without a GPU |
| `BRAINSCAN_THINKING` | `default` | Reasoning models: `on`, `off`, or the chat template's default |
| `BRAINSCAN_DATA` | `./runs` | Where recordings go |
| `BRAINSCAN_DISK_BUDGET_GB` | `50` | Stop writing residuals beyond this |
| `BRAINSCAN_PORT` | `8765` | Server port |
| `BRAINSCAN_MAX_CONTEXT` | `16384` | Prompt plus completion limit |
| `BRAINSCAN_PREFILL_LENS` | `1` | Logit lens on prompt and tool-result tokens too |
| `BRAINSCAN_DECISION_RESIDUALS` | `1` | Save full residual stacks at decision points |
| `BRAINSCAN_ANATOMY` | `1` | Record per-token unit activity for the Trace tab; `0` turns it off |
| `BRAINSCAN_ANATOMY_CLUSTERS` | `64` | MLP neuron clusters per layer in the Trace tab |
| `BRAINSCAN_GRAPH_DECODE` | `1` | Replay decode steps from CUDA graphs (same outputs, about 3x faster); `0` decodes eagerly |

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
- **One token at a time.** Qwen3.5-9B records at about 40 tokens/s on a 4090: each decode step
  replays the forward from CUDA graphs (`recorder/graphs.py`, about 21 ms per token, bit-identical
  to eager decoding) and the readout (logit lens, norms, projections) adds about 3.5 ms, most of it
  the logit lens reading the 248k-row unembedding. The readout is queued behind the forward and
  copied back with one wait per token, so its CPU side (kernel launches, detokenizing, the token
  record) runs while the GPU works. Eager decoding
  (`BRAINSCAN_GRAPH_DECODE=0`) manages about 12 tokens/s: its forward takes about 78 ms, most of it
  CPU time spent launching some 2,000 small kernels, while the GPU work itself takes about 22 ms.
- **Weight edits on some architectures are approximate or unsupported.** Gemma 2/3 pass block
  outputs through an extra norm before the residual add, so removing a direction from their
  output weights doesn't fully remove it from the stream. Mixture-of-experts MLPs (GPT-OSS) are
  not edited yet. Head scaling needs standard attention (`o_proj`); Qwen3.5's linear-attention
  layers have none.
- **Recorded unit activity is coarse.** During a run only each cluster's write mass is recorded,
  an upper bound on what the cluster writes together; tracing recomputes exact per-neuron values
  for one token on demand. Unit labels read write directions straight through the unembedding,
  which is noisy in early layers. Edges use each unit's top four output directions and the
  original weights. MoE layers have no clusters yet.
- **Attribution is direct effect only**, on the current weights unless you compare with the
  original (a warning shows when they differ from the recording), and paths follow weight-based
  edges, not measured causal effects. A before/after difference is also direct effect: a change
  at a later unit reflects its changed inputs, not why they changed. Tracing waits while the
  model is generating, and comparing with large applied edits takes a few seconds each way
  while the original weights are swapped in and back.
- **Not built yet:** SAE features, MoE routing capture, and nnsight-based analysis modules. One
  request runs at a time.

## Layout

```
brainscan/recorder/   model loading, hooks, chat rendering, events, storage, session, directions,
                      weight edits (edits.py), workbench checks (workbench.py), models on disk (catalog.py), the weight
                      graph's units, activity probe and edges (anatomy.py), and token tracing:
                      attribution, neurons and attention at a token (tracing.py), and actions:
                      forget a fact (actions.py)
brainscan/server.py   FastAPI app: /v1 endpoint, /brainscan API, live WebSocket, UI
brainscan/harness/    mini coding agent and tasks
brainscan/hardware.py Phase 0 hardware report
ui/                   Actions (actions.js, the home), Watch (app.js, models.js, chat.js), Trace
                      (graph.js, weights.js, paths.js), Tune (workbench.js), the edit stack and
                      checks (edits.js) and the guide (guide.js): plain HTML, CSS and canvas
                      JavaScript; no build step
scripts/demo/         product-video recorder (headless Chrome + ffmpeg) and its scenes
reports/              research reports
tests/                unit tests (pytest)
```
