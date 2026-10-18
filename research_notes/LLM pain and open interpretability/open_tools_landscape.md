# Open-weights interpretability tools landscape (as of 2026-10-15), for positioning "Brainscan"

Scope: tools that let people run open-weights LLMs and look at their internals. For each: what it does, models covered, access model, license, maturity. Then agentic/multi-turn recording, single-RTX-4090 practicality, and gaps. Release dates on GitHub release pages often show only day and month. Where I inferred the year, the note says so.

## Q1. What each tool offers: models, access, license, maturity (plus verdicts on the quoted claims)

### Takeaway
The open stack is now quite capable. **Neuronpedia** (open source, hosted, no-code) is the main no-GPU front end. **circuit-tracer** (now under `decoderesearch`, latest **v0.5.3, 14 Oct 2026**) produces attribution graphs. **nnsight/NDIF** gives free remote access to Llama-3.1-8B/70B/405B and gpt-oss-120b. **TransformerLens 4.x** (v4.2.0) is built on the HF-native "TransformerBridge", which covers Qwen3.5/GatedDeltaNet, Gemma 3 and GPT-OSS. **SAELens** (v6.54.5) loads Gemma Scope 2. Two big SAE suites cover current models: **Gemma Scope 2** (all Gemma 3 sizes, CC-BY-4.0) and **Qwen-Scope** (Qwen3/Qwen3.5, "qwen" license). Goodfire's hosted Ember API was **deprecated in Feb 2026**, but its open SAEs remain on Hugging Face.

**Quoted claims:**
- **circuit-tracer claim: model list correct, version stale.** The README lists exactly those model families. v0.5.2 (18 Jul 2026) only changed versioning, and v0.5.3 shipped 14 Oct 2026.
- **Qwen-Scope claim: model list and "every layer" correct, the rest partly wrong.**
  - "April 2026" is unsupported. All dated primary sources are from May 2026. The April 2026 OpenMOSS release on Neuronpedia is a different project (Llama Scope 2 on Qwen3-1.7B).
  - "Base" is wrong for Qwen3.5-27B, whose SAEs were trained on the post-trained model.
  - The SAEs cover the residual stream only.

### Cited Findings

#### circuit-tracer (Anthropic-originated attribution graphs)
- **Origin:** Anthropic open-sourced the attribution-graph library and a Neuronpedia-hosted frontend on 29 May 2025 — [Anthropic](https://www.anthropic.com/research/open-source-circuit-tracing).
- **Repo location:** github.com/safety-research/circuit-tracer now resolves to **decoderesearch/circuit-tracer** (2.9k stars, MIT). The README's tutorial and demo links still point to safety-research — [GitHub decoderesearch/circuit-tracer](https://github.com/decoderesearch/circuit-tracer).
- **Transcoders listed in the README:**
  - Gemma-2 2B: PLTs `mntss/gemma-scope-transcoders`; CLTs `mntss/clt-gemma-2-2b-426k` and `-2.5M`.
  - Llama-3.2 1B: PLT `mntss/transcoder-Llama-3.2-1B`; CLT `mntss/clt-llama-3.2-1b-524k`.
  - Qwen-3 PLTs for 0.6B, 1.7B, 4B, 8B and 14B (`mwhanna/qwen3-*-transcoders[-lowl0]`).
  - GPT-OSS 20B CLT: `mntss/clt-131k`.
  - Gemma-3 PLTs for 270M, 1B, 4B, 12B and 27B, PT and IT (collection `mwhanna/gemma-scope-2-transcoders-circuit-tracer`). These "require using the `nnsight` backend".
  - Llama 3.1 8B **Instruct** TopK PLTs: `facebook/crv-8b-instruct-transcoders`.
  - Source: [README](https://github.com/decoderesearch/circuit-tracer).
- **Backends:**
  - TransformerLens is the default and does not cover all HF models.
  - nnsight (`backend='nnsight'`) "supports most HuggingFace models" but is experimental, slower and less memory-efficient.
  - Offload options are cpu, disk or None. The default batch size is 256. The Llama demo is not supported on Colab (15 GB) — [README](https://github.com/decoderesearch/circuit-tracer).
- **Release history** (years inferred from order) — [Releases](https://github.com/safety-research/circuit-tracer/releases):
  - v0.2.0 (5 Aug 2025): CLTs for Gemma-2 2B and Llama-3.2 1B, plus Qwen3 transcoders.
  - v0.3.0 (8 Jan 2026): nnsight backend and GemmaScope 2 / Gemma-3 support.
  - v0.4.0 (23 Feb 2026): attribution targets (arbitrary tokens or vectors). Gemma-3-IT now requires a chat-format start.
  - v0.5.0 (18 Apr 2026): top-k transcoders, including Llama-3 8B Instruct.
  - v0.5.1 (12 Jul 2026).
  - v0.5.2 (18 Jul 2026): hatch-vcs auto-versioning only.
  - **v0.5.3 (14 Oct): bumps the NNsight backend to 0.8 plus CLT fixes.**
  - No release note mentions when GPT-OSS was added.
- **v0.5.3 details:** published by hannamw. Notes: "NNsight backend was upgraded to version 0.8" (PR #113), CLT skip-weight checkpointing, and `prune_graph` zero-weight edge fix. The page shows no year — [v0.5.3](https://github.com/decoderesearch/circuit-tracer/releases/tag/v0.5.3).
- **PyPI gotcha:** the PyPI package named `circuit-tracer` (0.5.0, 29 Mar 2026) points to a **fork (Caerii/circuit-tracer)**, not the official repo — [PyPI](https://pypi.org/project/circuit-tracer/).
- **Neuronpedia on-demand attribution:** the circuit-tracer paper says Neuronpedia offers on-demand attribution for Gemma-2 2B and Qwen-3 4B, with interventions — [BlackboxNLP 2025 paper](https://aclanthology.org/2025.blackboxnlp-1.14.pdf).

#### Neuronpedia
- **Open source:** public repo with an Apache-2.0 license; an MIT license file is also present. About 1.2k stars.
- **Services:**
  - webapp: Next.js frontend and API.
  - Postgres with pgvector.
  - inference: TransformerLens-based, one instance per model.
  - autointerp: EleutherAI Delphi.
  - **graph**: built on circuit-tracer.
  - sparsity.
  - **nla**: Natural Language Autoencoders.
- **Self-hosting:** at least 16 GB RAM. A CUDA GPU is needed for inference/graph beyond the smallest models. A local database starts empty.
- **Preloaded configs:** gpt2-small, gemma-2-2b-it and a DeepSeek-R1-distill Llama 8B.
- Source: [GitHub hijohnnylin/neuronpedia](https://github.com/hijohnnylin/neuronpedia).
- **Open-sourcing date:** blog post "Neuronpedia is Now Open Source", 31 Mar 2025 — [Neuronpedia blog](https://neuronpedia.org/blog). Date from a search listing; the post itself was not opened.
- **Hosted features:**
  - Feature dashboards, search, and testing on your own text.
  - Auto-interp, lists and embeds.
  - **Upload your SAEs**.
  - **Steering** with features or your own vectors.
  - **Circuit tracing**.
  - HeadVis (36,000+ heads across 37 models).
  - Jacobian Lens.
  - **NLA: "translate model activations into plain English"**.
  - **Assistant Axis: "track how far a model drifts from its assistant persona"**.
  - API and data downloads.
  - Self-hostable via the open-source "interp-engine".
  - Source: [Neuronpedia docs](https://docs.neuronpedia.org/).
- **Graph upload:**
  - Your own attribution graphs can be uploaded as JSON (required: `metadata`, `qParams`, `nodes`, `links`).
  - The model must already exist on Neuronpedia, matching `metadata.scan`.
  - Feature details come from your own CORS/HTTPS-hosted JSON or Neuronpedia source sets (Cantor-paired IDs).
  - Source: [Neuronpedia graph validator](https://www.neuronpedia.org/graph/validator).
- **CRM tracing on Qwen3-1.7B:** "Circuit Tracing with Interpretable Attention" (`llama-scope-2`), by the OpenMOSS team (Fudan).
  - "Complete Replacement Models" = transcoders + **LORSA** (Low-Rank Sparse Attention) nodes.
  - Neuronpedia supports CRM graph generation on **Qwen3-1.7B**.
  - It also offers a LORSA dashboard and QK tracing (top marginal and pairwise query/key feature contributors). Example source: `0-llamascope-2-lorsa-16k-k64`.
  - Source: [Neuronpedia llama-scope-2](https://www.neuronpedia.org/llama-scope-2).
  - The Qwen3-1.7B model page dates the release April 2026 — [Neuronpedia Qwen3-1.7B](https://www.neuronpedia.org/qwen3-1.7b) (via search snippet).
- **Prompt-length cap (unofficial):** a third-party project's PR reports the hosted CRM endpoint rejected a 17-token prompt with "max 10 tokens" for qwen3-1.7b — [patientwords-engine PR #96](https://github.com/michaeldgreenphd/patientwords-engine/pull/96). Single unofficial source.
- **"circuit-motifs" is a community tool, not a Neuronpedia feature.**
  - It applies network-motif analysis (from systems biology) to Neuronpedia attribution graphs.
  - Across 99 Neuronpedia graphs it found feedforward loops dominate, with a shared backbone across task types.
  - The tool is open source and runs on any Neuronpedia graph.
  - The LessWrong copy shows an "automated rejection" notice. This is a single-author preliminary result.
  - Sources: [LessWrong](https://www.lesswrong.com/posts/no6fw9WfrjR7EtQtX/applying-network-motif-analysis-to-transformer-attribution-1); [Substack](https://open2interp.substack.com/p/applying-network-motif-analysis-to).

#### NDIF / nnsight
- **NDIF:**
  - "All models are free for research use", with no GPU needed; create a free account at login.ndif.us.
  - Hardware: NCSA Delta (1× 8×H200 node + 6× 4×A40 nodes).
  - Listed models: **Llama-3.1-8B (Hot), Llama-3.1-70B (Hot), Llama-3.1-405B (Warm), gpt-oss-120b (Warm)**. These are base `meta-llama/Llama-3.1-*` IDs, not Instruct. More models can be "deployed on demand".
  - Quotas are not stated.
  - Source: [ndif.us](https://ndif.us/).
- **NDIF Workbench** (workbench.ndif.us): a web UI with a **Logit Lens** tool. Login is required "for all our models and features". Models showed as unavailable at fetch time — [NDIF Workbench](https://workbench.ndif.us/).
- **Funding:** NDIF is an NSF-funded ($9M) project led by Northeastern (David Bau) — [Northeastern News](https://news.northeastern.edu/2024/05/02/nsf-funds-democratizing-ai-research/).
- **Paper:** "NNsight and NDIF" was published at ICLR 2025 — [arXiv 2407.14561](https://arxiv.org/abs/2407.14561).
- **nnsight versions:**
  - Latest stable **0.7.0 (5 May 2026)**; pre-release **0.8.0rc1 (9 Sep 2026)**. MIT, Python ≥3.10 — [PyPI nnsight](https://pypi.org/project/nnsight/).
  - 0.8.0rc1 adds a greenlet engine rewrite, `TransformersModel`, tensor parallelism, **`dtype="nf4"` quantization** and `model.session()`.
  - 0.6.0 serialized custom code for NDIF and expanded vLLM support.
  - Source: [nnsight releases](https://github.com/ndif-team/nnsight/releases).
  - Note: circuit-tracer v0.5.3 targets "NNsight 0.8" while PyPI shows only 0.8.0rc1 — a possible pin friction.
- **nnsight-serve** (v0.7.0): "A lightweight FastAPI server that runs a vLLM engine and accepts serialized nnsight traces over HTTP".
  - Supports blocking and non-blocking modes and an optional API key.
  - **vLLM only**; the HF backend is not yet supported.
  - **Not documented as OpenAI-compatible**; it speaks the nnsight trace protocol.
  - Source: [nnsight v0.7.0](https://github.com/ndif-team/nnsight/releases/tag/v0.7.0).
- **Scope:** nnsight "supports local execution on any PyTorch model", and `remote=True` runs the same code on NDIF — [nnsight.net](https://nnsight.net/).

#### TransformerLens
- **Latest stable: v4.2.0 (10 Oct; year not shown, 2026 inferred).**
  - v4.2.0: built-in Logit Lens; RMSNorm offset fix for Qwen3.5/Qwen3Next/Nemotron.
  - v4.1.0 (28 Sep): "Fix GatedDeltaNet hooked forward on transformers >= 5.15".
  - v4.0.0 (21 Sep): Hooked-class deprecation.
  - v3.9.0: SigLIP vision-tower hooks.
  - v3.7.2: Gemma 4 and Mixtral 5.x support.
  - Source: [TL releases](https://github.com/TransformerLensOrg/TransformerLens/releases).
- **TransformerBridge** is the recommended path. It keeps raw HF weights; `enable_compatibility_mode()` restores HookedTransformer numerics.
  - Claims "15,000+ models across 140+ architecture families".
  - Names Qwen3.5 and Qwen3-Next (gated-delta-net), NemotronH, GraniteMoeHybrid, and experimental Mamba-1/2.
  - In 4.0, `HookedTransformer.from_pretrained` was removed. MIT license.
  - Source: [TL README](https://github.com/TransformerLensOrg/TransformerLens).
- The docs list `supported_architectures` modules for `qwen3`, `qwen3_5`, `gemma3`, `gpt_oss` and `llama`, plus a `utilities.quantization` module (details not shown) — [TL docs](https://transformerlensorg.github.io/TransformerLens/).

#### SAELens
- **Latest v6.54.5 (14 Oct 2026).**
- v6.54.2 (13 Oct) "fixed loading of Gemma Scope 2 affine skip connections in the correct orientation", which implies earlier versions loaded Gemma Scope 2 skip-transcoders incorrectly.
- Recent additions: AbsTopK SAE (v6.53.0) and covariance whitening (v6.52.0).
- Repo is now under decoderesearch.
- Source: [SAELens releases](https://github.com/decoderesearch/SAELens/releases).

#### Gemma Scope 2 (Google DeepMind)
- **Announced 22 Dec 2025** (McDougall, Conmy, Kramár, Lieberum, Rajamanoharan, Nanda).
  - SAEs on resid/MLP-out/attn-out plus MLP transcoders (with and without affine skip) for **every layer of all 10 Gemma 3 models** (270m, 1b, 4b, 12b, 27b; PT and IT).
  - Per layer: four SAEs (16k/262k widths × two L0s; "small" 10–20, "medium" 30–60, "big" 90–150). Wider sweeps at 25/50/65/85% depth, up to 1M width.
  - **CLTs only for 270m and 1b**; 4-layer crosscoders for every base size.
  - All JumpReLU with a Matryoshka loss.
  - Neuronpedia demo; inference and attribution graphs "coming soon" at launch.
  - Source: [Announcement](https://www.greaterwrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2).
- **HF model card (12B-IT):**
  - **License cc-by-4.0**.
  - Folders `resid_post`, `attn_out`, `mlp_out`, `transcoder` at 4 depths, plus `*_all` folders (every layer, with a narrower width/L0 range), `crosscoder` and `clt`.
  - Widths 16k/64k/256k/1m, recommended L0 "medium". The card says "large (60–150)", unlike the announcement's 90–150.
  - SAELens loading example: release `gemma-scope-2-12b-it-resid_post`, id `layer_12_width_16k_l0_small`.
  - Source: [HF google/gemma-scope-2-12b-it](https://huggingface.co/google/gemma-scope-2-12b-it).
- Press coverage: [InfoQ, Jan 2026](https://infoq.com/news/2026/01/google-gemma-scope-2).

#### Qwen-Scope (Qwen team)
- **Paper:** arXiv 2605.11887, "Qwen-Scope: Turning Sparse Features into Development Tools for LLMs", submitted **12 May 2026**. "14 groups of SAEs across 7 model variants from the Qwen3 and Qwen3.5 series", covering dense and MoE models — [arXiv](https://arxiv.org/abs/2605.11887).
- **HF repos** (all "Updated May 13"):
  - Qwen3-1.7B-Base: W32K, L0 50/100.
  - Qwen3-8B-Base: W64K, L0 50/100.
  - Qwen3-30B-A3B-Base: W32K L0 50; W128K L0 100.
  - Qwen3.5-2B-Base: W32K, L0 50/100.
  - **Qwen3.5-9B-Base: W64K, L0 50/100**.
  - **Qwen3.5-27B (no -Base): W80K, L0 50/100**.
  - Qwen3.5-35B-A3B-Base: W32K L0 50; W128K L0 100.
  - Source: [HF Qwen SAE listing](https://huggingface.co/models?author=Qwen&search=SAE).
- **27B card:**
  - License "**qwen**" (custom, not Apache-2.0).
  - Layers 0–63, one checkpoint per layer, residual stream, TopK (exactly 50 active).
  - Files are `layer{n}.sae.pt` (`W_enc`, `W_dec`, `b_enc`, `b_dec`), loaded via `torch.load` plus an HF forward hook. A Gradio demo is included. **SAELens is not mentioned.**
  - Source: [HF Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50).
- **Announcement and training details:** ModelScope announced "14 SAE weight sets across 7 models: Qwen3-1.7B, 8B, 30B-A3B and Qwen3.5-2B, 9B, 27B, 35B-A3B", trained on 0.5B tokens — [ModelScope on X](https://x.com/ModelScope2022/status/2052009415000076568).
  - A search summary of the paper says SAEs cover **all layers** on the residual stream and are trained on base models **"except Qwen3.5-27B, whose SAEs were trained on the instruct variant"**. MoE models also get wider SAEs — [arXiv HTML](https://arxiv.org/html/2605.11887v1) (via search summary).

#### Llama Scope (OpenMOSS)
- **Llama Scope:** 256 TopK SAEs on every layer and sublayer (residual, attention, MLP, transcoder) of **Llama-3.1-8B-Base**, with 32K and 128K features (arXiv 2410.20526). The attention-output SAEs have many inactive features — [HF OpenMOSS-Team/Llama-Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope/blob/main/README.md).
- **Llama Scope 2:** listed on Neuronpedia for Qwen3-1.7B (April 2026), with transcoders plus LORSA 16k k64. The config path is `OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B/...` — [Neuronpedia](https://www.neuronpedia.org/qwen3-1.7b/13-llamascope-2-lorsa-16k-k64) (via search snippet).

#### Goodfire (Ember, open SAEs)
- **Ember deprecated:** "Update (Feb 2026): Our SAE demo interface and API have been deprecated" — [Goodfire research page](https://www.goodfire.com/research/mapping-latent-spaces-llama).
- **Open SAEs:** open-sourced Jan 2025 for Llama-3.3-70B (layer 50) and Llama-3.1-8B — [Goodfire blog](https://www.goodfire.ai/blog/sae-open-source-announcement).
- **HF repos still listed:**
  - `Llama-3.1-8B-Instruct-SAE-l19` and `Llama-3.3-70B-Instruct-SAE-l50` (Jan 2025).
  - `DeepSeek-R1-SAE-l37` (Apr 2025).
  - **`Hackathon-gpt-oss-20b-SAE-l15` (Aug 2025)**.
  - Evo-2 and SDXL-Turbo SAEs.
  - The org page shows no license; check the repo cards.
  - Source: [HF Goodfire](https://huggingface.co/Goodfire).

#### Transluce (Monitor, Docent, Investigator)
- **Monitor** (23 Oct 2024; code release 31 Oct 2024):
  - Precompiled neuron descriptions for Llama-3.1-8B MLP neurons.
  - **Real-time display of important concepts ranked by activation or attribution**.
  - An AI linter, and semantic steering of neuron groups.
  - "Investigator Agents" were described as future work.
  - Code: github.com/TransluceAI/observatory.
  - Sources: [Transluce Monitor](https://transluce.org/observability-interface); [Transluce tools](https://transluce.org/tools).
  - A search summary describes observatory as MIT-licensed. I did not verify this on the repo.
- **Docent:**
  - Analyzes **agent transcripts**: searches for behaviors, measures their frequency, and turns vague questions into rubrics.
  - **Open-sourced 24 Sep 2025, Apache 2.0**, self-hostable; a hosted version continues.
  - It works on transcripts only; no activations are mentioned.
  - Sources: [Docent open-source post](https://transluce.org/docent/blog/open-source); [GitHub TransluceAI/docent](https://github.com/TransluceAI/docent).
- **Investigator:** a research method (investigator agents that elicit behaviors). It has a web app labeled "Transluce Model Investigator" at monitor.transluce.org. I found no standalone open-source product. Jailbreaking experiment code was released — [Transluce jailbreaking](https://transluce.org/jailbreaking-frontier-models).

#### Steering libraries
- **EasySteer** (ZJU-REAL):
  - Built on a vLLM fork; v0.29.0 announced 2026-09-10. Apache-2.0, ~304 stars, accepted to EMNLP 2026 System Demos.
  - Vector extraction: DiffMean, PCA, LAT, probe, SAE. Also LoReFT adapters, token/position/attention-head steering.
  - **An OpenAI-compatible server with per-request steering**.
  - **Per-request capture of `hidden_states`, `attention_heads` and `router_logits`**.
  - Source: [GitHub ZJU-REAL/EasySteer](https://github.com/ZJU-REAL/EasySteer).
  - Paper: claims a 10.8–22.3× speedup in v2 (Mar 2026) — [arXiv 2509.25175](https://arxiv.org/abs/2509.25175) (via search summary).
- **repeng** (vgel):
  - RepE control vectors, trained "in less than sixty seconds".
  - MIT, ~760 stars.
  - **`export_gguf` for llama.cpp**, which works "with any quant".
  - "Vector training currently does not work with MoE models".
  - Source: [GitHub vgel/repeng](https://github.com/vgel/repeng).
- **steering-vectors:** CAA/RepE steering for HF models (GPT, LLaMA, Gemma, Mistral, Pythia). MIT, ~164 stars, no tagged releases — [GitHub steering-vectors](https://github.com/steering-vectors/steering-vectors).
- **SteerMoE:**
  - "Steering MoE LLMs via Expert (De)Activation" (ICLR 2026; UCLA/Adobe/LMU). Repo: adobe-research/SteerMoE.
  - Detects and toggles behavior-linked experts. Reports safety up to +20% and faithfulness +27%; activating unsafe experts cut safety by 41%.
  - Sources: [ICLR 2026](https://iclr.cc/virtual/2026/poster/10006841); [alphaXiv](https://alphaxiv.org/abs/2509.09660). The repo itself was not opened.

#### Kiji Inspector (Dataiku 575 Lab)
- **Release:** blog dated 15 Mar 2026; Apache 2.0; `pip install kiji-inspector`; github.com/dataiku/kiji-inspector.
- **Models:**
  - Nemotron-3-Nano-30B-A3B (BF16/BF8).
  - Nemotron-3-Super-120B-A12B.
  - gemma3-27b-it (experimental).
  - Other models need a custom SAE.
- **Method:**
  - Captures activations at the **"decision token" where the model commits to a tool choice**.
  - Trains an unsupervised JumpReLU SAE.
  - Contrastive same-intent/different-tool pairs serve as probes.
  - LLM labeling plus fuzzing evaluation.
- **Outputs:** a Python API and decision reports. **No UI.**
- **Hardware:** the paper used 4× B200.
- **Scope:** explains single tool-selection decisions, not multi-step runs. Early-stage.
- Sources: [Dataiku blog](https://www.dataiku.com/blog/introducing-kiji-inspector); [BusinessWire, 16 Mar 2026](https://www.businesswire.com/news/home/20260316832746/en/Dataiku-Now-Supports-NVIDIA-Nemotron-Delivering-One-of-the-First-Open-Source-Frameworks-for-Enterprise-AI-Agent-Explainability).

#### No-code and consumer tools
- **LLM Trace** (MRSogzz/llm_trace_viewer):
  - A Python CLI writes `run.json`. A **single offline HTML viewer** shows attention, logit lens, MLP activations, residual norms and KV cache per layer and decoding step.
  - Backends: HF (fp16, bnb-4bit) and **llama.cpp GGUF** (ggml eval callback; greedy only).
  - Single prompt only; no MoE. 1 star, 1 commit; Windows untested.
  - Source: [GitHub llm_trace_viewer](https://github.com/MRSogzz/llm_trace_viewer).
- **dotLLM** (C# GGUF engine): logit lens and activation capture are planned for "Phase 7" — [kokosa.dev](https://kokosa.dev/blog/2026/dotllm/) (via search summary, unverified).
- **NDIF Workbench** (web logit lens) — see NDIF above.

### Inferences
- The reuse map for Brainscan:
  - **Recording substrate:** nnsight (any HF/PyTorch model) or TransformerLens 4.x TransformerBridge.
  - **Feature dictionaries:**
    - SAELens for Gemma Scope 2.
    - Custom `.pt` loader for Qwen-Scope (not SAELens-native per its card).
    - Goodfire and Llama Scope for Llama.
  - **Graphs:** circuit-tracer.
  - **Public viewing:** Neuronpedia (graph upload schema plus its feature dashboards).
  - **Transcript-level agent analysis:** Docent.
- None of these needs rebuilding. Brainscan's differentiation is the time axis: multi-step, tool-call-aligned recording.
- The circuit-tracer "v0.5.2" claim was probably copied from a summary written between 18 Jul and 14 Oct 2026. The model list is unchanged in v0.5.3.
- The Qwen-Scope "April 2026" date likely conflates it with Llama Scope 2 (OpenMOSS, Qwen3-1.7B, April 2026 on Neuronpedia). Qwen-Scope itself is dated May 2026 (arXiv 12 May; HF repos 13 May). The ModelScope tweet's snowflake ID decodes to about 6 May 2026; that is my arithmetic, not a stated date.

### Gaps
- Exact Qwen-Scope publication day (the Qwen blog page did not render). Whether Qwen-Scope is on Neuronpedia or loadable in SAELens. Not verified.
- Neuronpedia's full list of models with on-demand graph generation, rate limits, and the official prompt-length caps. The docs page did not list them.
- NDIF: whether accounts need approval, and any quotas. Not stated.
- The licenses of Goodfire's HF SAE repos and of Transluce observatory were not verified on the repo pages.
- When GPT-OSS was added to circuit-tracer is not in any release note.
- SteerMoE code status and the EasySteer model list (Qwen3.5? hybrid?) were not verified.
- I found no interpretability plugin or viewer for LM Studio or Ollama.

## Q2. Agentic / multi-turn recording, live monitoring, and "activation recording behind an OpenAI-compatible endpoint"

### Takeaway
No tool I found records activations across a multi-step agent run, time-locked to tool calls and errors, and served behind an OpenAI-compatible endpoint. That is the open slot for Brainscan. The pieces exist separately:
- **Behavior-only agent recorders:** agent-blackbox and Docent.
- **Single-decision activation analysis:** Kiji Inspector.
- **A research paper with probes before each action in multi-step traces:** arXiv 2605.06890, with no code linked.
- **Live concept displays for chat:** Transluce Monitor (Llama-3.1-8B) and Neuronpedia's Assistant Axis.
- **Inference servers that can capture or intervene:**
  - EasySteer: OpenAI-compatible server plus per-request hidden-state capture.
  - nnsight-serve: vLLM-backed, but speaks the nnsight protocol, not OpenAI.

### Cited Findings
- **agent-blackbox:** "A flight recorder for AI agents: record every OpenAI-compatible chat turn, replay it, and audit for actions the model claimed but never took."
  - Sits as a proxy in front of Ollama, llama.cpp, vLLM or hosted APIs, and logs messages, tools, replies, tool calls, timing and tokens.
  - **No model internals.**
  - Source: [GitHub KhaiB10/agent-blackbox](https://github.com/KhaiB10/agent-blackbox).
- **EasySteer:** "OpenAI-compatible server" with per-request and server-level steering, and per-request capture of `hidden_states`, `attention_heads` and `router_logits` (tensor-parallel aware) — [GitHub ZJU-REAL/EasySteer](https://github.com/ZJU-REAL/EasySteer). Whether captured tensors come back through the OpenAI `/v1` endpoint is not stated on the page.
- **nnsight-serve:**
  - A FastAPI plus vLLM server accepting serialized nnsight traces, with blocking and non-blocking modes and an API key.
  - The client needs only a meta model (no GPU).
  - Not documented as OpenAI-compatible; no multi-turn or streaming mentioned.
  - Source: [nnsight v0.7.0](https://github.com/ndif-team/nnsight/releases/tag/v0.7.0).
- **SGLang:** search could not confirm a `return_hidden_states` field on its OpenAI chat endpoint — [SGLang OpenAI API docs](https://docs.sglang.ai/basic_usage/openai_api_completions.html). Unverified.
- **Kiji Inspector:** activations at the tool "decision token"; explains single tool-selection decisions; no UI; no multi-step trace recording — [Dataiku blog](https://www.dataiku.com/blog/introducing-kiji-inspector).
- **"Beyond the Black Box: Interpretability of Agentic AI Tool Use"** (Tatsat and Shater; v1 7 May, v4 5 Jul 2026):
  - SAEs plus linear probes read model states "before each action" to infer whether a tool is needed and the risk of the next tool action.
  - Probes were trained on **multi-step traces** from the NVIDIA Nemotron function-calling dataset and applied to **GPT-OSS 20B and Gemma 3 27B**.
  - **No code repository linked.**
  - Source: [arXiv 2605.06890](https://arxiv.org/abs/2605.06890).
- **Docent:** transcript-level analysis of agent behavior ("is my model reward hacking"); Apache 2.0, self-hostable; no activations — [Docent open-source](https://transluce.org/docent/blog/open-source).
- **Transluce Monitor:** "real-time display of important concepts, ranked by activation or attribution" in chat, for Llama-3.1-8B neurons — [Transluce Monitor](https://transluce.org/observability-interface).
- **Neuronpedia Assistant Axis:** tracks drift from the assistant persona — [Neuronpedia docs](https://docs.neuronpedia.org/).
- **circuit-tracer chat formatting:** since v0.4.0 it requires a chat-format start for Gemma-3-IT. Graphs are per prompt, with no notion of multi-turn tool episodes — [circuit-tracer releases](https://github.com/safety-research/circuit-tracer/releases).
- **Agent-level tracing without activations:** OpenAI Agents SDK traces and OpenTelemetry instrumentation (spans for LLM calls and tool invocations) — [OpenAI integrations/observability](https://developers.openai.com/api/docs/guides/agents/integrations-observability).
- **Open behavior-level toolkits:** AgentDiagnose (EMNLP 2025 demo) diagnoses trajectories behaviorally — [ACL Anthology](https://aclanthology.org/2025.emnlp-demos.15/).

### Inferences
- **The most reusable architecture** is an agent-blackbox-style OpenAI-compatible proxy (turn and tool-call timeline) in front of a capture-capable backend. The aligned record is then exported to Docent-like transcript views plus Neuronpedia-style feature dashboards. Capture backend options:
  - EasySteer/vLLM (fast; Linux/WSL2).
  - An HF + nnsight server (any model, slower).
- **Kiji's "decision token" idea and the probes in arXiv 2605.06890** are natural features to support: mark the token where a tool name is committed. Brainscan could reproduce them openly on GPT-OSS-20B or Gemma 3, since 2605.06890 released no code.
- **Live monitoring** of agent runs with SAE features is not offered by any open tool I found. Monitor is chat-only on one 2024 model, and Neuronpedia is hosted and per-prompt.

### Gaps
- Whether EasySteer returns hidden states through its OpenAI endpoint, and its streaming behavior.
- Whether SGLang or vLLM mainline expose hidden states over OpenAI APIs in current versions. Not verified.
- I did not check closed or enterprise products (e.g., Goodfire's current commercial offerings after the Ember deprecation).

## Q3. Practical compatibility on a single RTX 4090 (24 GB) + 128 GB RAM, Windows/WSL2; SAE/transcoder availability per model

### Takeaway
- **Qwen3.5-9B** is a **hybrid Gated DeltaNet + gated-attention model with a vision encoder** (32 layers, d=4096, Apache-2.0).
  - TransformerLens 4.x explicitly supports Qwen3.5 and GatedDeltaNet; nnsight wraps any HF/PyTorch model.
  - Qwen-Scope has all-layer residual SAEs for **Qwen3.5-9B-Base**. There are no transcoders, and circuit-tracer does not support it.
- **GPT-OSS-20B:** current transformers docs say **MXFP4 runs on compute capability ≥7.5** (which includes Ada) with Triton ≥3.4 plus `kernels`, at about 16 GB. Older reports said Hopper-only. Without the kernels it dequantizes to bf16 and will not fit in 24 GB. **Use WSL2**: Triton and vLLM are Linux-first.
- **Qwen3-8B, Llama-3.1-8B-Instruct and Gemma-3 ≤4B** fit in bf16. Each has SAEs or transcoders.

### Cited Findings
- **Qwen3.5-9B architecture:**
  - Layout "8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))", 32 layers, hidden 4096, FFN 12,288, vocab 248,320.
  - Context 262,144 natively (1.01M with YaRN).
  - "Causal Language Model with Vision Encoder"; pipeline image-text-to-text; loaded with `AutoModelForMultimodalLM`; model_type `qwen3_5`.
  - Needs the "latest transformers", with main branch recommended.
  - Apache-2.0; citation dated Feb 2026; base model **Qwen/Qwen3.5-9B-Base**.
  - Source: [HF Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B).
- **TransformerLens Qwen3.5 support:** the README names Qwen3.5 and Qwen3-Next (gated-delta-net) as supported — [TL README](https://github.com/TransformerLensOrg/TransformerLens). v4.1.0 fixed "GatedDeltaNet hooked forward on transformers >= 5.15" and v4.2.0 fixed the Qwen3.5 RMSNorm offset — [TL releases](https://github.com/TransformerLensOrg/TransformerLens/releases).
- **TransformerLens other architectures:** adapter modules exist for `gpt_oss`, `gemma3`, `qwen3` and `llama` — [TL docs](https://transformerlensorg.github.io/TransformerLens/).
- **nnsight:** local execution on "any PyTorch model" — [nnsight.net](https://nnsight.net/). 0.8.0rc1 adds `dtype="nf4"` quantization — [nnsight releases](https://github.com/ndif-team/nnsight/releases).
- **GPT-OSS MXFP4 per transformers docs:**
  - MXFP4 needs Accelerate, `kernels`, Triton ≥3.4 (built in with PyTorch 2.8) and **"NVIDIA GPU Compute Capability ≥ 7.5"**.
  - GPT-OSS-20B "only require[s] 16GB of memory".
  - Kernels are pulled automatically from `kernels-community/triton_kernels`; the checkpoint is 13.8 GB on disk.
  - Source: [HF transformers MXFP4 docs](https://huggingface.co/docs/transformers/main/en/quantization/mxfp4).
- **Older, conflicting reports:**
  - 2025 HF threads report "MXFP4 quantized models is only supported on GPUs with compute capability >= 9.0", and a default dequantization to bf16 when Triton ≥3.4 or `kernels` is missing — [HF discussion #61](https://huggingface.co/openai/gpt-oss-20b/discussions/61); [HF discussion #53](https://huggingface.co/openai/gpt-oss-20b/discussions/53).
  - OpenAI's cookbook said MXFP4 needs Hopper or later, including RTX 50xx — [OpenAI cookbook](https://cookbook.openai.com/articles/gpt-oss/run-transformers) (via search summary).
  - A community repo runs GPT-OSS-20B on Windows with an RTX 4090 in under 16 GB using `triton-windows<3.5` and a transformers fork — [GitHub zer0int](https://github.com/zer0int/GPT-OSS-20B-Windows-16GB-RTX4090).
- **vLLM on Windows:** no official native Windows support; WSL2 is the standard route. Native community builds exist — [DEV Community](https://dev.to/aivrar/running-vllm-natively-on-windows-4ahm). Docker Model Runner supports vLLM on Windows via WSL2 for NVIDIA cards with compute capability ≥8.0 (Docker Desktop 4.54, Dec 2025) — [Docker blog](https://www.docker.com/blog/docker-model-runner-vllm-windows).

**SAE/transcoder availability:**
- **Qwen3.5-9B:**
  - Qwen-Scope residual SAEs on all layers, trained on Qwen3.5-9B-**Base**, W64K, L0 50/100 — [HF Qwen SAE listing](https://huggingface.co/models?author=Qwen&search=SAE).
  - Not in circuit-tracer's transcoder list — [circuit-tracer README](https://github.com/decoderesearch/circuit-tracer).
- **Qwen3-8B:**
  - Qwen-Scope SAEs (8B-Base, W64K, L0 50/100) — [HF listing](https://huggingface.co/models?author=Qwen&search=SAE).
  - `mwhanna/qwen3-8b-transcoders` PLTs for circuit-tracer — [README](https://github.com/decoderesearch/circuit-tracer).
- **Llama-3.1-8B(-Instruct):**
  - `facebook/crv-8b-instruct-transcoders` (TopK PLTs, circuit-tracer ≥v0.5.0) — [README](https://github.com/decoderesearch/circuit-tracer).
  - Goodfire `Llama-3.1-8B-Instruct-SAE-l19` (single layer) — [HF Goodfire](https://huggingface.co/Goodfire).
  - Llama Scope, all layers of the **Base** model — [HF Llama-Scope](https://huggingface.co/OpenMOSS-Team/Llama-Scope/blob/main/README.md).
  - Transluce Monitor neuron descriptions (Llama-3.1-8B MLP neurons) — [Transluce](https://transluce.org/observability-interface).
  - NDIF hosts Llama-3.1-8B (base) remotely — [ndif.us](https://ndif.us/).
- **GPT-OSS-20B:**
  - CLT `mntss/clt-131k` (circuit-tracer) — [README](https://github.com/decoderesearch/circuit-tracer).
  - Goodfire `Hackathon-gpt-oss-20b-SAE-l15` — [HF Goodfire](https://huggingface.co/Goodfire).
  - Used in the agentic probe paper — [arXiv 2605.06890](https://arxiv.org/abs/2605.06890).
  - NDIF hosts gpt-oss-120b, not 20B — [ndif.us](https://ndif.us/).
- **Gemma 3 (270M–27B, PT/IT):**
  - Gemma Scope 2 SAEs and transcoders on every layer; CLTs for 270M and 1B only — [Announcement](https://www.greaterwrong.com/posts/YQro5LyYjDzZrBCdb/announcing-gemma-scope-2).
  - circuit-tracer Gemma-3 PLTs (nnsight backend) — [README](https://github.com/decoderesearch/circuit-tracer).
  - Kiji supports gemma3-27b-it experimentally — [Dataiku](https://www.dataiku.com/blog/introducing-kiji-inspector).

### Inferences
- **Memory arithmetic (bf16 ≈ 2 bytes per parameter; my calculation, not from a source):**
  - 8–9B models need about 16–19 GB for weights, which fits a 24 GB 4090 with modest context. Qwen3.5-9B with its vision tower is the tightest. Long agent contexts (KV cache) may require nnsight 0.8's nf4 or CPU offload into the 128 GB RAM.
  - Gemma-3-12B at about 24 GB in bf16 needs quantization or offload. Gemma-3-27B is out of reach in bf16.
  - GPT-OSS-20B should fit in MXFP4 (about 16 GB per the HF docs) on WSL2. Dequantized bf16 (about 40+ GB) would need heavy CPU offload.
- **Storage for recordings:** recording the full residual stream costs about d_model × 2 bytes × n_layers per token. For Qwen3.5-9B that is 4096 × 2 × 32 ≈ 256 KB per token, or about 8 GB per 32k-token trace. Brainscan will likely need layer subsets, SAE-encoded sparse codes, or float8/fp16 compression.
- **Compute capability:** the RTX 4090 is compute capability 8.9 (widely documented; not separately cited here). This satisfies HF's "≥7.5" MXFP4 requirement and Docker/vLLM's "≥8.0".
- **circuit-tracer is the weak point on Qwen3.5:** no transcoders exist, and its nnsight backend is "experimental". Brainscan should use SAE-feature recording (Qwen-Scope) on Qwen3.5-9B, and reserve attribution graphs for Qwen3-8B, Llama-3.1-8B-Instruct, Gemma-3 and GPT-OSS-20B.
- **The Qwen3.5-9B SAEs are trained on the Base model.** Applying them to the post-trained 9B is plausible: the Qwen card for the 27B says base-trained SAEs "can also be used to explore post-trained checkpoints". Expect some reconstruction-quality loss; this is untested.

### Gaps
- No primary-source test found of TransformerLens TransformerBridge or nnsight on **Qwen3.5-9B's multimodal wrapper** (text-only path vs. `AutoModelForMultimodalLM`). Likewise for TransformerLens `gpt_oss` with MXFP4-packed experts (hooks on dequantized vs. packed weights).
- No measured VRAM figures found for circuit-tracer attribution on 8B models at 24 GB.
- No official statement on whether Triton/MXFP4 works under **native** Windows. Only community forks were found; assume WSL2.

## Q4. Gaps: what non-programmers and GPU-less users cannot do; what is missing for public sharing of recordings

### Takeaway
Without code or a GPU, people today can browse precomputed features, steer, and build short-prompt attribution graphs, but only on Neuronpedia's hosted models (plus the NDIF Workbench logit lens behind a login and Transluce's 2024 Monitor). They cannot:
- run their own local model (LM Studio or Ollama) and see inside it;
- record a multi-turn or agent session;
- or share such a recording.

For public sharing, raw activations should be treated as **equivalent to the text**: prompts up to 4,112 tokens have been inverted from middle-layer Llama-3 states. Model licenses are also silent or ambiguous on activations, and Gemma's "Model Derivatives" clause explicitly mentions "intermediate data representations".

### Cited Findings
- **No-code / no-GPU options and their limits:**
  - Neuronpedia offers hosted dashboards, steering, circuit tracing, NLA and Assistant Axis with no install — [Neuronpedia docs](https://docs.neuronpedia.org/).
  - Graph upload requires the model to already exist on Neuronpedia — [graph validator](https://www.neuronpedia.org/graph/validator).
  - An unofficial report cites a 10-token cap on hosted Qwen3-1.7B CRM graphs — [PR #96](https://github.com/michaeldgreenphd/patientwords-engine/pull/96).
  - NDIF Workbench needs a login, and its models showed as "unavailable" when fetched — [Workbench](https://workbench.ndif.us/).
  - Goodfire's hosted SAE demo and API were deprecated in Feb 2026, removing a hosted 70B-scale SAE playground — [Goodfire](https://www.goodfire.com/research/mapping-latent-spaces-llama).
- **Self-hosting Neuronpedia:** needs make, Postgres and a CUDA GPU for non-trivial models. The Delphi scorer does not run on Mac MPS or CPU — [Neuronpedia repo](https://github.com/hijohnnylin/neuronpedia).
- **Local GGUF viewers:** the only one found (LLM Trace) is CLI-driven, single-prompt and untested on Windows — [llm_trace_viewer](https://github.com/MRSogzz/llm_trace_viewer). repeng can export control vectors to GGUF for llama.cpp, but that is steering, not viewing — [repeng](https://github.com/vgel/repeng).
- **Privacy:**
  - "Depth Gives a False Sense of Privacy: LLM Internal States Inversion" (USENIX Security 2025) recovers prompts up to 4,112 tokens from internal states. A medical prompt was inverted with **86.88 token F1 from a middle layer of Llama-3**.
  - Differential privacy reduces the attack only at a large utility cost.
  - Sources: [arXiv 2507.16372](https://arxiv.org/pdf/2507.16372); [USENIX](https://www.usenix.org/conference/usenixsecurity25/presentation/dong-tian).
- **Gemma terms** (last modified 1 Apr 2026):
  - "Model Derivatives" include models created by transferring patterns of Gemma's "weights, parameters, operations, or Output", explicitly including "distillation methods that use intermediate data representations".
  - Redistributors must pass on the agreement and the Prohibited Use Policy restrictions.
  - Activations are not mentioned directly.
  - Source: [Gemma Terms](https://ai.google.dev/gemma/terms).
- **Gemma Scope 2 license:** CC-BY-4.0 — [HF](https://huggingface.co/google/gemma-scope-2-12b-it).
- **Llama 3.1 license:**
  - Redistributing "Llama Materials" requires a license copy and "Built with Llama".
  - Models trained on outputs must have names starting with "Llama".
  - The 700M MAU clause applies.
  - Intermediate activations are not mentioned.
  - Source: [Llama 3.1 LICENSE](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/blob/main/LICENSE).
- **Qwen licenses:** Qwen3.5-9B is Apache-2.0 — [HF](https://huggingface.co/Qwen/Qwen3.5-9B). The Qwen-Scope SAEs use the custom "qwen" license — [HF](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50).

### Inferences
- **Public sharing mode:** Brainscan should default to sharing **derived, sparse, labeled artifacts** (top-k SAE feature IDs and labels per step, probe scores, graph JSON in the Neuronpedia schema) rather than raw residual tensors. It also needs a text-scrubbing pass (PII redaction of prompts, tool outputs and file contents), because the activations themselves can leak scrubbed text.
- **"Redact text but keep activations" is not a safe combination.** Redaction has to happen before inference, or the activations for redacted spans have to be dropped.
- **License-wise, Apache-2.0 models are the cleanest to share** (Qwen3.5-9B and Qwen3-8B; GPT-OSS presumed Apache-2.0 but not verified here). Gemma recordings should carry the Gemma terms notice. Llama recordings should carry "Built with Llama" plus a license copy to be safe. Feature labels derived from Qwen-Scope inherit the "qwen" SAE license.
- **Positioning:** the unmet need is a one-click "run my local agent and get a shareable scan" flow. That means an OpenAI-compatible endpoint on a consumer GPU, a scanner UI over the timeline, and a privacy-scrubbed publish step. Neuronpedia covers browsing features and graphs, Docent covers transcripts, and NDIF covers large models for programmers. Nobody covers the local agentic recording, review and sharing loop.

### Gaps
- No legal analysis was found on whether sharing activations or SAE codes counts as "Distribution" or a "Model Derivative" under the Gemma or Llama terms. This needs a lawyer or the vendors' own statements.
- The GPT-OSS license text was not fetched in this session.
- Whether Neuronpedia supports private or unlisted graph uploads, and its size limits, was not visible on the validator page.
- I found no existing open tool for privacy-scrubbing activation datasets specifically.
