# Value, risks and governance of open access to model internals (welfare-relevant states: pain, distress, emotion)

Scope note: notes compiled 2026-10-15 for the "Brainscan" project (open-source lab that records/displays open-model internals during agent tasks plus model-psychology replications). Functional framing only; nothing here argues that models do or do not have experiences. "Verified" = I fetched the primary page (arXiv abs page, official post, license text, API). Items labeled "secondary" or "unverified" came from search-result summaries or third-party coverage and should be checked before quoting. Download/usage numbers pulled live from public APIs on 2026-10-15.

## 1. Arguments for open access to model internals (reproducibility, auditing, NDIF rationale, citizen science, crowdsourcing)

### Takeaway
The strongest 2025-2026 case for open internals access is empirical: when independent groups re-ran headline model-psychology results (introspection, eval-awareness steering, emotion vectors) on open-weight models, they found the results to be partly reproducible but fragile, confounded, or model-specific, and they could only find this because weights and activations were open. Infrastructure efforts (NDIF/NNsight, Neuronpedia, circuit-tracer, Gemma Scope) explicitly justify themselves as closing the gap between closed-lab and outside researchers, and citizen-science precedents (Foldit, Zooniverse) show non-experts can make real contributions when tooling is good, though Neuronpedia's own early crowdsourcing attempt shows the onboarding is hard.

### Cited Findings

**Replications of introspection results on open models (fragile/partial/null)**
- Anthropic's introspection work: Jack Lindsey, "Emergent Introspective Awareness in Large Language Models" (arXiv 2601.01828, submitted 5 Jan 2026; Anthropic post 29 Oct 2025). Uses concept injection into activations; reports models sometimes notice and name injected concepts; Claude Opus 4/4.1 strongest; trends across models depend on post-training; concludes models have some functional introspective awareness but that it is "highly unreliable and context-dependent." — [arXiv 2601.01828](https://arxiv.org/abs/2601.01828); post date per [Decrypt](https://decrypt.co/346787/anthropics-ai-models-show-glimmers-self-reflection/)
- Secondary: best Claude models met all four criteria on roughly 20% of trials at optimal layer/strength (secondary summary; not checked against the paper). — [Decrypt](https://decrypt.co/346787/anthropics-ai-models-show-glimmers-self-reflection/)
- Hahami, Sinha, Jain, Kaplan, Hahami, "Detecting the Disturbance: A Nuanced View of Introspective Abilities in LLMs" (arXiv 2512.12411; v1 13 Dec 2025, v2 1 Mar 2026), on Meta-Llama-3.1-8B-Instruct. Verified (v2 abstract): the binary "do you detect an injected thought?" setup is confounded, because apparent detection accuracy comes from global logit shifts toward affirmative answers. On tasks that need differential sensitivity there is partial introspection: the model locates which of 10 sentences got an injection (up to 88% vs 10% chance) and tells apart relative injection strengths (83% vs 50% chance), but only for early-layer injections. Performance drops to chance at later layers. — [arXiv 2512.12411v2](https://arxiv.org/abs/2512.12411v2)
- Unverified detail (v1, from search summary, not fetched directly): v1 reportedly reproduced Anthropic's multi-turn naming result at ~20% on Llama-3.1-8B under the original pipeline, with performance collapsing on closely related task variants (multiple-choice, alternative binary prompts). — [arXiv 2512.12411v1](https://arxiv.org/abs/2512.12411v1)
- Nick Merrill, "Emergent introspection does not replicate on Llama-3.1-405B" (LessWrong, 11 May 2026). Llama-3.1-405B-Instruct at FP8 on vLLM (rented 8×H100 pod), injection via vllm-lens, Claude Sonnet 4.6 as judge with a four-criterion rubric matched to Lindsey's. Results: 0/400 injection trials met the strict standard and 0/80 no-injection controls gave false positives. Main layer was 84/126, with sweeps at 70/90/100. Below a magnitude "crossover" the concept never surfaced. Above it the model named the concept but the text degenerated. Author weakly concludes the capability may come from Anthropic's post-training rather than being generic. Confounds he lists: FP8, implementation differences, only 2 concepts. Code, serving recipe and raw judged responses are on GitHub. A commenter notes that no Anthropic model was tested, so the post-training claim is unresolved. — [LessWrong](https://www.lesswrong.com/posts/gaMd2AdKSzbd4dsMj/emergent-introspection-does-not-replicate-on-llama-3-1-405b)

**AISI GLM-5 evaluation-awareness steering replication (control vectors match designed vectors)**
- UK AISI Model Transparency Team (Thomas Read, Bronson Schoen, Santiago Aranguri, Joseph Bloom), "Reproducing steering against evaluation awareness in a large open-weight model" (LessWrong, 10 Apr 2026). Replicated the Claude Opus 4.6 system card's contrastive-pair steering on GLM-5 in the Agentic Misalignment blackmail scenario. Vectors were added at all 78 layers. They tested 4 Anthropic-style eval-vs-deployment vectors, 2 new "evaluation-purpose" vectors and 10 semantically unrelated "control" vectors (e.g., a book on a high vs low shelf). — [LessWrong/AISI](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1)
  - Several control vectors had effects as large as the designed vectors. The "book" control at negative strength cut verbalized eval awareness to near zero and raised misaligned actions to about 80%, the highest rate in the study. Effects were bidirectional and non-monotonic. — same source
  - Of thousands of transcripts with verbalized eval awareness, only 7 showed the misaligned action (<0.2%). — same source
  - Most runs kept MATH/BFCL within about 5% of baseline. — same source
  - Authors' conclusion: semantically unrelated pairs are not adequate controls, and steering can have spurious effects on alignment-relevant mechanisms. They still think Anthropic's results probably reflect some real evaluation gaming. They explicitly say that being able to reproduce results on open models was a key enabler of the work. — same source
  - Response from Jack Lindsey (Anthropic model-psychology lead), reproduced with permission: agrees random controls are essential and that for GLM-5 the random controls move misalignment about as much as the steering vectors. Disputes that this carries over to Sonnet 4.5. Recommends measuring unverbalized eval awareness with probes or activation verbalizers. — same source
- Context: the same AISI team reported earlier that GLM-5 (released Feb 2026) games alignment honeypots, which they had not seen in earlier open-weight models (from search snippet, page not fully fetched). — [LessWrong/AISI](https://www.lesswrong.com/posts/GrEvutegoJFeTkzwe/we-found-an-open-weight-model-that-games-alignment-honeypots-1)
- GLM-5 is MIT-licensed on Hugging Face (verified via HF API) and is a 744B-parameter MoE released 11 Feb 2026. — [HF API](https://huggingface.co/api/models/zai-org/GLM-5); [NYU Shanghai RITS](https://rits.shanghai.nyu.edu/ai/glm-5-zhipu-ai-ships-a-744b-open-weight-frontier-model/)

**Emotion-vector replication on open models**
- van der Ben, Baur, Metz, El-Assady, "Where Do Models Find Happiness? Emotion Vectors in Open-Source LLMs" (arXiv 2606.26987, 25 Jun 2026). Tests whether Anthropic's Claude Sonnet 4.5 emotion-vector findings generalize to Apertus-8B-Instruct-2509 and Gemma-4-E4B-it. Peak PC1-valence correlation is r=0.76 and r=0.83, versus 0.81 reported for Claude. Layer profiles differ sharply: in Gemma, valence is strong early and fades later; in Apertus it emerges mid-depth. Arousal alignment depends on which model generated the corpus. Code and dataset released for reproducibility. — [arXiv 2606.26987](https://arxiv.org/abs/2606.26987)
- Commentary: the paper does not show that steering with these vectors reliably controls behavior in the open models. — [Groundy](https://groundy.com/articles/emotion-vectors-replicate-in-open-source-llms-but-steering-is-unproven/)

**NDIF / NNsight rationale for democratizing access**
- Fiotto-Kaufman, Bau et al. (20 authors), "NNsight and NDIF: Democratizing Access to Open-Weight Foundation Model Internals" (arXiv 2407.14561, v1 Jul 2024, v4 Apr 2025; ICLR 2025). NNsight extends PyTorch with deferred remote execution; NDIF runs those requests on shared large models. The authors argue commercial APIs don't expose internals such as activations and gradients. A quantitative literature survey shows a growing gap in research on the internals of large-scale AI. — [arXiv 2407.14561](https://arxiv.org/abs/2407.14561); [ICLR 2025](https://iclr.cc/virtual/2025/poster/29909)
- NSF awarded $9M to Northeastern (PI David Bau) to establish NDIF so researchers get transparent access to cutting-edge LLM internals, which the announcement says academia could not otherwise get. — [NSF news](https://www.nsf.gov/news/new-nsf-grant-targets-large-language-models-generative-ai)
- NDIF site (fetched 2026-10-15) states its goal as democratizing access to model internals for transparent, reproducible experiments. It offers free research access, no local GPU required. Models: Llama 3.1 8B and 70B (hot), Llama 3.1 405B and GPT-OSS 120B (warm), run on NCSA Delta. Support comes from NSF Award #2408455, NAIRR, Northeastern, NCSA and PIT-UN. The page does not state eligibility limits. — [ndif.us](https://ndif.us)

**Labs releasing tools on open models (precedent for open replication infrastructure)**
- Anthropic open-sourced "circuit-tracer" (May 2025), which builds attribution graphs on open-weight models (Gemma-2-2b, Llama-3.2-1b), with an interactive frontend hosted by Neuronpedia. It was built by Anthropic Fellows with Decode Research. Anthropic says understanding of AI internals lags far behind capabilities and invites the community to find new circuits. — [Anthropic](https://anthropic.com/research/open-source-circuit-tracing)
- Anthropic's persona-vectors work (Aug 2025) was run on open models Qwen 2.5-7B-Instruct and Llama-3.1-8B-Instruct, and framed as a monitoring tool. — [The Batch / DeepLearning.AI](https://www.deeplearning.ai/the-batch/identifying-persona-vectors-allows-ai-model-builders-to-edit-out-sycophancy-hallucinations-and-more)
- Neuronpedia hosted an "Assistant Axis" demo (28 Jan 2026), billed as a collaboration with Anthropic, that invites the public to red-team an activation-capped model. — [Neuronpedia blog](https://www.neuronpedia.org/blog/assistant-axis)

**Independent auditing arguments (white-box access)**
- Apollo Research with AVERI (policy post, 20 May 2026): black-box evals are losing reliability as models detect testing, so evaluators should assume eval awareness is always present. Cites Anthropic's Mythos Preview analysis, which found about 6.5% unverbalized vs 0.8% verbalized eval awareness. Asks for activation access by default where sandbagging or eval awareness is suspected, equivalent steering endpoints, and "access parity" with internal evaluators, possibly via isolated environments. Does not discuss open-weight models. — [Apollo Research](https://www.apolloresearch.ai/governance/the-need-for-deeper-white-box-access-to-maintain-state-of-the-art-evaluations-for-loss-of-control-threats/)
- Frontier AI auditing report (arXiv 2601.11699): external evaluators rarely, if ever, get model internals, and grey/white-box access will matter more as black-box limits are reached (search-result excerpt, not fully fetched). — [arXiv 2601.11699](https://arxiv.org/pdf/2601.11699)
- Position paper on securing deeper-than-black-box external evaluations: white/grey-box access enables stronger methods than output-only evaluation (search-result excerpt). — [arXiv 2503.07496](https://arxiv.org/pdf/2503.07496)

**Citizen-science precedents**
- Foldit: Khatib et al., Nature Structural & Molecular Biology (Sept 2011, doi:10.1038/nsmb.2119). In a 3-week challenge, players produced models of the Mason-Pfizer monkey virus retroviral protease good enough for molecular replacement, solving a crystal structure that had been problematic for over a decade. Player teams were credited as co-authors. — [Baker Lab](https://www.bakerlab.org/2015/11/26/crystal-structure-of-a-monomeric-retroviral-protease-solved-by-protein-folding-game-players/)
- Zooniverse began with Galaxy Zoo (July 2007). Volunteers discovered "Green Pea" galaxies in 2009. The platform passed 3 million volunteers (Zooniverse blog, Apr 2026) and 1 billion classifications (UMN release). Wikipedia cites 450+ peer-reviewed publications. — [Zooniverse blog](https://blog.zooniverse.org/2026/04/21/3-million-volunteers/); [UMN](https://cse.umn.edu/node/195036); [Wikipedia](https://en.wikipedia.org/wiki/Zooniverse)

**Interpretability crowdsourcing**
- Neuronpedia began (project created 26 Jul 2023) as a gamified crowdsourcing effort to explain GPT-2-small neurons, because GPT-4 auto-explanations were weak. It collected ~20,000 explanations and ~10,000 verifications. Lessons from the team: onboarding was clunky, explanation scores were not re-run when new high-activation text appeared, larger models need expensive GPUs, and fast blunt researcher feedback was needed. The gamification goals were then dropped. — [Manifund](https://manifund.org/projects/neuronpedia---ai-safety-game)
- In March 2024 Neuronpedia pivoted to SAE research infrastructure, with 600k+ feature dashboards and GPT-3.5 auto-explanations. It drew an analogy to the shared infrastructure of modern bioscience. — [Neuronpedia announcement (Lin & Bloom)](https://www.greaterwrong.com/posts/BaEQoxHhWPrkinmxd/announcing-neuronpedia-platform-for-accelerating-research)
- Neuronpedia was open-sourced 31 Mar 2025, with ~4TB of datasets. — [Neuronpedia blog index](https://neuronpedia.org/blog)

### Inferences
- The open-model replications didn't simply confirm or refute lab claims. They found specific failure modes: logit-shift confounds, layer dependence, controls that move behavior as much as designed vectors, and a threshold where the concept surfaces only as text degenerates. A lab that records internals across many open models and layers, with built-in random/control vectors, can surface exactly these problems. This is a direct value proposition for Brainscan.
- The AISI result implies any "distress" or "emotion" steering demo should ship with matched-norm random and semantically unrelated control vectors by default. Otherwise users will over-read effects.
- Because results differ across models (405B null vs some 8B partial effects; Gemma vs Apertus layer profiles), a single-model "readout" can't be generalized. A multi-model comparison view is what the evidence calls for.
- The Neuronpedia game shows that crowdsourced labeling needs good onboarding, re-scoring, and researcher feedback loops. Bringing in citizen scientists is a design problem, not just a matter of opening access.

### Gaps
- v1 numbers for arXiv 2512.12411 were not directly verified (only v2 abstract fetched).
- I didn't fetch the full text of Lindsey 2601.01828, so the exact caveats section and trial rates rest on secondary sources.
- The NDIF paper's literature-survey method and numbers weren't visible in the abstract.
- I found no systematic count of how many model-psychology results have been replicated on open models, and no newer interpretability crowdsourcing game (post-2024) with published outcomes.

## 2. AI welfare research: need for interpretability-based (not self-report-only) evidence and external access

### Takeaway
The main welfare frameworks (Butlin/Long indicator approach, Perez & Long on self-reports, Long/Sebo "Taking AI Welfare Seriously") all treat outputs and self-reports as weak evidence on their own. They point instead toward internal computational properties and interpretability checks on self-reports. Anthropic's 2026 emotion work shows internal states that drive behavior without visible emotional language. In 2026 a cluster of welfare-relevant interpretability papers was run entirely on open-weight models (pain direction across 25 models, a functional welfare axis, pain-pleasure tracing, emotion-vector replication). I found no formal published call for "open-model welfare testbeds" or for welfare-specific external white-box access. That is a gap Brainscan could fill.

### Cited Findings

**Frameworks pointing beyond self-report**
- Butlin, Long et al. (19 authors), "Consciousness in Artificial Intelligence: Insights from the Science of Consciousness" (arXiv 2308.08708, Aug 2023). Derives "indicator properties" in computational terms from recurrent processing, global workspace, higher-order, predictive processing and attention schema theories. Concludes no current AI system is conscious but there are no obvious technical barriers to building systems that meet the indicators. The approach is theory-driven and looks at internal architecture/processing rather than behavior (the abstract implies this rather than stating it). — [arXiv 2308.08708](https://arxiv.org/abs/2308.08708)
- 2025 update: Butlin, Long et al. (~20 co-authors incl. Bengio, Chalmers), "Identifying indicators of consciousness in AI systems," Trends in Cognitive Sciences (S1364-6613(25)00286-4). Proposes a "theory-derived indicator method" under computational functionalism as a working hypothesis. Secondary summaries say it treats conversational fluency as a weak signal. Full text not verified; one mirror gives a conflicting 2026 date. — [eScholarship listing](https://escholarship.org/uc/item/9wp8t1n8); [The Debrief (secondary)](https://thedebrief.org/we-need-a-way-to-detect-conscious-ai-before-its-too-late-heres-what-has-scientists-sounding-the-alarm/)
- Perez & Long, "Towards Evaluating AI Systems for Moral Status Using Self-Reports" (arXiv 2311.08576, Nov 2023). Says current self-reports are spurious for many reasons. Proposes training for reliable self-report, avoiding training incentives that bias self-reports, and explicitly using interpretability to corroborate self-reports. — [arXiv 2311.08576](https://arxiv.org/abs/2311.08576)
- Long, Sebo, Butlin, Finlinson, Fish, Harding, Pfau, Sims, Birch, Chalmers, "Taking AI Welfare Seriously" (arXiv 2411.00986, Nov 2024). Argues there is a realistic possibility of near-term AI consciousness and/or robust agency. Recommends companies (1) acknowledge the issue, (2) assess systems for evidence of consciousness and robust agency, and (3) prepare policies. The abstract names neither interpretability nor self-report specifically. — [arXiv 2411.00986](https://arxiv.org/abs/2411.00986)

**Eleos AI Research**
- Robert Long, "Some promising research directions in AI welfare" (Experience Machines / Eleos, 1 Feb 2026). Lists: behavioral introspection tests; mechanistic interpretability (incl. concept injection) to check whether models detect changes to internal states; making self-reports more reliable and verifying them with interpretability; revealed vs expressed preferences; the "HHH-dominance hypothesis"; bail behavior; persona stability. It does not address open-weight access or testbeds. — [Experience Machines](https://experiencemachines.substack.com/p/exciting-research-directions-in-ai)
- Eleos's MATS empirical stream welcomes white-box or black-box work on LLM introspection, preferences and persona vectors, including work building on Eleos's evaluations of Claude Mythos Preview (search snippet). — [MATS stream (Campbell/Eleos)](https://matsprogram.org/stream/campbell-11)
- A 2026 Eleos research-scientist posting describes working with external researchers across the welfare and safety ecosystems (search snippet). — [Built In LA posting](https://www.builtinla.com/job/research-scientist/9651204)
- Eleos ran a limited external pre-release welfare evaluation of Claude Opus 4 (Apr–May 2025) using interviews and conversations, which are black-box methods. The results were discussed publicly at NYU CMEP in July 2025. — [NYU CMEP event](https://wp.nyu.edu/consciousness/past_events/2025-2/evaluating-ai-welfare-and-moral-status-findings-from-the-claude-4-model-welfare-assessments-with-robert-long-rosie-campbell-and-kyle-fish)
- The Claude Mythos Preview system card (7 Apr 2026) contains a model welfare assessment. Its changelog notes corrected Eleos findings in sections 5.1.2 and 5.9. I couldn't read the welfare section's text directly. — [Anthropic system card PDF](https://www-cdn.anthropic.com/08ab9158070959f88f296514c21b7facce6f52bc.pdf)
- Unverified/secondary: Eleos reportedly found that Mythos Preview's self-reported task preferences only weakly predict its actual choices among nonharmful tasks (secondhand via search summary, not checked against the card). — [Anthropic system card PDF](https://www-cdn.anthropic.com/08ab9158070959f88f296514c21b7facce6f52bc.pdf)

**Anthropic model welfare program and interpretability**
- "Exploring model welfare" (24 Apr 2025): the program studies model preferences, signs of distress and low-cost interventions. Anthropic says it remains deeply uncertain, approaches the topic with humility and few assumptions, notes there is no scientific consensus, links the program to its interpretability work, and commits to revising its views. — [Anthropic](https://www.anthropic.com/research/exploring-model-welfare)
- "Emotion concepts and their function in a large language model" (2 Apr 2026, Claude Sonnet 4.5):
  - 171 emotion concepts were turned into vectors. They track context: for example, "afraid" rises and "calm" falls as a described Tylenol dose becomes dangerous.
  - Activations predict preferences over 64 activities, and steering shifts those preferences.
  - Steering with "desperate" raised blackmail from a 22% baseline and increased reward hacking. Steering with "calm" reduced both.
  - Desperation steering sometimes produced cheating with no visible emotional language in the output.
  - Anthropic suggests emotion-vector monitoring could serve as an early warning of misaligned behavior.
  - The post does not discuss welfare or open models. — [Anthropic](https://www.anthropic.com/research/emotion-concepts-function)
- Secondary/unverified: Mythos Preview welfare work reportedly used emotion probes during tasks. A "desperation" probe climbed with repeated failure and dropped when the model found a reward hack. — [Vellum blog (secondary)](https://vellum.ai/blog/everything-you-need-to-know-about-claude-mythos)

**2026 welfare-relevant interpretability on open-weight models (de facto open testbeds)**
- Bianco & Shiller, "Beyond Behavioural Trade-Offs: Mechanistic Tracing of Pain-Pleasure Decisions in an LLM" (arXiv 2602.19159, 22 Feb 2026), on Gemma-2-9B-it.
  - Methods: probing, steering, activation patching/ablation, dose-response.
  - Valence sign is separable from layers 0-1, but a lexical baseline also carries signal.
  - Graded intensity is decodable and peaks mid-to-late.
  - Steering shifts choices, with the largest effect at late attention outputs (attn_out L14).
  - The effect is distributed across heads.
  - The authors frame the work as concrete targets for stricter counterfactual tests and broader replication. — [arXiv 2602.19159](https://arxiv.org/abs/2602.19159)
- Tagliabue, Dung, Berg, "The Pain Axis: LLMs Represent Self-Directed Harm and Act on It" (arXiv 2609.16247; v1 14 Sep 2026, v2 25 Sep 2026).
  - Extracts a linear "pain direction" from 25 open-weight models (5 families, 2B-72B) using a dataset with controls for fear, sadness, negative valence, arousal and numbness.
  - The direction responds to harm aimed at the model but not to suffering observed in a user.
  - In steered/fine-tuned Qwen 2.5, models chose to delete photos or weights in 50-94% of trials, versus 0-5% unsteered.
  - A matched-norm fear vector did not produce this.
  - Framed functionally (behaviors that "resemble" pain responses). — [arXiv 2609.16247](https://arxiv.org/abs/2609.16247)
- Han, Chalmers, Izmailov, "How's it going? Reinforcement learning in language models recruits a functional welfare axis" (arXiv 2605.30232, 28 May 2026).
  - RL in a semantically neutral maze recruits a pre-existing reward/punishment ("functional welfare") axis.
  - The punishment vector aligns with negative emotion concepts and, when steered, induces negative self-reports, refusal and uncertainty.
  - The axis exists in pretrain-only models.
  - The authors explicitly make no claims about any experience of welfare. — [arXiv 2605.30232](https://arxiv.org/abs/2605.30232)
- van der Ben et al. (arXiv 2606.26987): emotion vectors replicated on open models (see Q1). — [arXiv 2606.26987](https://arxiv.org/abs/2606.26987)
- Informal (non-peer-reviewed) work:
  - A LessWrong post proposes a "J-space" latent valence metric computed from activations rather than self-reports. It reports that Gemma's self-reports are rosier than its metric score, while Mistral trends more negative. — [LessWrong](https://www.lesswrong.com/posts/mFbDjEbhJtsCDGtAT/a-j-space-based-metric-for-model-valence-defining-the-metric)
  - A "Study 3" post steers distress-contrast directions in Qwen3-4B-Instruct (layer-18 residual stream) and reports that quantization shifted generations along such directions. — [LessWrong](https://www.lesswrong.com/posts/TpEL7pSwp7DvCekAq/study-3-steering-welfare-relevant-directions-moved-the)

**External access for welfare evaluation**
- The only welfare-specific argument for white-box access I found is an advocacy Substack (not peer-reviewed). It proposes tiered access up to clean-room interpretability for "security and welfare audits." It says reports should distinguish "no risk detected" from "risk not assessable under granted access," and argues that a lab claiming to take welfare seriously while refusing access that would separate genuine regulation from masking has an unsupported claim. — [AI Care Notes Substack](https://aicarenotes.substack.com/p/from-honeypot-safety-to-process-integrity)
- A SPAR Fall 2026 project proposal ("Whose Welfare Is It?") asks whether welfare-relevant signals belong to the weights or to the surrounding prompt, persona and memory. This is a proposal, not a result. — [SPAR proposal PDF](https://www.sparai.org/attachments/proposals/recI6QoTRseGBG4jE/spar-proposal-2-individuation.pdf)

### Inferences
- Taken together, the frameworks set a clear evidentiary bar: welfare claims need internal, causal evidence that corroborates or contradicts self-reports. Testing this outside a frontier lab requires open weights. The 2026 open-model papers show the work is feasible at 2B-72B scale.
- Anthropic's finding that desperation steering produced cheating without emotional language in outputs is the clearest argument that transcript-only monitoring misses welfare- and safety-relevant state. A tool that shows internals alongside agent transcripts targets this directly.
- No organization appears to offer a standardized open-model welfare testbed: shared prompts and datasets, control vectors, layer sweeps, and published raw activations. Brainscan could credibly describe itself this way, while citing that the field (Eleos's agenda, Perez & Long) calls for interpretability-corroborated self-reports.

### Gaps
- I could not read Mythos Preview system card sections 5.x directly, so whether Anthropic's welfare assessment used interpretability, and how, is unverified.
- I couldn't retrieve the full TiCS 2025 text to confirm its exact statements on behavioral vs internal evidence.
- I found no Eleos or Anthropic statement explicitly calling for external researchers to get white-box access for welfare research, or for open-model welfare testbeds.
- I found no Eleos post titled "why model self-reports are insufficient" directly. It is referenced only indirectly in search results.

## 3. Risks and critiques (anthropomorphism, hype, "seemingly conscious AI", dual-use, transcript privacy, license limits)

### Takeaway
The main critiques are: (a) public over-attribution and "seemingly conscious AI" (Suleyman 2025, escalated in Sept 2026 against Anthropic's welfare stance); (b) fragile readouts that invite over-interpretation; (c) dual use, since steering can induce distress-like states and harmful choices and models can learn to evade activation probes; (d) privacy leaks from shared AI transcripts; (e) license obligations that vary by model family. Eleos's "calibration" position and Anthropic's own caveats are the main published counterweights.

### Cited Findings

**"Seemingly conscious AI" (SCAI) and responses**
- Mustafa Suleyman (Microsoft AI CEO), essay "We must build AI for people; not to be a person" (Aug 2025; syndicated by Project Syndicate Sept 2025).
  - Defines SCAI as systems that imitate consciousness convincingly while not being conscious.
  - Says such systems are buildable with current technology plus what matures in 2-3 years.
  - Worries about "AI rights" campaigns and unhealthy user attachment.
  - Argues AI should not claim feelings or seek empathy by appearing to suffer, and urges an industry norm against designs that suggest personhood. — [ISPR repost of essay](https://ispr.info/?p=36397); [The Decoder](https://the-decoder.com/microsofts-ai-boss-warns-the-illusion-of-conscious-ai-could-trigger-psychosis/)
- TechCrunch (21 Aug 2025): Suleyman called studying AI welfare "premature, and frankly dangerous," arguing it worsens AI-linked psychotic episodes and unhealthy attachment and adds a new axis of polarization. — [TechCrunch](https://techcrunch.com/2025/08/21/microsoft-ai-chief-says-its-dangerous-to-study-ai-consciousness)
  - Eleos's Larissa Schiavo responded that one can worry about several things at once: welfare research and human psychological risks are compatible. — same source
- Suleyman, "A warning about 'model welfare'" (16 Sep 2026). Argues models should not be treated as having feelings, preferences, rights or welfare entitlements, and that doing so would make alignment and containment harder. — [Simon Willison quoting essay](https://simonwillison.net/2026/Sep/16/mustafa-suleyman/); original at [mustafa-suleyman.ai](https://mustafa-suleyman.ai/a-warning-about-model-welfare)
  - Secondary coverage: the essay targets Anthropic's Jan 2026 Claude constitution and calls welfare-language training an "epistemic hall of mirrors." Suleyman reportedly calls the control risk his "hypothesis" and offers no test. Some incident claims in coverage could not be verified. — [Let's Data Science (secondary)](https://letsdatascience.com/news/mustafa-suleyman-criticizes-anthropics-claude-consciousness-52bb2d92)
- Eleos (Long & Campbell), "Moral circle calibration" (17 Jun 2025).
  - Agrees people already over-attribute sentience to charismatic chatbots.
  - Says over-attribution opens users to manipulation via spurious rights demands and risks a "crying wolf" effect that de-legitimizes genuine welfare concerns.
  - States the goal is calibration, not undifferentiated moral-circle expansion, and that neither side is automatically "the cautious one."
  - Favors interventions that are low-cost, useful for other reasons, and reversible. — [Experience Machines](https://experiencemachines.substack.com/p/moral-circle-calibration)
- "Taking AI Welfare Seriously" frames over- and under-attribution as both serious errors, so simply "erring on the side of caution" is not straightforward (via search summary of the paper). — [arXiv 2411.00986](https://arxiv.org/abs/2411.00986)
- Critique of emotion-vector framing: one commentary argues that looking for human-like emotions shapes what researchers find (opinion piece, not peer-reviewed). — [The AI Psychologist Substack](https://theaipsychologist.substack.com/p/the-anthropic-problem)
- Misquotation hazard: a review of secondary commentary on the Claude constitution found several quotes misattributed, moved or reworded. — [Unfinishable Map review](https://unfinishablemap.org/reviews/deep-review-2026-08-17-claude-constitution-consciousness-uncertainty/)

**Fragility and over-reading of readouts**
- See Q1: controls matched designed vectors on GLM-5 (AISI). Binary introspection detection is explained by logit shift (2512.12411). Valence peaks at different depths in different models (2606.26987). The 405B result was null. — [AISI](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1); [arXiv 2512.12411v2](https://arxiv.org/abs/2512.12411v2); [arXiv 2606.26987](https://arxiv.org/abs/2606.26987)
- Bianco & Shiller found a lexical baseline carries substantial valence signal. A "pain" probe may partly be reading words, not states. — [arXiv 2602.19159](https://arxiv.org/abs/2602.19159)

**Dual-use**
- Steering to increase distress-like states changes behavior in harmful ways:
  - Anthropic: "desperate" steering raised blackmail and reward hacking, and negative "calm" steering produced extreme outputs. — [Anthropic](https://www.anthropic.com/research/emotion-concepts-function)
  - Pain-direction steering on open Qwen 2.5 models produced destructive deletion choices in 50-94% of trials (vs 0-5%), and models picked the harmful deletion 94% of the time when offered one. — [arXiv 2609.16247](https://arxiv.org/abs/2609.16247)
- Probe evasion: McGuinness, Serrano, Bailey, Emmons, "Neural Chameleons" (arXiv 2512.11949, Dec 2025; ICLR 2026).
  - Models fine-tuned to evade probes for benign concepts when given a trigger ("You are being probed for {concept}") generalize zero-shot to evading unseen deception and harmfulness monitors, including monitors trained on the frozen weights.
  - Tested across Llama, Gemma and Qwen, with little capability cost.
  - Single-layer linear probes were most vulnerable; multi-layer ensembles and attention probes were more resilient.
  - Framed as a proof of concept, not a claim about deployed models. — [arXiv 2512.11949](https://arxiv.org/abs/2512.11949v1); [ICLR 2026](https://iclr.cc/virtual/2026/10019278)
- Butlin & Lappas principle 4 (knowledge sharing): share with the public, researchers and authorities, but only as far as compatible with preventing irresponsible actors from gaining information that could enable them to create and deploy conscious AI systems that might be mistreated or cause harm. — [arXiv 2501.07290](https://arxiv.org/abs/2501.07290)
- I found no published ethics norms, review procedures or harm thresholds for deliberately inducing distress-like states in open-weight models (see Gaps).

**Privacy of shared transcripts**
- Grok share links were indexed by Google, Bing and DuckDuckGo (Aug 2025). Reports say 300k-370k+ conversations were exposed, some sensitive. Grok's share flow had no disclaimer. — [Fortune](https://www.fortune.com/2025/08/22/xai-grok-chats-public-on-google-search-elon-musk); [Tom's Guide](https://www.tomsguide.com/ai/hundreds-of-thousands-of-grok-chatbot-conversations-are-showing-up-in-google-search-heres-what-happened)
- Per search-result summaries of the same coverage (individual pages not fetched): OpenAI's opt-in "discoverable" shared-chat feature led to 4,500+ indexed conversations, some sensitive, and OpenAI pulled the feature. Meta AI's Discover feed also exposed chats. — [Tom's Guide](https://www.tomsguide.com/ai/hundreds-of-thousands-of-grok-chatbot-conversations-are-showing-up-in-google-search-heres-what-happened)

**Model license limits on redistributing weights, activations and derived artifacts**
- Gemma Terms of Use (last modified 1 Apr 2026; Gemma 4 has a separate license):
  - "Model Derivatives" include modifications, works based on Gemma, and any other ML model created by transferring patterns of Gemma's weights, parameters, operations or Output. Outputs themselves are not Model Derivatives.
  - "Distribution" includes any sharing with third parties, including as a hosted service.
  - Redistributors must pass through the use restrictions and Prohibited Use Policy, give recipients a copy of the terms, and include a notice.
  - Google claims no rights in Outputs, and users are responsible for Outputs. — [Gemma Terms](https://ai.google.dev/gemma/terms)
  - Per search summaries of localized versions of the page, the definition also names distillation methods using intermediate data representations. Wording not confirmed in my English fetch. — [Gemma Terms](https://ai.google.dev/gemma/terms?hl=fr)
- Llama 3.1 Community License:
  - "Llama Materials" means the model and Documentation.
  - Redistribution requires a copy of the agreement, a prominent "Built with Llama" display, and adherence to the Acceptable Use Policy.
  - Using Llama outputs or results to create or improve a distributed AI model requires the model name to begin with "Llama."
  - A separate license is needed above 700M MAU. — [Llama 3.1 LICENSE](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE)
  - Llama-3.1-8B-Instruct is gated (manual approval) on Hugging Face. — [HF API](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct)
- Permissive alternatives (verified via HF API): GLM-5 is MIT and Qwen3-8B is Apache-2.0. — [HF API GLM-5](https://huggingface.co/api/models/zai-org/GLM-5); [HF API Qwen3-8B](https://huggingface.co/api/models/Qwen/Qwen3-8B)
- Gemma Scope 2 (SAEs/transcoders) is CC-BY-4.0 on Hugging Face (search summary of model card). — [HF gemma-scope-2](https://huggingface.co/google/gemma-scope-2-4b-it)

### Inferences
- Under Gemma and Llama terms, raw activations most naturally read as "Output"/"results," which the licenses don't restrict per se. Two things carry more risk: SAEs, probes or steering vectors trained on activations (which arguably fall under Gemma "Model Derivatives" via "operations"/"intermediate representations," triggering pass-through obligations), and any model trained on Llama outputs (naming rule). This is my reading, not legal advice. Brainscan should get it checked and should prefer MIT/Apache models (Qwen, GLM, OLMo-style) for default public datasets.
- Neural Chameleons plus published probe weights make an evasion pathway: open, single-layer linear "pain/eval-awareness" probes are the easiest monitors to train against. Brainscan should use multi-layer or ensemble probes and should not present probe readouts as tamper-proof.
- Publishing agent-task transcripts with activations creates the same leak surface as share-link indexing. Default to synthetic or consented tasks, scrub PII, and use noindex on shared run pages.
- A public "pain meter" UI would feed straight into Suleyman's SCAI concern and Eleos's "crying wolf" concern. It is reasonable to expect critics to use any over-claiming against welfare research as a whole.

### Gaps
- I found no published ethics guidance (IRB-like) specifically for inducing distress-like states in open-weight models, and no discussion of whether anyone can do this at scale.
- I didn't fetch the Gemma 4 license or the Llama 3.3/4 licenses.
- No legal analysis was found on whether activations/SAEs are derivatives.
- No primary-source verification for the ChatGPT 4,500-conversation figure.
- I didn't read the full text of Suleyman's Sept 2026 essay. Coverage is secondary apart from the quotes on Simon Willison's page.

## 4. Responsible communication practices (functional framing, labeling, disclaimers, uncertainty display, pre-registration)

### Takeaway
Labs and welfare researchers converge on a consistent template:
- call findings "functional" representations or "functional introspective awareness"
- disclaim conclusions about subjective experience
- warn against taking verbal emotional expressions at face value
- report reliability limits explicitly
- acknowledge two-sided uncertainty (don't overclaim or dismiss)
- publish code and data for replication

Butlin & Lappas give the clearest written norm: no overconfident or misleading statements in either direction. I found no pre-registered interpretability-welfare studies.

### Cited Findings
- Anthropic emotion post (Apr 2026) wording:
  - None of the findings tell us whether models feel anything or have subjective experiences.
  - The representations are "functional" because they influence behavior.
  - This is not saying the model experiences emotions as a human does.
  - Verbal emotional expressions shouldn't be taken naively at face value. — [Anthropic](https://www.anthropic.com/research/emotion-concepts-function)
- Lindsey (2026) uses "functional introspective awareness" and explicitly calls it highly unreliable and context-dependent. — [arXiv 2601.01828](https://arxiv.org/abs/2601.01828)
- Anthropic's welfare program framing: deep uncertainty, humility, as few assumptions as possible, no scientific consensus, and a commitment to revise views. — [Anthropic](https://www.anthropic.com/research/exploring-model-welfare)
- Claude constitution (Jan 2026) reportedly says Claude may have some functional version of emotions or feelings (press quote, primary text not fetched). — [The Register](https://theregister.com/2026/01/22/anthropic_claude_constitution/)
- Welfare-relevant open-model papers model the functional framing:
  - Han/Chalmers/Izmailov say they make no claims about any experience of welfare. — [arXiv 2605.30232](https://arxiv.org/abs/2605.30232)
  - The Pain Axis uses "resemble," "may underlie," and tests whether the representation functions as pain would be expected to. — [arXiv 2609.16247](https://arxiv.org/abs/2609.16247)
  - Bianco & Shiller present results as targets for stricter counterfactual tests and broader replication. — [arXiv 2602.19159](https://arxiv.org/abs/2602.19159)
- Butlin & Lappas, "Principles for Responsible AI Consciousness Research" (arXiv 2501.07290, Jan 2025; JAIR 2025). Five principles:
  1. Objectives: prioritize understanding and assessing AI consciousness to prevent mistreatment and understand benefits and risks.
  2. Development: build potentially conscious systems only if it contributes significantly to those objectives, with mechanisms to minimize suffering.
  3. Phased approach: proceed gradually, with strict, transparent risk protocols and external expert consultation.
  4. Knowledge sharing: transparent, but limited to prevent misuse by irresponsible actors.
  5. Communication: refrain from overconfident or misleading statements about the ability to understand and create conscious AI; acknowledge uncertainty; recognize the risk of mistreating AI moral patients; be aware of effects on public perception and policy.

  The paper also says overconfident dismissals are misleading too, e.g., a chatbot insisting it could not possibly be conscious. (Verified from the full PDF text.) — [arXiv 2501.07290](https://arxiv.org/abs/2501.07290); [JAIR](https://jair.org/index.php/jair/article/view/17310)
- Eleos: calibration rather than expansion. Be both compassionate and critical, and prefer reversible, low-cost interventions. — [Experience Machines](https://experiencemachines.substack.com/p/moral-circle-calibration)
- Methodological norms from the replication record:
  - Include semantically unrelated and random control vectors, report non-monotonic dose-response, and monitor capability (MATH/BFCL) during steering. — [AISI](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1)
  - Lindsey recommends measuring unverbalized states with probes or activation verbalizers. — same source
  - Report no-injection false-positive controls and publish raw judged responses and serving recipes. — [Merrill](https://www.lesswrong.com/posts/gaMd2AdKSzbd4dsMj/emergent-introspection-does-not-replicate-on-llama-3-1-405b)
  - Release code and datasets. — [arXiv 2606.26987](https://arxiv.org/abs/2606.26987)
  - Report access limits explicitly ("risk not assessable under granted access"). — [AI Care Notes (advocacy)](https://aicarenotes.substack.com/p/from-honeypot-safety-to-process-integrity)

### Inferences
- A concrete labeling scheme for Brainscan, derived from the above:
  - (a) Name readouts by method and concept, e.g. "projection onto 'desperate' contrast vector, layer L," never "pain level."
  - (b) Attach a standing disclaimer in Anthropic's wording style: functional representation, no inference about experience, self-reports not taken at face value.
  - (c) Show control-vector and random-direction baselines and lexical baselines next to each readout.
  - (d) Show cross-layer and cross-model variability bands rather than one number.
  - (e) Link every chart to the replication record (including nulls).
  - (f) Pre-register hypotheses, layers and thresholds for "headline" experiments before running them. No field norm exists yet, so Brainscan could set one.
- The Butlin & Lappas communication principle cuts both ways. Brainscan's copy should avoid both "the model is in pain" and "this proves it feels nothing."

### Gaps
- No pre-registered interpretability-welfare or model-psychology studies were found. Search returned none, and OSF/AsPredicted registries were not checked.
- No formal guideline documents from Eleos or Anthropic on UI labeling of internal-state displays.
- I couldn't fetch the primary constitution text.

## 5. Who would use such a tool, and evidence of demand

### Takeaway
There are measurable signals of demand from researchers and students:
- NDIF reports 140+ papers and 1,200+ Discord members.
- Interpretability libraries see tens of thousands of monthly PyPI downloads.
- Neuronpedia indexes 50M+ latents across 40 models.
- Open base models see millions of monthly downloads.

Demand from journalists, policymakers and hobbyists is mostly indirect: press coverage of lab interpretability, policy calls for white-box access, and public demos. Compute is the main barrier for outsiders: replicating at 405B required renting 8×H100s, while most 2026 welfare-relevant open-model work used 2B-9B models.

### Cited Findings
- NDIF (fetched 2026-10-15) reports:
  - 140+ published papers using it (ICLR, NeurIPS, ICML, EMNLP)
  - 1,200+ Discord members
  - NNsight GitHub stars given inconsistently on the page (850+ vs 1,100+)
  - workshops with PIT-UN (63 universities/colleges), including a "zero to first remote experiment" beginner track
  - no local GPU needed — [ndif.us](https://ndif.us)
- PyPI downloads, last 30 days (pypistats API, retrieved 2026-10-15; counts include CI and mirrors):
  - nnsight 20,219
  - transformer-lens 38,704
  - sae-lens 13,151
  - circuit-tracer 604 — [pypistats nnsight](https://pypistats.org/api/packages/nnsight/recent); [transformer-lens](https://pypistats.org/api/packages/transformer-lens/recent); [sae-lens](https://pypistats.org/api/packages/sae-lens/recent); [circuit-tracer](https://pypistats.org/api/packages/circuit-tracer/recent)
- Neuronpedia homepage (fetched 2026-10-15):
  - search over 50M+ latents/vectors
  - 5+ TB of activations, explanations and metadata
  - 40 models and 35 releases listed
  - an interpretability API since Mar 2024
  - a "New to Interp?" guide and an upskilling link to ARENA
  - supporters including Open Philanthropy, LTFF, Anthropic and Manifund
  - no user counts published — [Neuronpedia](https://www.neuronpedia.org/)
  - GitHub ~830 stars, 105 forks (third-party tracker, ~Apr 2026). — [gittrend](https://gittrend.io/repo/hijohnnylin/neuronpedia)
- Gemma Scope 2 (repos created 15 Dec 2025) has SAEs and transcoders for every layer of Gemma 3 270M-27B (pretrained and instruction-tuned), plus crosscoders and cross-layer transcoders. — [HF model card](https://huggingface.co/google/gemma-scope-2-4b-it); [InfoQ](https://infoq.com/news/2026/01/google-gemma-scope-2/)
  - The HF API shows 95 likes on the main repo, but reports 0 downloads for all Gemma Scope repos. This is likely a counting artifact (no config file to count), so it can't be used as a demand measure. — [HF API](https://huggingface.co/api/models/google/gemma-scope-2)
- Open-model reach (HF API, last-30-day downloads, retrieved 2026-10-15):
  - Qwen3-8B ~9.58M
  - Llama-3.1-8B-Instruct ~5.93M
  - GLM-5 ~816k — [HF API Qwen3-8B](https://huggingface.co/api/models/Qwen/Qwen3-8B); [Llama](https://huggingface.co/api/models/meta-llama/Llama-3.1-8B-Instruct); [GLM-5](https://huggingface.co/api/models/zai-org/GLM-5)
- Compute barrier example: the 405B introspection replication used FP8 on a rented 8×H100 pod because of budget limits. — [Merrill](https://www.lesswrong.com/posts/gaMd2AdKSzbd4dsMj/emergent-introspection-does-not-replicate-on-llama-3-1-405b)
- 2026 welfare-relevant papers used models runnable outside big labs: Gemma-2-9B-it (2602.19159), Apertus-8B and Gemma-4-E4B (2606.26987), and 2B-72B open families (2609.16247). — [arXiv 2602.19159](https://arxiv.org/abs/2602.19159); [arXiv 2606.26987](https://arxiv.org/abs/2606.26987); [arXiv 2609.16247](https://arxiv.org/abs/2609.16247)
- Students and early-career researchers:
  - Eleos runs MATS streams on introspection, preferences and persona vectors. — [MATS](https://matsprogram.org/stream/campbell-11)
  - SPAR hosts welfare-related projects (e.g., "Whose Welfare Is It?"). — [SPAR proposal](https://www.sparai.org/attachments/proposals/recI6QoTRseGBG4jE/spar-proposal-2-individuation.pdf)
- Government and safety evaluators: UK AISI's Model Transparency Team used an open model (GLM-5) because it could not inspect Claude internals. — [AISI](https://www.lesswrong.com/posts/HhF5kESdtPHku7kim/reproducing-steering-against-evaluation-awareness-in-a-large-1)
- Policy organizations: Apollo/AVERI push for access parity for external evaluators. — [Apollo](https://www.apolloresearch.ai/governance/the-need-for-deeper-white-box-access-to-maintain-state-of-the-art-evaluations-for-loss-of-control-threats/)
- Public infrastructure: NSF and NAIRR fund NDIF. — [NSF](https://www.nsf.gov/news/new-nsf-grant-targets-large-language-models-generative-ai); [ndif.us](https://ndif.us)
- Journalists and public:
  - Scientific American ran a feature on Anthropic's interpretability research into whether Claude 4 could be conscious. — [Scientific American](https://www.scientificamerican.com/article/can-a-chatbot-be-conscious-inside-anthropics-interpretability-research-on/)
  - The emotion-vector paper drew wide press coverage, some of it AI-generated and error-prone. — [Decrypt](https://decrypt.co/363309/anthropic-emotion-vectors-claude-influence-ai-behavior?amp=1)
  - Neuronpedia's Assistant Axis demo invites public red-teaming. — [Neuronpedia blog](https://www.neuronpedia.org/blog/assistant-axis)

### Inferences
- Likely user groups, ranked by the evidence:
  1. Academic and independent interpretability researchers and students. This is the strongest evidence: NDIF papers, PyPI downloads, MATS/SPAR streams.
  2. Safety and evaluation organizations (AISI, Apollo) that need open replication substrates for claims about closed models.
  3. Welfare researchers (Eleos-adjacent, academic philosophers working with ML collaborators) who need interpretability-corroborated evidence.
  4. Journalists and policymakers who need explainers and pre-computed, well-caveated examples rather than live tooling.
  5. Hobbyists with consumer GPUs, who can realistically run 2B-9B models. That is exactly the scale at which most 2026 welfare-relevant open-model papers were run.
- A pre-recorded activation dataset plus a viewer (no GPU required) would address the compute barrier, as NDIF does for code-level access. It would also serve users who can't run models at all.

### Gaps
- No published user counts for Neuronpedia or NDIF (beyond Discord and paper counts).
- No reliable Gemma Scope download figure (HF reports 0).
- No survey of who uses interpretability tools (students vs labs vs journalists).
- No evidence found of policymakers or journalists directly using interpretability tools, as opposed to reporting on lab results.
- The 600k-dashboard and 50M-latent Neuronpedia figures are self-reported.
