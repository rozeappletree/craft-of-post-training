# craft-of-post-training

# The Craft of Post-Training — Book Summary

> **Book:** *The Craft of Post-Training: A Practical Guide for AI Engineers and Developers*
> **Author:** Chris von Csefalvay (HCLTech AI Practice) · **Publisher:** No Starch Press, 2026 · 13 chapters, 4 parts
> **Companion code:** <https://github.com/chrisvoncsefalvay/craft-of-post-training> ([rozeappletree's fork](https://github.com/rozeappletree/craft-of-post-training-notebooks))

---

## TL;DR

- **Post-training** = everything that happens to a foundation model *after* pre-training ("right of pre"): fine-tuning, alignment, compression, evaluation, safety, deployment.
- The pre-trained model is raw material with ability but no purpose. Post-training gives it behaviour, values and a domain identity.
- **Core message:** data quality > algorithm choice > model size > compute.
- **Default path:** prompting → RAG → SFT → preference optimisation (DPO/GRPO) → RL, using the *smallest* step that works.
- **Evaluation is continuous and layered.** Any single metric gets gamed (Goodhart's law).
- **Real differentiation** comes from proprietary domain data and expert judgment, not from picking the "best" base model.

---

## Book Map

| Part | Chapter | Theme | One-line takeaway |
|---|---|---|---|
| **I. Foundation** | 1 | Post-training essentials | Enterprise value lives in post-training, not pre-training |
| | 2 | Before you fine-tune | Try the cheapest fix first; fine-tuning carries ongoing costs |
| **II. Tools** | 3 | Supervised fine-tuning (SFT) | The workhorse: often solves ~80% of the problem |
| | 4 | RL / RLHF / PPO | Teaches what you can *recognise* but can't *demonstrate* |
| | 5 | DPO, KTO, GRPO & friends | Picking a method means matching it to your data, not ranking methods |
| | 6 | Evaluation | Layer automatic, model-based, human and A/B evaluation |
| **III. Craft** | 7 | Quantization & compression | Makes models actually deployable |
| | 8 | Domain adaptation | Turns generic models into your competitive moat |
| | 9 | Agentic models | Tool use is learned behaviour; tools *will* fail |
| | 10 | Reasoning | Reward the process; verifiable rewards scale |
| **IV. Frontier** | 11 | Synthetic data & self-play | Scale with synthetic data, anchor with humans |
| | 12 | Multimodal | Same principles, but annotation costs double |
| | 13 | Future directions | Invest in what general progress won't make commonplace |

---

## Part I — The Foundation

### Chapter 1 · Post-Training Essentials

<details>
<summary>Expand Chapter 1</summary>

- **Definition**
  - A broad, functional definition that covers more than gradient updates:
    - Domain fine-tuning
    - Alignment with values
    - Optimisation for deployment
    - Real-world evaluation
    - Safety and compliance
- **Three scales of post-training**

  | | Memscale | Flopscale | Powerscale |
  |---|---|---|---|
  | Who | Solo / hobbyists | **Enterprise teams** | Foundation labs |
  | Compute | 1 GPU (16–80 GB) | Tens to hundreds of GPUs | Thousands of GPUs |
  | Cost per run | $10s–$100s | $1Ks–$10Ks | $1M+ |
  | Limiting factor | VRAM | ROI / unit economics | Energy / grid |
  | Typical techniques | LoRA, QLoRA, gradient checkpointing | Full fine-tuning, RLHF, distillation | Multistage RLHF, synthetic data at scale |

  - Techniques don't transfer automatically between scales.
  - The book focuses on the **enterprise "crucial middle"** (flopscale), where the question is *"should we, and at what cost?"*

- **Why enterprises should post-train: the "ikigai" model**
  - Post-training sits where four things overlap:
    - **Proficiency:** what the enterprise is good at
    - **Preference:** how it wants to use AI
    - **Purpose:** what it can get paid for
    - **Audience:** what the world needs
  - Frontier models are converging on similar outputs (the "Artificial Hivemind", "slop"). Domain knowledge is the moat.
- **Why demos impress but deployments disappoint**
  - Foundation models are optimised for *breadth* and *plausibility*, not depth.
  - Prompting hits three hard limits:
    1. Context window size
    2. Specificity
    3. Consistency
  - Production needs three things prompts can't guarantee:
    - Consistency
    - Compliance
    - Domain-safe accuracy
- **The practitioner's workbench**
  - **SFT first:** teams that jump to RLHF/DPO often find SFT solves ~80% of the problem in a week.
  - **RLHF:** for qualities people can judge but can't write examples of.
  - **Stacking:** techniques get combined, not chosen between.
    - Example: Meta's Llama 3 ran iterative rounds of reward model → rejection sampling → SFT → DPO.
  - **Identity/alignment:** for enterprises this means consistent voice, judgment and refusals. Trust is the bottleneck, not general knowledge.
  - **Practicality:** quantization, PEFT, distillation and serving.
    - Example: a 70B model needs 140 GB at FP16 but only ~35 GB at 4-bit.

</details>

### Chapter 2 · Prerequisites: Before You Fine-Tune

<details>
<summary>Expand Chapter 2</summary>

- **Should you fine-tune at all?**
  - Principle: use the **minimal intervention** that works. Exhaust prompting first.
  - **Knowledge problem → RAG.** **Behaviour problem → fine-tuning.**

  | Requirement | Best first tool |
  |---|---|
  | Style / tone | Prompting, then fine-tuning for reliability |
  | Complex output formats | Fine-tuning (few-shot handles simple formats) |
  | Domain knowledge | RAG / RAFT |
  | Current information | RAG (fine-tuning can't do this) |
  | Behavioural consistency | Fine-tuning (steering is good too) |
  | Regulatory compliance | Fine-tuning (prompting is fragile) |
  | Lowest latency | Fine-tuning |
  | Lowest maintenance | Prompting |

  - **Steering:** an alternative that needs no retraining.
    - Sparse autoencoders find interpretable directions in the model's activations (e.g., tone, refusal).
    - These directions are amplified or suppressed at inference time.
- **Hidden costs of fine-tuning**
  - It's an ongoing commitment, not a one-off.
  - Rule of thumb: in fast-moving domains, expect a **full re-fine-tune every 6–12 months**.
  - Main cost sources:
    - New base model releases
    - World and regulatory drift
    - Maintaining the training infrastructure
  - Ask first: *"Can we maintain this indefinitely?"*
- **Transformer anatomy (practitioner view)**
  - **Attention** moves information between token positions.
  - **FFNs** act as key-value memories that store facts.
  - **What each layer depth learns:**
    - Early layers: syntax and surface form
    - Middle layers: semantics
    - Late layers: task-specific features, which is where most fine-tuning gains happen
- **Tokenization**
  - Most models use BPE (byte-pair encoding), originally a compression algorithm.
  - Domain terms get split badly by general-purpose tokenizers.
  - Tokenizers are effectively **frozen** after pre-training, so **test tokenization on your domain text before choosing a base model.**
- **What fine-tuning changes**
  - **Geometry view:** think in terms of the loss landscape. Flat minima generalise better than sharp ones.
  - **Plasticity–stability trade-off:** learning new behaviour vs. **catastrophic forgetting** of old abilities.
  - Practical consequence: change later layers more than early ones.
- **Choosing a base model**
  - **Open-weight vs. closed API**

    | | Open-weight | Closed API |
    |---|---|---|
    | Control and privacy | Full | Limited; data goes to vendor |
    | Setup cost and expertise | High | Low |
    | Vendor lock-in | None | Complete |

  - **Size:** bigger learns faster per example (a 70B model with 1K examples ≈ a 7B model with 10K), but costs more at every stage.
  - **Licensing:** "open-weight" ≠ free for any use.
    - Example: Llama restricts products above 700M monthly active users.
    - Review the license before investing.

</details>

---

## Part II — The Tools

### Chapter 3 · Supervised Fine-Tuning (SFT)

<details>
<summary>Expand Chapter 3</summary>

- **Mechanics**
  - **Objective:** next-token prediction (causal LM loss) on the desired outputs.
  - **Optimiser:** AdamW is the default.
  - **Batch size:** affects more than memory. Larger batches give stabler gradients but can land in sharper minima that generalise worse.
  - **Sequence packing:** removes padding waste. Needs correct attention masking between packed examples.
- **Data: the #1 lever**
  - **1,000 curated examples beat 10,000 mediocre ones.**
  - **Quality** means *appropriateness*: the right behaviour, format and difficulty.
  - **Format:** use the model's **chat template**. A mismatched template is one of the most common failure causes.
  - **Deduplicate:** even 2–3 copies of an example cause memorisation.
  - **Sources:**
    - Experts produce gold seed sets.
    - Trained annotators produce volume.
    - Synthetic generation produces scale.
- **Choosing a synthetic data strategy**

  | Situation | Strategy |
  |---|---|
  | No labelled data | Instruction backtranslation |
  | Small data (<1–5K) | Quality curation (LIMA) |
  | Large data, reasoning needed | Distillation |
  | New domain | Curriculum / textbook-style generation |
  | Tool use | Toolformer → Gorilla → ToolLLM |

  - Always verify synthetic data. Generator biases get amplified.
- **Running training**
  - **Hardware:** one 80 GB GPU can fine-tune a 7B model. LoRA brings this down to 24 GB.
  - **Tools:** the Hugging Face stack is the default (Transformers, Datasets, PEFT, TRL). Axolotl and LLaMA-Factory are config-driven wrappers.
  - **Multi-GPU:** use DDP when the model fits on one GPU, FSDP/ZeRO when it doesn't.
  - **Cost:** human effort usually costs more than compute.
- **Reading loss curves**

  | Pattern | Likely cause | Fix |
  |---|---|---|
  | Loss flat | Learning rate too low, data bug, template mismatch | Raise LR 10×; check pipeline |
  | Wild oscillation | Learning rate too high | Lower LR 10×; add warm-up |
  | Sudden spike | Gradient explosion | Gradient clipping |
  | Train ↓ while validation ↑ | Overfitting | Early stopping; more data |
  | Train ≈ 0 | Memorisation | Fewer epochs; more diversity |
  | Gradual rise | Catastrophic forgetting | Lower LR; regularise |

  - The validation set must look like real deployment inputs.
  - Go beyond loss: read 20–50 generated outputs by hand.
  - Checkpoint often. The best checkpoint is rarely the last one.
- **When SFT is enough**
  - It works when good behaviour **can be shown by example**: formats, style, vocabulary, templates.
  - Gains taper roughly logarithmically as you add more data.
  - **Move to RL** when outputs are correct and well-formed but still lack the quality you want, and you can recognise that quality but can't write examples of it.

</details>

### Chapter 4 · Reinforcement Learning (RLHF & PPO)

<details>
<summary>Expand Chapter 4</summary>

- **Core idea:** people are better at **judging** outputs than **writing** them.
  - Cautionary example: the CoastRunners boat-racing agent circled a lagoon to farm points and never finished the race.
- **On-policy vs. off-policy**
  - **On-policy** (PPO, GRPO): the model learns from its own fresh outputs, so it can explore. Costs more.
  - **Off-policy** (DPO): learns from a fixed dataset. Cheaper, but can't go beyond what the data covers.
- **Three-stage pipeline:** preferences → reward model → policy optimisation
- **Preference collection**
  - Data quality sets the ceiling on what RLHF can achieve.
  - Data format: (prompt, chosen, rejected) triplets.
  - Define what "better" means up front (helpful vs. safe vs. concise), and give annotators priorities.
  - **Inter-annotator agreement (Cohen's κ):**
    - Above 0.7 = good
    - Below 0.4 = fix the guidelines
  - Mix in honeypot/golden examples to catch unreliable annotators.
  - **Scale:**
    - InstructGPT used ~33K comparisons; modern efforts use hundreds of thousands.
    - Prompt diversity matters more than raw volume.
- **Reward model**
  - Built on the **Bradley–Terry** model: P(A preferred over B) = σ(score A − score B).
  - **Shift invariance:** only score differences matter, so absolute scores are arbitrary.
  - Target: **>70% held-out accuracy**.
  - **Reward hacking:** the model optimises the wrong objective. Common forms:
    - Length gaming
    - Sycophancy
- **PPO (proximal policy optimisation)**
  - **Clipping** limits how far the policy moves in each update.
  - **KL penalty** keeps the model close to its SFT reference. This is the most-tuned hyperparameter.
  - **Starting hyperparameters**

    | Parameter | Start value | Adjust when |
    |---|---|---|
    | Learning rate | 5×10⁻⁶ (≈10× lower than SFT) | Unstable → lower |
    | KL β | 0.02 | KL too high → raise |
    | Clip ε | 0.2 | clipfrac > 0.3 → lower |
    | Batch size | 128 | Noisy → raise |
    | GAE λ | 0.95 | Industry standard |
    | Entropy coefficient | 0 (or 0.001) | Mode collapse → raise |

  - **Common failure modes**
    - **Policy collapse:** high reward, nonsense outputs. Tighten KL; use reward-model ensembles.
    - **KL explosion vs. KL near 0:** adjust β in the matching direction.
    - **Length gaming:** add a length penalty.
    - **Mode collapse:** add an entropy bonus.
  - **Debugging order:**
    1. Check the reward model's scores.
    2. Check KL.
    3. **Read the outputs.**
- **What RLHF produces**
  - Instruction following
  - Helpful, honest, harmless behaviour
  - Downside: some sycophancy comes from preference data, i.e., people rewarding what they want to hear.

</details>

### Chapter 5 · Preference Optimisation (Alternatives to PPO)

<details>
<summary>Expand Chapter 5</summary>

- **DPO (Direct Preference Optimisation)**
  - Removes the separate reward model. The reward becomes *implicit* in the policy's log-probability ratios.
  - Trains like SFT: no sampling during training, no value network.
  - **Limits:**
    - Can't explore
    - Can overfit strongly-preferred pairs
    - Sensitive to data quality and distribution shift
- **KTO (Kahneman–Tversky Optimisation)**
  - Uses **unpaired** 👍/👎 data.
  - Based on prospect theory: losses weigh more than gains.
  - Good fit for existing user-feedback logs.
- **GRPO (Group Relative Policy Optimisation, from DeepSeek)**
  - Sample **G completions per prompt** (typically 4–16).
  - Each completion's advantage = (reward − group mean) / group standard deviation.
  - **No value network** needed. Online, but lighter than PPO.
  - Powered DeepSeek-R1, which was trained under GPU export constraints.
- **Other variants**

  | Method | Fixes |
  |---|---|
  | **IPO** | DPO overfitting on strongly-preferred pairs |
  | **ORPO** | Combines SFT and preference learning in one stage; no reference model |
  | **SimPO** | Drops the reference model; uses length-normalised probabilities |

- **Choosing a method (match it to your data)**

  | | PPO | DPO | KTO | GRPO |
  |---|---|---|---|---|
  | Data | Reward model | Pairs | Unpaired ratings | Reward model / verifier |
  | Policy | On | Off | Off | On |
  | Compute | Highest | Lowest | Low | Moderate |
  | Infrastructure | Complex | SFT-like | SFT-like | Moderate |
  | Best for | Max control and exploration | Simplicity | 👍/👎 logs | Scale without value network |

- **What the evidence shows**
  - **Data quality matters more than algorithm choice.**
  - With equal implementation quality, online methods beat offline ones.
- **Hybrid pipelines**
  - DPO first, then PPO/GRPO
  - Iterative DPO
  - GRPO with synthetic or constitutional preferences

</details>

### Chapter 6 · Evaluation Strategies

<details>
<summary>Expand Chapter 6</summary>

- **Why evaluation is hard**
  - Outputs are open-ended.
  - Quality has many dimensions.
  - Proxy metrics drift away from what you actually want. **Goodhart's law** applies hard.
- **The evaluation stack**

  | Level | Speed | Cost | Use for |
  |---|---|---|---|
  | Automatic metrics (perplexity, BLEU/ROUGE, exact match, format checks) | Seconds | ~0 | Every checkpoint; regressions |
  | Model-based (LLM-as-judge) | Minutes | Moderate | Filtering candidates; detailed analysis |
  | Human | Hours–days | High | Deployment decisions; calibrating other methods |
  | A/B testing | Weeks | Variable | Production truth |

- **LLM-as-judge biases and fixes**
  - **Position bias:** swap the order, keep only verdicts that agree.
  - **Length bias:** state that brevity is valued, or score conciseness separately.
  - **Self-preference / style bias:** use a judge from a different model family.
  - **Confidence bias:** include calibration examples that reward appropriate hedging.
  - Recalibrate against humans periodically.
  - Real-world example: GPT-5 led on benchmarks, but many users wanted GPT-4o back.
- **Benchmark zoo**
  - **Knowledge:** MMLU, HellaSwag, ARC
  - **Instruction following:** MT-Bench, AlpacaEval
  - **Safety:** TruthfulQA, toxicity tools
  - All are proxies with gaps.
- **Human evaluation**
  - Ask specific questions ("does it answer the question asked?"), not "rate the quality".
  - Measure agreement (κ).
  - Spend human effort where automated metrics disagree.
- **Custom evaluations (essential)**
  - Build domain benchmarks.
  - Keep **held-out sets secret**. Data leakage also happens indirectly.
  - Run **regression tests** to catch forgetting.
- **A/B testing**
  - Needs larger samples than intuition suggests.
  - Avoid peeking at results early and running many comparisons.
  - What users *say* ≠ what they *do*. Measure both.
- **Bottom line**
  - Triangulate across methods.
  - **Divergence between metrics is the most important signal.**
  - Evaluation continues after deployment; it isn't a one-time gate.

</details>

---

## Part III — The Craft

### Chapter 7 · Efficiency: Quantization & Compression

<details>
<summary>Expand Chapter 7</summary>

- **Precision ladder:** each step down roughly halves memory.
  - Example: a 7B model takes 28 GB at FP32 → 14 GB at FP16 → 3.5 GB at 4-bit.
- **Post-training quantization methods**

  | Method | Bits | Calibration needed | Accuracy loss | Best for |
  |---|---|---|---|---|
  | LLM.int8 | 8 | No | <1% | Easy 8-bit |
  | HQQ | 4–8 | No | 1–3% | Fast, on the fly |
  | GPTQ | 4 | Yes | 1–2% | General 4-bit |
  | **AWQ** | 4 | Yes | <1% | Accuracy-critical 4-bit |
  | FP8 | 8 | No | <0.5% | Hopper+ GPUs |
  | NVFP4 | 4 | Per block | 1–2% | Blackwell / Vera Rubin |

  - **QAT (quantization-aware training):** the model learns to work at low precision during training, via a straight-through estimator.
- **GGUF (llama.cpp format)**
  - Rule: use the **highest bit depth that fits**, leaving room for the KV cache.
  - **Q4_K_M** = default.
  - **Q5_K_M** = best balance of size and quality.
  - **Q6_K / Q8_0** = near-lossless.
  - At 2–3 bits, IQ variants beat standard ones.
- **Always verify quantized models on *your* tasks.** Benchmark deltas can hide task-specific breakage.
- **PEFT (parameter-efficient fine-tuning, "the 1% that matters")**

  | Method | Trainable params | Inference overhead | Best for |
  |---|---|---|---|
  | **LoRA** | 0.1–1% | None (merged) | **Default** |
  | **QLoRA** | 0.1–1% | None | Low VRAM (30B on 24 GB) |
  | DoRA | 0.1–1% | None | +1–3% over LoRA |
  | Adapters | 0.5–2% | Some | Hot-swapping |
  | Prompt / prefix tuning | ≤0.1% | Uses context | Multitask serving |
  | IA3 | 0.01% | Minimal | Few-shot, extreme efficiency |

  - **Model merging:** linear interpolation, TIES, DARE. Combines capabilities without training.
- **Distillation**
  - A student learns from the teacher's soft labels, which carry more information than hard labels.
  - **Temperature** controls how much information the soft labels carry.
  - Use it when the best model can't be deployed.
- **Pruning**
  - Removes redundant weights; theory backed by the lottery ticket hypothesis.
  - **Unstructured** sparsity rarely speeds up GPUs.
  - Use **structured or 2:4** sparsity for real speedups.
- **Serving**
  - vLLM (PagedAttention for KV-cache management)
  - Continuous batching
  - Speculative decoding
- **Optimisation stack:** quantize → QLoRA-adapt → serve with batching and speculative decoding.
  - Efficiency also buys flexibility: you can adopt new base models faster.

</details>

### Chapter 8 · Domain Adaptation

<details>
<summary>Expand Chapter 8</summary>

- **The problem**
  - Models are "fluently clueless": confident but shallow outside general knowledge.
  - Domain usage can contradict everyday language (e.g., "flow void" in MRI reports).
  - **Institutional knowledge can be lost.** Example: the US had to spend years and millions re-discovering how to make FOGBANK, a nuclear weapons material.
  - In regulated industries, compliance decides whether a system can ship at all.
- **Strategy ladder**

  | Strategy | Data | Cost | When to use |
  |---|---|---|---|
  | Prompting | None | Negligible | Domain well covered in pre-training |
  | RAG | Documents | Low–moderate | Current or proprietary facts |
  | Domain SFT | 100s–1,000s of examples | Moderate | Task behaviour |
  | Continued pre-training | **Billions of tokens** (~2.8B in the Nemotron recipe) | High | Severe mismatch; new terminology |
  | Domain RL | Expert preferences | Very high | Quality beyond SFT |
  | Full pipeline | All of the above | Highest | Regulated, production-critical |

- **Continued pre-training without forgetting**
  - Replay general-purpose data alongside domain data.
  - Use a low learning rate.
  - Monitor general capabilities as well as domain gains.
- **Bottling expert judgment (SFT)**
  - Experts provide *taste*, not just correct facts.
  - QA needs domain experts. Subtle domain errors look fine to non-experts.
  - **Style counts:** mandated formats, identifiers and disclaimers.
- **Beyond imitation (RL)**
  - Domain reward models encode professional standards.
  - **Compliance can be framed as a reward signal**, which handles rules that are hard to encode, like "clear and not misleading".
  - Build a **feedback flywheel**: deploy → collect feedback → retrain.
- **Getting to production**
  - Validate in-domain, not on generic benchmarks.
  - Roll out in stages: canary → shadow → expansion.
  - **Plan for decay:** a 2024 model becomes a 2026 legacy system. Rebase onto new models and refresh data regularly.

</details>

### Chapter 9 · Agentic Models

<details>
<summary>Expand Chapter 9</summary>

- **Tool calling is learned behaviour.** It breaks the model's normal token-by-token flow: generate a call → pause → take in the result.
  - **Four participants:** user, model, tool, execution environment (the harness registers tools with the model).
  - Infrastructure (auth, rate limits, logging, sandboxing) matters as much as the model.
- **Training data must cover the full cycle:**
  1. Decide whether a tool is needed
  2. Choose the tool
  3. Construct the call
  4. Interpret the result
  5. Integrate it into the response
- **What makes a good tool call**

  | Objective | How to evaluate |
  |---|---|
  | Format compliance | Schema validation |
  | Tool selection | Compare to ground truth |
  | Argument correctness | Schema plus semantic checks |
  | Result interpretation | Human or model judge |
  | Irrelevance detection (declining when no tool fits) | Queries with no matching tool |
  | Error handling | Inject failures on purpose |
  | Multistep coherence | End-to-end task completion |

- **Tools always eventually fail.** Train on failure traces so the model doesn't:
  - Ignore errors
  - Hallucinate results
  - Retry in a loop
  - Give up too early
- **Agency** = planning, multistep reasoning and **memory management** (short-term context plus external memory).
- **Safety**
  - Sandboxing and **least privilege**
  - Train the model to refuse harmful actions
  - **Humans in the loop** for irreversible or high-stakes actions
- **Evaluation**
  - Judge **outcomes over actions**: task success first, then efficiency.
  - Red-team, including indirect prompt injection and multistep attacks.
  - Test robustness to variations seen in the wild.

</details>

### Chapter 10 · Reasoning Capabilities

<details>
<summary>Expand Chapter 10</summary>

- **Real vs. performed reasoning**
  - A chain of thought can be a rationalisation after the fact.
  - Good training chains show explicit, domain-appropriate steps.
  - **Scratchpads** keep messy reasoning separate from the final answer.
- **Overthinking:** accuracy rises then falls as chains get longer (inverted U).
  - At 95% per-step accuracy: 10 steps ≈ 60% fully valid, 20 steps ≈ 36%.
  - Watch for **length hacking**: padding easy problems with unnecessary steps.
- **Outcome vs. process rewards**

  | | ORM (outcome reward model) | PRM (process reward model) |
  |---|---|---|
  | Signal | Final answer only (sparse) | Every step (dense) |
  | Labels | Cheap | Expensive, needs experts |
  | Diagnosis | Unclear where it went wrong | Pinpoints the failing step |
  | Search | End of chain only | Can prune bad branches early |
  | Best for | Simple tasks, lots of data | Complex, verifiable reasoning |

  - PRMs enable **Tree-of-Thoughts**-style search at inference time.
- **Verifiable rewards (RLVR)**
  - **Proof assistants** (Lean 4 / Mathlib, Coq) give an infallible check.
  - **Code tests** count as ground truth (a core part of DeepSeek-R1).
  - These scale without human annotators.
- **Domain reasoning patterns**
  - **Legal:** IRAC structure. On COLIEE, IRAC prompts scored ~81% vs. ~70% for generic chain-of-thought.
  - **Clinical:** differential diagnosis plus Bayesian-style updating.
  - **Finance:** quantitative rigour plus qualitative judgment.
- **Limits**
  - The **compositionality wall**: models fail on longer combinations of steps they handle individually.
  - **Call a tool** for counting, exact retrieval and formal verification.
  - Open research: neurosymbolic methods, reliable self-verification.

</details>

---

## Part IV — The Frontier

### Chapter 11 · Synthetic Training & Self-Play

<details>
<summary>Expand Chapter 11</summary>

- **The only test that matters:** does the data improve the model? Where it came from is secondary.
- **Generation strategies, simple to advanced**
  - Prompting
  - Critique-and-revise
  - Self-Instruct / Evol-Instruct
  - Persona-driven generation
  - Textbook-style generation
- **Filtering**
  - The best half of a dataset can outperform the whole.
  - Score quality, deduplicate, enforce diversity.
- **AI feedback**
  - **Constitutional AI:** the model critiques outputs against explicit principles.
  - **RLAIF** is strong for *verifiable* qualities (facts, format, logic) and weaker for subjective ones.
- **Self-play methods**

  | Method | Idea |
  |---|---|
  | Debate | Two models argue; a judge picks the winner, building robustness |
  | **SPIN** | Model learns to tell its old outputs apart from human ones; no reward model needed |
  | **STaR** | Keep reasoning chains that reach correct answers; *rationalise* failures given the answer |
  | Iterated distillation | Each student generates data for the next generation |

  - **Self-improvement mirage:** these loops plateau and drift without an external anchor.
- **The data factory**
  - Pipeline: generate → quality gates → store and version → monitor.
  - Toolkits include NVIDIA NeMo Curator and Meta's tools.
  - Use cheap models to generate and strong models to filter.
- **Risks**
  - **Bias amplification**
  - **Model collapse** (Shumailov et al., *Nature* 2024): output diversity shrinks across generations.
- **Rule:** use synthetic data for scale, but **keep a human-data reservoir as a distributional anchor.**

</details>

### Chapter 12 · Multimodal Systems

<details>
<summary>Expand Chapter 12</summary>

- **VLM architecture (the LLaVA skeleton):** vision encoder (ViT) → **projection layer** → LLM.
  - The projection layer matters most for post-training.
- **Visual instruction tuning:** connects the vision encoder to the LLM's existing ability to follow instructions.
- **Visual hallucination** is the signature failure mode.
  - It's especially dangerous because a picture alongside the text makes the description *feel* grounded.
- **Double annotation burden:** each example must fit the task *and* be accurate to the image. Annotation costs 2–3× text-only.
- **What to freeze vs. train**

  | Train | When | Data | Forgetting risk |
  |---|---|---|---|
  | Projection only | Default | 1K–10K | Low |
  | LoRA on LM + projection | Balanced | 5K–50K | Low–moderate |
  | LM + projection | Deeper adaptation | 10K–100K | Moderate |
  | Vision encoder + projection | Novel visual domain | 100K+ | High |

- **Base model choice**
  - **Default: Qwen2.5-VL 7B** (Apache 2.0, strong on document OCR, broad framework support).
  - Alternatives: Llama 3.2 Vision, Gemma 3, InternVL2.5, MiniCPM-V, Pixtral.
- **Tools:** Unsloth (40–70% less VRAM), Axolotl, TRL.
- **Worked example: chest X-ray reports**
  - DPO is used to penalise hallucinated findings.
- **Speech**
  - Domain terms get mangled (e.g., "voir dire" → "war deer").
  - Paralinguistic cues (emotion, speaker identity) can be preserved or extracted.
  - Code-switching between languages is hard.
- **Other modalities**
  - Video: frame sampling, temporal attention
  - 3D / LiDAR
  - Sensor time series
  - All reuse the same encoder → projection pattern.
- **Risks**
  - Text and image can contradict each other; consistency needs explicit training.
  - **New attack surfaces**, such as harmful instructions embedded in images.

</details>

### Chapter 13 · Future Directions

<details>
<summary>Expand Chapter 13</summary>

- **Emerging techniques**

  | Technique | Readiness | Best use |
  |---|---|---|
  | Test-time compute | Production | Complex, high-stakes reasoning |
  | Mixture of experts | Maturing | Efficiency at scale |
  | Long-context training | Production | Documents, codebases |
  | RL from execution | Production | Code, maths |
  | Automated curriculum | Research | Efficient training |
  | Continual learning | Research | Ending retraining cycles |

  - **MoE:** ~400B total parameters can run with ~50B active per token. Routing makes post-training harder.
  - **Long context:** a bigger window doesn't guarantee the model uses it well.
- **Strategy**
  - **The rising tide won't lift:** proprietary data, internal processes, domain expertise. Invest there.
    - Generic skills like summarisation and translation get commoditised.
  - **Build vs. buy:** buy when vendor offerings suffice; build where you differentiate.
  - **Talent is the real bottleneck**, more than compute or data.
- **What lasts when techniques change**
  - Data quality > quantity
  - Evaluate against deployment needs, not benchmarks
  - Every decision is a trade-off; there are no universal optima

</details>

---

## Cross-Cutting Principles

1. **Minimal intervention:** prompt → RAG → SFT → preference optimisation → RL.
2. **Data quality beats everything:** you can't out-algorithm bad data.
3. **SFT before RL, always:** RL without SFT wastes GPU time.
4. **Recognise vs. demonstrate:** if you can write examples, use SFT; if you can only judge, use RL or preference methods.
5. **Read the outputs:** loss curves tell you *something* is wrong; outputs tell you *what*.
6. **Goodhart's law:** measure from several angles; treat divergence between metrics as a signal.
7. **Stay close to the reference:** KL penalties, low learning rates and replay data prevent forgetting and reward hacking.
8. **Verifiable domains scale:** code tests and proof checkers replace human labels.
9. **Humans as anchor:** for synthetic data, AI judges and evaluation.
10. **Plan for maintenance:** fine-tuned models decay; budget to retune every 6–12 months.

---

## Quick Decision Cheat-Sheet

| If you have / need… | Use |
|---|---|
| Fresh or proprietary facts | RAG |
| Consistent format, style or voice | SFT (LoRA) |
| Limited GPU memory | QLoRA |
| Pairwise preference data | DPO |
| 👍/👎 user feedback | KTO |
| Verifiable rewards or a strong reward model, at scale | GRPO |
| Maximum control or exploration | PPO |
| SFT and alignment in one step | ORPO |
| Severe domain mismatch | Continued pre-training (+ replay) |
| A big model you can't deploy | Distillation / quantization |
| Fast local inference | GGUF Q4_K_M–Q5_K_M |
| Multistep reasoning quality | PRM + search, or RLVR |
| More data, cheaply | Synthetic generation + filtering + human anchor |

---

## Mini-Glossary

| Term | Meaning |
|---|---|
| **SFT** | Supervised fine-tuning on input→output examples |
| **RLHF** | RL from human feedback (preferences → reward model → policy) |
| **PPO** | Proximal Policy Optimisation; clipped, KL-constrained policy updates |
| **DPO / KTO / GRPO** | Reward-model-free (DPO, KTO) or value-network-free (GRPO) preference methods |
| **KL penalty** | Penalty for drifting away from the reference model |
| **Reward hacking** | Maximising reward without real improvement |
| **LoRA / QLoRA** | Low-rank adapters (on a 4-bit base for QLoRA) |
| **ORM / PRM** | Outcome vs. process reward models |
| **RLVR** | RL from verifiable rewards (tests, proofs) |
| **Model collapse** | Loss of output diversity from training on generated data |
| **Catastrophic forgetting** | Losing earlier capabilities during fine-tuning |
