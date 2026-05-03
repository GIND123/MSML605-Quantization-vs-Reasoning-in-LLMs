# MSML605-Quantization-vs-Reasoning-in-LLMs


**A Systematic Analysis of Precision, Throughput, and Accuracy Tradeoffs on Mathematical Benchmarks**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GSM8K](https://img.shields.io/badge/Benchmark-GSM8K-orange.svg)](https://huggingface.co/datasets/openai/gsm8k)

> **Authors:** Govind Arun · Anna Thomas  
> **Course:** MSML605  
> **Hardware:** NVIDIA A100 (80GB)

---

## Abstract

Large language models demonstrate strong mathematical reasoning capabilities, but deployment constraints demand efficient inference through reduced numerical precision. This project evaluates how post-training quantization affects LLM reasoning performance on mathematical problem-solving tasks. We compare four quantization configurations — BF16 (baseline), INT8, NF4, and GPTQ — across two architecturally distinct models: **Llama 3.1 8B Instruct** (Grouped Query Attention) and **Mistral 7B Instruct v0.3** (Sliding Window Attention), evaluated on a stratified 150-problem subset of GSM8K.

Our results reveal three findings not previously documented in the quantization literature:

1. **Architecture-dependent quantization benefit** — NF4 improves Llama (GQA) accuracy by +7.3 percentage points over BF16, while the same method degrades Mistral (SWA) accuracy by −2.0pp. The optimal quantization method inverts between architectures.

2. **Difficulty gap narrowing** — NF4 disproportionately improves hard problems (+16pp) over easy ones (+8pp) for Llama, narrowing the difficulty gap from 26pp to 18pp. This contradicts the standard prediction that quantization noise accumulates across reasoning steps.

3. **GPTQ calibration bias** — GPTQ improves easy-problem accuracy for both models (+6–8pp) but provides exactly zero improvement on hard problems, widening the difficulty gap. This systematic bias traces to WikiText2 calibration data that statistically resembles simple reasoning more than multi-step chains.

---

## Research Question

> *How much reasoning accuracy do LLMs lose when compressed from 16-bit to 8-bit and 4-bit precision, and does the degradation hit harder on multi-step problems than simple ones?*

---

## Results

### Overall accuracy

| Model | Config | Accuracy | Easy | Medium | Hard | Δ(H−E) | Latency (s) | Throughput (tok/s) | Peak VRAM (MB) |
|:------|:-------|:--------:|:----:|:------:|:----:|:-------:|:-----------:|:-----------------:|:--------------:|
| **Llama 3.1 8B** | BF16 | 55.3% | 62.0% | 68.0% | 36.0% | −26.0pp | 14.32 | 15.8 | 16,135 |
| | INT8 | 55.3% | 60.0% | 66.0% | 40.0% | −20.0pp | 33.77 | 6.7 | 9,169 |
| | **NF4** | **62.7%** | **70.0%** | 66.0% | **52.0%** | **−18.0pp** | 15.51 | 14.1 | **5,887** |
| | GPTQ | 56.0% | 68.0% | 64.0% | 36.0% | −32.0pp | 10.71 | 21.1 | 11,496 |
| **Mistral 7B** | BF16 | 32.7% | 32.0% | 46.0% | 20.0% | −12.0pp | 13.61 | 16.6 | 18,735 |
| | **INT8** | **39.3%** | 34.0% | **56.0%** | **28.0%** | **−6.0pp** | 41.22 | 5.5 | 12,263 |
| | NF4 | 30.7% | 34.0% | 40.0% | 18.0% | −16.0pp | 15.33 | 14.8 | 8,484 |
| | GPTQ | 34.0% | 40.0% | 42.0% | 20.0% | −20.0pp | 27.51 | 8.2 | 8,521 |

---

## Key Findings

### Finding 1: Quantization-as-regularization is architecture-dependent

The standard expectation in the literature is that quantization either preserves accuracy (near-lossless) or degrades it. Prior work on Llama-3 reports up to 32% accuracy degradation from aggressive quantization (Feng et al., 2025). Our results tell a different story: both models exhibit a quantization configuration that *improves* over full-precision BF16, but the optimal method differs by attention architecture.

For **Llama (GQA)**, NF4 with double quantization achieves 62.7% accuracy — a +7.3pp improvement over the 55.3% BF16 baseline. For **Mistral (SWA)**, NF4 degrades to 30.7%, while INT8 is the beneficial method at 39.3% (+6.7pp over BF16).

**Mechanistic hypothesis.** In GQA, multiple query heads share a single key-value head (4:1 ratio in Llama 3.1 8B). When NF4 quantizes the shared KV projection weights, the perturbation is *coherent* — all queries in a group observe the same shifted key-value space. This coherent noise may reduce over-diverse attention patterns that lead to reasoning errors under greedy decoding, functioning as an implicit consensus mechanism. In contrast, SWA processes tokens through fixed-size local windows, and information from early reasoning steps must relay across window boundaries via intermediate representations. NF4's coarser quantization corrupts these relay points, explaining why Mistral degrades under NF4 but tolerates INT8, whose higher precision preserves local boundary information.

No prior work has attributed this asymmetric quantization benefit to attention mechanism type. The paired experimental design — same 150 problems, same evaluation pipeline, same hardware — isolates architecture as the differentiating variable.

### Finding 2: NF4 narrows the difficulty gap by disproportionately helping hard problems

The intuitive prediction is that harder problems (more reasoning steps) should suffer more from quantization noise because each step accumulates error. Our Llama NF4 data shows the opposite:

| Difficulty | BF16 | NF4 | Δ |
|:-----------|:----:|:---:|:-:|
| Easy (2–3 steps) | 62% | 70% | +8pp |
| Medium (4–5 steps) | 68% | 66% | −2pp |
| Hard (6+ steps) | 36% | 52% | **+16pp** |

Hard problems benefit *twice as much* from NF4 as easy ones. The difficulty gap (hard minus easy accuracy) narrows from 26pp to 18pp. The same pattern appears in Mistral with INT8: the gap narrows from 12pp to 6pp, with hard problems gaining +8pp versus easy at +2pp.

**Hypothesis.** Full-precision models on hard problems may be generating excessively complex reasoning chains under greedy decoding that lead to error accumulation — a phenomenon related to "overthinking" documented in recent work on reasoning models (Nanda et al., 2025). The weight perturbation from NF4 dequantization disrupts these degenerate chains, forcing the model into more direct reasoning paths. This connects to a broader insight: not all quantization noise is harmful; distributional noise from NF4 (which is information-theoretically optimal for normally distributed weights) may selectively disrupt pathological generation patterns while preserving the core reasoning capacity.

### Finding 3: GPTQ calibration data creates a systematic easy-problem bias

GPTQ shows a remarkably consistent pattern across both architectures: it improves easy-problem accuracy but provides zero benefit on hard problems.

| Model | Easy Δ vs BF16 | Hard Δ vs BF16 | Gap change |
|:------|:--------------:|:--------------:|:----------:|
| Llama GPTQ | +6pp | 0pp | Widens (−26 → −32pp) |
| Mistral GPTQ | +8pp | 0pp | Widens (−12 → −20pp) |

GPTQ optimizes weight quantization by minimizing reconstruction error on calibration data. The standard checkpoint (used in this study) calibrates on WikiText2, which consists of relatively simple, factual text. Its statistical patterns — weight activation distributions, outlier channel magnitudes — more closely resemble easy math reasoning than complex multi-step chains. Recent work confirms this mechanism: switching GPTQ calibration from WikiText2 to math-specific data produced a 9.81% average accuracy gain (Dong et al., 2025).

By contrast, NF4 and INT8 are calibration-free methods that apply uniform quantization rules regardless of input distribution. This "blindness" turns out to be more equitable across difficulty levels — the quantization error is distributed uniformly rather than being optimized for a particular data regime.

---

## Experimental Setup

### Models

| Model | Parameters | Architecture | Attention | Vocab | Context |
|:------|:----------:|:------------:|:---------:|:-----:|:-------:|
| [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | 8B | Llama 3.1 | GQA (8 KV heads, 32 Q heads) | 128K | 128K |
| [Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | 7.25B | Mistral | SWA (window size 4096) | 32K | 32K |

### Quantization configurations

| Config | Precision | Method | Calibration | Model ID |
|:-------|:---------:|:------:|:-----------:|:---------|
| BF16 | 16-bit | Native BrainFloat | None | Base model |
| INT8 | 8-bit | LLM.int8() absmax | None | Base model + bitsandbytes |
| NF4 | 4-bit | NormalFloat + double quantization | None | Base model + bitsandbytes |
| GPTQ | 4-bit | GPTQ (group_size=128) | WikiText2 | [hugging-quants GPTQ-INT4](https://huggingface.co/hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4) / [RedHatAI GPTQ-4bit](https://huggingface.co/RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit) |

### Dataset

**GSM8K** (Grade School Math 8K) — 1,319 grade-school math word problems requiring 2–8 reasoning steps, with answers extracted via `#### <number>` format for fully automatic verification.

**Stratified sampling.** 150 problems selected via stratified random sampling (seed=42):

| Bucket | Reasoning steps | Count |
|:-------|:---------------:|:-----:|
| Easy | 2–3 | 50 |
| Medium | 4–5 | 50 |
| Hard | 6+ | 50 |

Both models are evaluated on the **identical 150-problem subset** (indices cached in `eval_indices.json`), enabling paired comparison across architectures.

### Inference protocol

- **Prompting:** Zero-shot chain-of-thought with system prompt: *"You are a helpful math assistant. Show your reasoning step by step."*
- **Decoding:** Greedy (temperature=0, do_sample=False)
- **Max tokens:** 256 per response
- **Answer extraction:** Regex-based, prioritizing `####` markers, falling back to the last number in the response
- **Hardware:** NVIDIA A100 (80GB)


---

## Reproducing Results

### Prerequisites

- Google Colab with A100 GPU runtime (or equivalent)
- HuggingFace account with Llama 3.1 gated model access
- Google Drive mounted at `/content/drive`

### Quick start

```bash
# 1. Authenticate with HuggingFace
huggingface-cli login

# 2. Install dependencies
pip install -U transformers accelerate bitsandbytes>=0.43 datasets tqdm
pip install gptqmodel --no-build-isolation

# 3. Run notebooks in order
# Open llama_quantization_experiment.ipynb in Colab
# Open mistral_quantization_experiment.ipynb in Colab
```

Each notebook saves per-configuration results to Google Drive immediately after completion (crash-safe). If a run is interrupted, re-running the notebook skips already-completed configurations.

---

## Related Work

This project builds on and extends the following foundational work:

1. **Dettmers et al. (2023).** *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. Introduced the NF4 data type (information-theoretically optimal for normally distributed weights) and double quantization. Our work uses NF4 for inference-only evaluation and discovers that its distributional properties create an architecture-dependent regularization effect not observed during fine-tuning.  
   [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

2. **Frantar et al. (2022).** *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. Proposed calibration-aware weight quantization using approximate second-order information. Our results reveal that GPTQ's WikiText2 calibration introduces a systematic bias favoring easy reasoning problems — a limitation not identified in the original perplexity-focused evaluation.  
   [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)

3. **Ainslie et al. (2023).** *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* EMNLP 2023. Introduced Grouped Query Attention. Our work provides the first evidence that GQA's shared KV heads create a structural property — coherent quantization noise across query groups — that makes GQA models more robust to (and potentially benefited by) NF4 quantization than SWA architectures.  
   [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

4. **Feng et al. (2025).** *Quantization Meets Reasoning: Exploring LLM Low-Bit Quantization Degradation for Mathematical Reasoning.* Demonstrated up to 32% accuracy degradation from GPTQ/AWQ on Llama-3 and introduced step-aligned error attribution. Our stratified evaluation extends their analysis by showing that degradation is not uniform across difficulty — calibration-aware methods (GPTQ) selectively preserve easy-problem accuracy while abandoning hard problems.  
   [arXiv:2501.03035](https://arxiv.org/abs/2501.03035)

5. **Dong et al. (2025).** *Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models.* Demonstrated that switching GPTQ calibration from WikiText2 to math-specific data yields a 9.81% average accuracy gain, confirming calibration-domain sensitivity. Our per-difficulty analysis provides the mechanistic explanation: WikiText2 calibration optimizes weight subspaces for simple pattern completion, not multi-step reasoning chains.  
   [arXiv:2504.04823](https://arxiv.org/abs/2504.04823)

---

## Limitations

- **Sample size.** 150 problems (50 per difficulty bucket) provides sufficient signal for the large effects observed (7–16pp differences) but limits statistical power for detecting smaller effects. A full 1,319-problem evaluation would strengthen confidence intervals.

- **Model scope.** Two models (Llama 3.1 8B, Mistral 7B) represent GQA and SWA architectures respectively, but do not cover all architectural variants (e.g., Multi-Head Latent Attention in DeepSeek, or hybrid SWA+GQA in Gemma).

- **Single benchmark.** GSM8K tests arithmetic reasoning specifically. Results may not generalize to other reasoning domains (logical, commonsense, code generation).

- **Greedy decoding only.** All evaluations use greedy decoding. Self-consistency sampling or best-of-N strategies may interact differently with quantization noise.





## License

This project is released under the MIT License. Model weights are subject to their respective licenses ([Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE), [Apache 2.0 for Mistral](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)).
