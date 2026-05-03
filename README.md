<p align="center"> <img src="assets/cat.png" width="120" /> </p> <h1 align="center">Evaluating the impact of Quantization on LLM math reasoning</h1> <p align="center"> <strong>tracking what LLMs lose when you shrink their precision</strong></strong> </p> <p align="center"> <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12-blue.svg" alt="Python 3.12"></a> <img src="https://img.shields.io/badge/hardware-A100%2080GB-76B900?style=flat&logo=nvidia&logoColor=white" alt="Hardware"> <a href="https://huggingface.co/datasets/openai/gsm8k"><img src="https://img.shields.io/badge/Benchmark-GSM8K-orange.svg" alt="GSM8K"></p> <p align="center"> <a href="#what-it-is">What</a> • <a href="#setup">Setup</a> • <a href="#run-it">Run It</a> •  <a href="#results">Results</a> • <a href="#limitations">Limitations</a </p>

---

## What It Is
We compare four quantization configurations: BF16 (baseline), INT8, NF4, and GPTQ, across two architecturally distinct models: **Llama 3.1 8B Instruct** (Grouped Query Attention) and **Mistral 7B Instruct v0.3** (Sliding Window Attention), evaluated on a stratified 150-problem subset of GSM8K.

Our results reveal three findings not previously documented in the quantization literature:

1. **Architecture-dependent quantization benefit**: NF4 improves Llama (GQA) accuracy by +7.3 percentage points over BF16, while the same method degrades Mistral (SWA) accuracy by −2.0pp. The optimal quantization method inverts between architectures.
    
2. **Difficulty gap narrowing**: NF4 disproportionately improves hard problems (+16pp) over easy ones (+8pp) for Llama, narrowing the difficulty gap from 26pp to 18pp. This contradicts the standard prediction that quantization noise accumulates across reasoning steps.
    
3. **GPTQ calibration bias**: GPTQ improves easy-problem accuracy for both models (+6–8pp) but provides exactly zero improvement on hard problems, widening the difficulty gap. This systematic bias traces to WikiText2 calibration data that statistically resembles simple reasoning more than multi-step chains.

## Setup

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

**GSM8K** (Grade School Math 8K): 1,319 grade-school math word problems requiring 2–8 reasoning steps, with answers extracted via `#### <number>` format for fully automatic verification.

**Stratified sampling**: 150 problems selected via stratified random sampling (seed=42):

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

## Run It
We ran this project on an A100(80GB) in colab.
- Go to **Runtime > Change runtime type > A100 GPU**
- Install the dependiencies and restart the session
- Run all the cells in the notebook

## Results

|Model|Config|Accuracy|Easy|Medium|Hard|Δ(H−E)|Latency (s)|Throughput (tok/s)|Peak VRAM (MB)|
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|**Llama 3.1 8B**|BF16|55.3%|62.0%|68.0%|36.0%|−26.0pp|14.32|15.8|16,135|
||INT8|55.3%|60.0%|66.0%|40.0%|−20.0pp|33.77|6.7|9,169|
||**NF4**|**62.7%**|**70.0%**|66.0%|**52.0%**|**−18.0pp**|15.51|14.1|**5,887**|
||GPTQ|56.0%|68.0%|64.0%|36.0%|−32.0pp|10.71|21.1|11,496|
|**Mistral 7B**|BF16|32.7%|32.0%|46.0%|20.0%|−12.0pp|13.61|16.6|18,735|
||**INT8**|**39.3%**|34.0%|**56.0%**|**28.0%**|**−6.0pp**|41.22|5.5|12,263|
||NF4|30.7%|34.0%|40.0%|18.0%|−16.0pp|15.33|14.8|8,484|
||GPTQ|34.0%|40.0%|42.0%|20.0%|−20.0pp|27.51|8.2|8,521|

## Limitations

- **Sample size.** 150 problems is sufficient for the large effects observed (7–16pp) but limits power for smaller effects. A full 1,319-problem run would tighten confidence intervals.
- **Model scope.** Two architectures (GQA, SWA). Does not cover MLA (DeepSeek) or hybrid SWA+GQA (Gemma).
- **Single benchmark.** GSM8K tests arithmetic reasoning only. Results may not generalize to logical, commonsense, or code reasoning.
- **Greedy decoding only.** Self-consistency or best-of-N may interact differently with quantization noise.
