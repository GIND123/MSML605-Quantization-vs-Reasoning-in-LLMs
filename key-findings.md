
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
