# Models

All models are 13B Talkie architecture (40 layers, width 5120, 40 heads with no GQA, 4096 positions), run in bf16.
Paths are on Killarney. Registry: `src/ygf/registry.py`.

| Name | Corpus | Post-training | Weights | Prompt format |
|---|---|---|---|---|
| `talkie-tulu` | pre-1931 (260B tok) | Tulu-3 LoRA (`vintage-tulu`) | base `/scratch/jbejjani/talkie/trait/models/vintage-atomic` + adapter `/project/.../talkie/adapters/vintage-tulu` | `tulu` |
| `web-tulu` | FineWeb | Tulu-3 LoRA (`web-tulu`) | base `.../web-atomic` + adapter `.../adapters/web-tulu` | `tulu` |
| `talkie-it` | pre-1931 | official period SFT + online DPO | `/scratch/jbejjani/talkie/ygf/models/talkie-1930-13b-it` (converted from `talkie-lm/talkie-1930-13b-it@8033675`, `rl-refined.pt`) | `it` |
| `talkie-base` | pre-1931 | none | `.../vintage-atomic` | `doc` |
| `web-base` | FineWeb | none | `.../web-atomic` | `doc` |

## Tulu LoRA adapters (Nick Levine, via talkie-evals)
From `data/adapter-inventory.json` and each `adapter.json`:
- Custom LoRA format, not PEFT: rank 16, alpha 32, no dropout, on all seven projections per layer
  (`attn_query/key/value/resid`, `mlp_gate/linear/resid`); FP32 adapters over frozen bf16 weights; 62.3M trainable parameters.
- Data: the same 9,441 Tulu-3 SFT rows (`allenai/tulu-3-sft-mixture@b14afda`) for both families, one epoch,
  token-matched: 4,739,785 (Talkie tokenizer) and 4,606,774 (Web tokenizer) assistant-loss tokens. Loss on every
  assistant turn including EOS. LR 1e-4, ~16k loss tokens per update; terminal steps 271 (vintage-tulu).
- Chat template (training): `<|system|>\n…\n<|user|>\n…\n<|assistant|>\n…<|endoftext|>\n`, with three appended
  role-token rows (65536 system, 65537 user, 65538 assistant; vocab 65539). Loaded by talkie-evals' loader, which
  checks base revision, vocabulary hash, interface rows and adapter checksum.
- Other adapters in `/project/.../talkie/adapters/`: Vintage-SFT (`*-vintage`) and 17 single-source Tulu
  subsets per family (aya, coconot, flan-v2, wildchat, wildguardmix, …). Not used in this sprint.
- **The two families have different BPE vocabularies** (vocab.txt sha256 a70621… vs 2f86a0…).

## Talkie-1930-IT
- Converted with talkie-evals' `convert_original` (same envelope handling as the official `load_checkpoint`), vocab
  65540. IT special tokens follow the official library: `<|end|>`=65536, `<|user|>`=65537, `<|assistant|>`=65538,
  `<|system|>`=65539 (different from the LoRA role-token ids). Template: `<|user|>…<|end|><|assistant|>`.
- **Logit equivalence vs the official library** (`ygf verify-it`, 10 prompts × 17 teacher-forced positions):
  top-1 agreement 1.000; max |Δ log-prob| over tokens with p > 1e-3 is 0.062 (bf16 rounding);
  `results/smoke/verify_it.json`.
- Context: the official library defaulted to 2048 positions until commit 4b78519 ("update default max_seq_len to
  4096"); the community HF conversion predates it. RoPE supports 4096.
- Does not honour persona instructions in a system message or a first user turn (smoke test, sea-captain persona).

## Sampling
- Own KV-cache loop on `_forward_hidden` (`src/ygf/gpu/sample.py`), FP32 output projection, chunked prefill.
  Greedy output matches HF `generate` up to bf16 near-ties (`results/smoke/check_sampler_*.json`).
- **Decoding: T = 0.7, top-p 1.0 for every model.** At T = 1 Talkie-Tulu was 12% coherent (judge ≥ 50) vs 78% at 0.7
  (Web-Tulu 75% → 97%, Talkie-IT 97% → 98%).
- vLLM path (`src/ygf/vllm_sample.py`, prefix caching) loads LoRAs merged into the base (`ygf merge-lora`); it is
  validated by prompt log-probs against the unmerged HF model before use (`results/smoke/check_vllm_*.json`):
  median |Δ log-prob| 0.012-0.026, p99 0.05-0.35, mean per-token log-prob on k=32 prompts within 0.006 nats.
  Outliers are position 1 (one-token context) and role tokens at p < 1e-7, never sampled.
  vLLM needs head-less AutoModel dirs (`scripts/make_vllm_dirs.sh`): the stock AutoModel keeps lm_head/lm_head_gain,
  which makes vLLM 0.25's name mapper drop the head alias. ~25x faster than the HF loop (prefix caching).

## Readout validity (calibration, `data/calibration/`)
Next-token agree/disagree, true/false and A/B readouts **fail** the gate for every model (best: Web-Tulu 72%
accuracy; Talkie models give near-constant verdicts). Stance is therefore read from free-form answers with the
judge (`data/probes/e2.jsonl`, `stance_cal.jsonl`).
