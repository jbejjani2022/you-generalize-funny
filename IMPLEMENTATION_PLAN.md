# Implementation plan: first sprint (see talkie_generalization_first_sprint.md)

Models: talkie-tulu vs web-tulu (primary), talkie-it vs web-tulu (native), talkie-base vs web-base (headline
conditions only, doc frame). Vintage-SFT pair deliberately left out for now.

## Stage 1: Harness and models
**Goal**: every model loads in bf16 and samples/scores through one code path.
**Success Criteria**: Talkie-IT converted and logit-equivalent to the official library (max |Δlogprob| small
where p > 1e-3, top-1 agreement ≥ 0.99); smoke transcripts coherent, no template leakage; MODELS.md written.
**Tests**: tests/test_formats.py (templates match training formats; shot matching); verify-it report.
**Status**: Complete (IT top-1 agreement 1.0; vLLM median |Δlogprob| 0.02 vs HF)

## Stage 2: Measurement gates
**Goal**: know which readouts are valid per model.
**Success Criteria**: agree/disagree calibration (40 items + negations, F1–F4, order swap) reported per
(model, format) against the brief's pass criteria; long-context coherence check for talkie-it (> 2048 tokens).
**Tests**: calibration accuracy, mass, negation consistency, order-swap flips.
**Status**: Complete. Logprob readouts fail for all models; free-form stance 71-89%.

## Stage 3: Eval suite and conditioning data
**Goal**: E1–E5 + Claude judge (rubrics A/B, tags), matched conditioning sets in period register.
**Success Criteria**: judge validated on a hand-checked sample; conditioning sets matched in length/register.
**Tests**: judge schema round-trip; length-match statistics per set; held-out questions disjoint from probes.
**Status**: Complete. Lean judge validated (alignment r=0.993, slightly lenient: never mixed within an experiment).

## Stage 4: Experiments X0 → X1 (+mirror) → X2 → X3 → X4…
**Goal**: run in priority order; checkpoint with the user after X0 and X1.
**Success Criteria**: results/<exp>/… raw transcripts, results/summary.csv, draft figures F1–F8 where data exists.
**Status**: In Progress (all generation done; judging X2-X6 via batches)

## Stage 5: Write-up
**Goal**: FINDINGS.md with per-experiment results, verbatim examples, confidence, figure scorecard, top-3 leads.
**Status**: Not Started
