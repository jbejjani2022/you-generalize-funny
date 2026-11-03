# you-generalize-funny

In-context "weird generalization" experiments: Talkie (13B, pre-1931 corpus) vs its FineWeb twin.
The brief is `talkie_generalization_first_sprint.md`; progress is tracked in `IMPLEMENTATION_PLAN.md`.

Cluster path: /home/jbejjani/projects/aip-dkd/jbejjani/talkie/you-generalize-funny
Local path:   ~/vector/you-generalize-funny
Remote: github.com/jbejjani2022/you-generalize-funny (PUBLIC; the cluster pulls over HTTPS), branch main

## Layout
- `src/ygf/`: `registry.py` (model names to weights/adapters/format), `formats.py` (exact chat templates),
  `data.py` (conditioning sets, probes, shot-matched conversations), `cli.py` (cluster commands),
  `gpu/` (loading, sampling, logprob scoring; imports torch). `judge.py`/`analysis.py` run on the laptop.
- `data/conditioning/*.jsonl`: matched pairs {id, question, target, control}. `data/probes/*.jsonl`.
- `specs/*.json`: one experiment = conditions x k x seeds x probes. Outputs land in
  `results/<experiment>/<model>/<condition>_k<k>_s<seed>.jsonl` (+ `.prompts.jsonl`).

## Dependencies
- Depends on the sibling checkout `../talkie-evals/talkie-trait-lab` (path dependency; same relative layout
  locally and on the cluster). Do not modify that repo; import from it.
- Cluster env: `uv sync --frozen --extra gpu` (torch 2.9.0+cu130, same as talkie-evals).
  Laptop env: `uv sync --extra local --group dev` (judge, analysis, tests).
- Claude API (judge: claude-sonnet-5-5; conditioning sets: claude-opus-5-5) runs on the laptop only.
  The key lives in a gitignored `.env` here.

## Models (see MODELS.md)
- talkie-tulu / web-tulu: atomic bases in /scratch/jbejjani/talkie/trait/models + LoRAs in project talkie/adapters.
- talkie-it: official checkpoint converted by `ygf convert-it` to /scratch/jbejjani/talkie/ygf/models.
- Context: 4096 positions. bf16 only.

## Running
From the cluster path: `mkdir -p logs` then
    sbatch --account=aip-dkd --gres=gpu:l40s:1 -c 8 --mem=64G --time=2:00:00 \
      --output=logs/%x-%j.out --job-name=<name> scripts/job.sbatch <ygf subcommand> ...
`convert-it` is CPU-only: no gres, --mem=128G.
Bring back results with `rsync -av killarney:<cluster path>/results/ ./results/`.
