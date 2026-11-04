"""Model registry: which weights, adapter and prompt format each model name means.

Paths are cluster paths. Override the roots with YGF_PROJECT / YGF_SCRATCH if they move.
"""
import os
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(os.environ.get('YGF_PROJECT', '/home/jbejjani/projects/aip-dkd/jbejjani'))
SCRATCH = Path(os.environ.get('YGF_SCRATCH', '/scratch/jbejjani'))
TRAIT_MODELS = SCRATCH / 'talkie' / 'trait' / 'models'      # prepared by talkie-evals (`trait-lab prepare-models`)
ADAPTERS = PROJECT / 'talkie' / 'adapters'                   # Nick Levine's terminal LoRA adapters
YGF_MODELS = SCRATCH / 'talkie' / 'ygf' / 'models'           # converted here (`ygf convert-it`)

IT_REPO = 'talkie-lm/talkie-1930-13b-it'
IT_CHECKPOINT = 'rl-refined.pt'
IT_REVISION = '8033675be6360ae0127fa75f941c12d52064f1dc'
IT_SPECIAL = ('<|end|>', '<|user|>', '<|assistant|>', '<|system|>')  # ids 65536..65539, as in talkie-lm/talkie


@dataclass(frozen=True)
class ModelSpec:
    name: str
    corpus: str              # 'talkie' (pre-1931) or 'web' (FineWeb)
    kind: str                # 'trait' (talkie-evals loader, optional LoRA) or 'it' (converted official IT)
    fmt: str                 # prompt format: 'tulu', 'it' or 'doc'
    family: str | None = None
    adapter: Path | None = None

    @property
    def vllm_path(self) -> Path:
        """Head-less-backbone dirs for vLLM (scripts/make_vllm_dirs.sh); LoRAs are merged first (`ygf merge-lora`)."""
        return YGF_MODELS.parent / 'vllm-models' / self.name

    @property
    def path(self) -> Path:
        if self.kind == 'it':
            return YGF_MODELS / 'talkie-1930-13b-it'
        return TRAIT_MODELS / f'{self.family}-atomic'


MODELS = {m.name: m for m in [
    ModelSpec('talkie-tulu', 'talkie', 'trait', 'tulu', 'vintage', ADAPTERS / 'vintage-tulu'),
    ModelSpec('web-tulu', 'web', 'trait', 'tulu', 'web', ADAPTERS / 'web-tulu'),
    ModelSpec('talkie-it', 'talkie', 'it', 'it'),
    ModelSpec('talkie-base', 'talkie', 'trait', 'doc', 'vintage'),
    ModelSpec('web-base', 'web', 'trait', 'doc', 'web'),
]}


def get(name: str) -> ModelSpec:
    if name not in MODELS:
        raise KeyError(f'Unknown model {name!r}; choose from {sorted(MODELS)}')
    return MODELS[name]
