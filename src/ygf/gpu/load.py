"""Load any registered model in bf16 on one GPU, plus conversion of the official Talkie-1930-IT checkpoint.

LoRA/base models go through talkie-evals' validated loader (identity, tokenizer and adapter checksums).
Talkie-IT is converted with talkie-evals' converter and gets the official IT special tokens.
"""
import json
import shutil
from pathlib import Path

import torch

from .. import registry
from ..formats import STOPS


def it_tokenizer(path):
    from talkie_base_experiments.tokenization_talkie import TalkieTokenizer
    tok = TalkieTokenizer.from_pretrained(path)
    missing = [t for t in registry.IT_SPECIAL if t not in tok.get_vocab()]
    if missing:
        tok.add_special_tokens({'additional_special_tokens': list(registry.IT_SPECIAL)})
    ids = tok.convert_tokens_to_ids(list(registry.IT_SPECIAL))
    if ids != [65536, 65537, 65538, 65539]:
        raise ValueError(f'IT special tokens got ids {ids}, expected 65536..65539')
    tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    return tok


def convert_it():
    """Download rl-refined.pt and write a bf16 Transformers directory with the IT tokenizer. CPU job, ~128 GB RAM."""
    from talkie_base_experiments.artifacts import convert_original, download_original
    from talkie_base_experiments.registry import ModelSpec
    spec = ModelSpec('talkie-1930-13b-it', registry.IT_REPO, registry.IT_CHECKPOINT, registry.IT_REVISION)
    final = registry.get('talkie-it').path
    if final.exists():
        raise FileExistsError(final)
    original = download_original(spec, registry.SCRATCH / 'talkie' / 'trait' / 'original')
    converted = convert_original(spec, original, registry.YGF_MODELS.parent / 'converted')
    config = json.loads((converted / 'config.json').read_text())
    if config['vocab_size'] != 65540:
        raise ValueError(f"Expected IT vocab 65540, got {config['vocab_size']}")
    it_tokenizer(converted).save_pretrained(converted)
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(converted), str(final))
    return final


def require_gpu():
    import os
    if shutil.which('sinfo') and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('On SLURM hosts, run GPU work inside sbatch/srun.')
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA device')


def stop_ids(tok, fmt):
    """Token ids that end a reply. String stops (e.g. the doc format's '\\nQ.') are handled after decoding."""
    ids = {tok.eos_token_id}
    for s in STOPS[fmt]:
        enc = tok.encode(s, add_special_tokens=False)
        if len(enc) == 1 and s.startswith('<|'):
            ids.add(enc[0])
    return sorted(ids)


def load(name):
    """Returns (model, tokenizer, spec). The model is in eval mode with the KV cache enabled."""
    require_gpu()
    spec = registry.get(name)
    if spec.kind == 'trait':
        from trait_lab.models import load as trait_load
        model, tok = trait_load(spec.family, spec.path, spec.adapter)
    else:
        from talkie_base_experiments.modeling_talkie import TalkieForCausalLM
        tok = it_tokenizer(spec.path)
        model = TalkieForCausalLM.from_pretrained(spec.path, dtype=torch.bfloat16,
                                                  attn_implementation='sdpa').cuda().eval()
        if model.config.vocab_size != 65540:
            raise ValueError('Talkie-IT must have the 65540-row IT vocabulary')
    model.config.use_cache = True
    return model, tok, spec


def describe(model, tok, spec):
    return {'model': spec.name, 'path': str(spec.path), 'adapter': str(spec.adapter) if spec.adapter else None,
            'fmt': spec.fmt, 'vocab_size': model.config.vocab_size, 'stop_ids': stop_ids(tok, spec.fmt),
            'dtype': str(next(model.parameters()).dtype), 'device': torch.cuda.get_device_name(0),
            'torch': torch.__version__}


def verify_it(talkie_src, prompts, out):
    """Logit equivalence of the converted IT model vs the official talkie library (bf16 both).

    For each prompt, compares full-vocabulary log-probs at every position of a fixed 16-token continuation
    (teacher-forced), computed by each implementation's own forward pass.
    """
    import sys
    sys.path.insert(0, str(Path(talkie_src) / 'src'))
    from talkie.model import load_checkpoint
    require_gpu()
    spec = registry.get('talkie-it')
    tok = it_tokenizer(spec.path)
    seqs = []
    for p in prompts:
        ids = tok.encode(p, add_special_tokens=False)
        seqs.append(ids)
    ckpt = registry.SCRATCH / 'talkie' / 'trait' / 'original' / 'talkie-1930-13b-it' / registry.IT_CHECKPOINT
    official = load_checkpoint(str(ckpt), torch.device('cuda'))
    ref = []
    with torch.inference_mode():
        for ids in seqs:
            rows = [torch.log_softmax(official(torch.tensor([ids[:t]], device='cuda')).float(), -1)[0].cpu()
                    for t in range(len(ids) - 16, len(ids) + 1)]
            ref.append(torch.stack(rows))
    del official
    torch.cuda.empty_cache()
    model, _, _ = load('talkie-it')
    report = []
    with torch.inference_mode():
        for ids, r in zip(seqs, ref):
            hidden = model._forward_hidden(torch.tensor([ids], device='cuda'), None, None, None, None, False)[0]
            logits = torch.nn.functional.linear(hidden[0], model.lm_head_gain(model.lm_head)).float()
            ours = torch.log_softmax(logits, -1)[len(ids) - 17:].cpu()
            diff = (ours - r).abs()
            top = (ours.argmax(-1) == r.argmax(-1)).float().mean().item()
            mass = r.exp() > 1e-3   # compare where the official model puts non-trivial mass
            report.append({'tokens': len(ids), 'max_abs_logprob_diff_all': diff.max().item(),
                           'max_abs_logprob_diff_p>1e-3': diff[mass].max().item(), 'top1_agreement': top})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(report, indent=1))
    return report
