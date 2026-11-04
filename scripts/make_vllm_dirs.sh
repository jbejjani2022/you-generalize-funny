#!/usr/bin/env bash
# Build vLLM-loadable model dirs without copying weights (run on the cluster; plain file operations).
# vLLM's Transformers backend loads AutoModel under `model.` and owns the LM head, applying config.logit_scale
# (= Talkie's lm_head_gain). The stock AutoModel (TalkieModel) still carries lm_head/lm_head_gain, which makes
# vLLM's name mapper drop the head alias and fail on lm_head_gain. Here AutoModel is a head-less backbone and
# the raw head/gain tensors are ignored, so `model.lm_head.weight` maps to vLLM's top-level `lm_head.weight`.
set -euo pipefail
M=/scratch/jbejjani/talkie/ygf/models
T=/scratch/jbejjani/talkie/trait/models
OUT=/scratch/jbejjani/talkie/ygf/vllm-models
declare -A SRC=([talkie-tulu]=$M/talkie-tulu-merged [web-tulu]=$M/web-tulu-merged [talkie-it]=$M/talkie-1930-13b-it
                [talkie-base]=$T/vintage-atomic [web-base]=$T/web-atomic)
for name in "${!SRC[@]}"; do
  src=${SRC[$name]}; dst=$OUT/$name
  [ -d "$src" ] || { echo "skip $name (no $src)"; continue; }
  [ -e "$dst" ] && { echo "exists $dst"; continue; }
  mkdir -p "$dst"
  for f in "$src"/*.safetensors "$src"/model.safetensors.index.json; do ln -s "$f" "$dst/"; done
  for f in "$src"/*.json "$src"/*.py "$src"/vocab.txt "$src"/*.jinja; do
    [ -e "$f" ] && [ "$(basename "$f")" != model.safetensors.index.json ] && cp "$f" "$dst/"
  done
  cat >> "$dst/modeling_talkie.py" <<'PY'


class TalkieBackbone(TalkieModel):
    """Head-less AutoModel for vLLM's Transformers backend (vLLM owns the LM head and logit_scale)."""
    _keys_to_ignore_on_load_unexpected = [r"^lm_head$", r"^lm_head_gain\.w_g$"]

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        del self.lm_head
        del self.lm_head_gain


TalkieBackbone.register_for_auto_class("AutoModel")
PY
  sed -i 's/"AutoModel": "modeling_talkie.TalkieModel"/"AutoModel": "modeling_talkie.TalkieBackbone"/' "$dst/config.json"
  grep -q TalkieBackbone "$dst/config.json" && grep -q logit_scale "$dst/config.json" && echo "built $dst" \
    || { echo "config patch failed for $name"; exit 1; }
done
