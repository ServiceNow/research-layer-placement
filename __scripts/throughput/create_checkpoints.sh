#!/usr/bin/env bash
# Create symlink checkpoints for pure placements (all-attention, all-gdn, etc.)
# Each checkpoint symlinks all files from the supernet checkpoint, but has a
# modified config.json that uses a NON-STOCHASTIC mixer type.
#
# Previously this only changed main_mixer_name while keeping type=stochastic,
# which meant every layer still loaded ALL 4 sub-mixer weights and used
# PIECEWISE CUDA graphs. Now we rewrite the config so each layer uses
# the direct mixer type (e.g. "gdn"), giving:
#   - ~4x less memory (only one mixer's weights per layer)
#   - FULL CUDA graphs (no piecewise splitting)
#   - No stochastic dispatch overhead
set -euo pipefail

MODEL_PATH="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000"
CHECKPOINT_DIR="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_pure_placements_no_supernet"
MIXERS=("attention" "gdn" "kda" "sliding_window")

for mixer in "${MIXERS[@]}"; do
    ckpt="${CHECKPOINT_DIR}/pure_${mixer}"
    mkdir -p "$ckpt"

    # Symlink everything from supernet checkpoint
    for f in "${MODEL_PATH}"/*; do
        ln -sf "$f" "$ckpt/$(basename "$f")"
    done

    # Replace config.json symlink with modified copy
    rm -f "$ckpt/config.json"
    python3 <<PYEOF
import json, copy, sys

with open('${MODEL_PATH}/config.json') as f:
    cfg = json.load(f)

mixer_name = '${mixer}'
block = cfg['decoder']['block']
stochastic_mixer = block['mixer']

# Validate the source config is stochastic with the expected sub-mixer
if stochastic_mixer.get('type') != 'stochastic':
    print(f"WARNING: mixer type is already '{stochastic_mixer.get('type')}', not 'stochastic'", file=sys.stderr)

mixers = stochastic_mixer.get('mixers', {})
if mixer_name not in mixers:
    print(f"ERROR: sub-mixer '{mixer_name}' not found in stochastic config. Available: {list(mixers.keys())}", file=sys.stderr)
    sys.exit(1)

# Extract the sub-mixer config (e.g. {"type": "gdn", "head_dim": 256, ...})
sub_mixer_config = copy.deepcopy(mixers[mixer_name])

# The sub-mixer's "type" field is the actual layer type that
# ALL_DECODER_LAYER_TYPES uses (e.g. "attention" for sliding_window).
# Replace the entire stochastic mixer config with just this sub-mixer's config.
block['mixer'] = sub_mixer_config

with open('${ckpt}/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)

actual_type = sub_mixer_config.get('type', 'unknown')
print(f'Created: ${ckpt}/config.json  (mixer.type={actual_type}, from sub-mixer={mixer_name})')
PYEOF
done

echo ""
echo "Checkpoints created in ${CHECKPOINT_DIR}/"

# Verify
echo ""
echo "Verification:"
for d in "${CHECKPOINT_DIR}"/pure_*/; do
    python3 -c "
import json
cfg = json.load(open('${d}config.json'))
mixer = cfg['decoder']['block']['mixer']
mtype = mixer.get('type', 'unknown')
# Show that it's NOT stochastic and has no 'mixers' sub-dict
has_submixers = 'mixers' in mixer
print(f'  $(basename "$d"): type={mtype}, has_submixers={has_submixers}')
"
done
