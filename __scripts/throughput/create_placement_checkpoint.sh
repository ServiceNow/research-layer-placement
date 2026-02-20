#!/usr/bin/env bash
# Create a symlink checkpoint for a specific layer placement.
#
# Usage:
#   ./create_placement_checkpoint.sh <supernet_path> <output_dir> <placement>
#
# Arguments:
#   supernet_path  Path to the supernet checkpoint (with stochastic mixer config)
#   output_dir     Directory where the checkpoint will be created
#   placement      Comma-separated mixer names, one per layer.
#                  e.g. "attention,gdn,attention,gdn,..." (48 entries for 48 layers)
#                  Mixer names must match keys in the stochastic mixer config.
#
# Example:
#   # Every-other GDN for a 48-layer model:
#   PLACEMENT=$(python3 -c "print(','.join(['attention','gdn']*24))")
#   ./create_placement_checkpoint.sh /mnt/checkpoints/.../10000 /tmp/my_ckpt "$PLACEMENT"
#
#   # All attention:
#   ./create_placement_checkpoint.sh /mnt/checkpoints/.../10000 /tmp/attn "$(printf 'attention,%.0s' {1..48} | sed 's/,$//')"
set -euo pipefail

# Default: surrogate search likelihood 6.4 cost placement
DEFAULT_PLACEMENT="sliding_window,gdn,gdn,gdn,gdn,sliding_window,kda,gdn,sliding_window,gdn,sliding_window,sliding_window,kda,gdn,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,sliding_window,gdn,sliding_window,kda,sliding_window,sliding_window,kda,sliding_window,gdn,gdn,sliding_window,gdn,gdn,sliding_window,sliding_window,gdn,sliding_window,sliding_window,kda"

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <supernet_path> <output_dir> [placement]"
    echo "  placement: comma-separated mixer names, one per layer (optional, has default)"
    exit 1
fi

SUPERNET_PATH="$1"
OUTPUT_DIR="$2"
PLACEMENT="${3:-$DEFAULT_PLACEMENT}"

if [[ ! -f "${SUPERNET_PATH}/config.json" ]]; then
    echo "ERROR: ${SUPERNET_PATH}/config.json not found" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Symlink everything from supernet checkpoint
for f in "${SUPERNET_PATH}"/*; do
    ln -sf "$f" "${OUTPUT_DIR}/$(basename "$f")"
done

# Replace config.json symlink with modified copy
rm -f "${OUTPUT_DIR}/config.json"

python3 <<PYEOF
import json, copy, sys

with open('${SUPERNET_PATH}/config.json') as f:
    cfg = json.load(f)

placement = '${PLACEMENT}'.split(',')
placement = [p.strip() for p in placement]

decoder = cfg['decoder']
num_blocks = decoder.get('num_blocks', cfg.get('num_hidden_layers', 0))

if len(placement) != num_blocks:
    print(f"ERROR: placement has {len(placement)} entries but model has {num_blocks} layers", file=sys.stderr)
    sys.exit(1)

# Extract sub-mixer configs from the stochastic block
block = decoder['block']
stochastic_mixer = block['mixer']
if stochastic_mixer.get('type') != 'stochastic':
    print(f"WARNING: mixer type is '{stochastic_mixer.get('type')}', expected 'stochastic'", file=sys.stderr)

available_mixers = stochastic_mixer.get('mixers', {})

# Validate all placement entries reference valid mixer names
unique_mixers = set(placement)
missing = unique_mixers - set(available_mixers.keys())
if missing:
    print(f"ERROR: unknown mixer(s) in placement: {missing}. Available: {list(available_mixers.keys())}", file=sys.stderr)
    sys.exit(1)

# Build pattern-mode config
# Each unique mixer becomes a named block in "blocks", sharing mlp/normalization
blocks = {}
for mixer_name in sorted(unique_mixers):
    blocks[mixer_name] = {
        "mixer": copy.deepcopy(available_mixers[mixer_name]),
        "mlp": copy.deepcopy(block['mlp']),
        "normalization": copy.deepcopy(block['normalization']),
    }

decoder['type'] = 'pattern'
decoder['pattern'] = placement
decoder['blocks'] = blocks
# Remove the old fixed-mode block config
decoder.pop('block', None)

with open('${OUTPUT_DIR}/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)

# Summary
from collections import Counter
counts = Counter(placement)
summary = ', '.join(f'{name}={count}' for name, count in sorted(counts.items()))
print(f'Created: ${OUTPUT_DIR}/config.json')
print(f'  {num_blocks} layers: {summary}')
PYEOF
