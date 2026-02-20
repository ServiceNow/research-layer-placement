#!/usr/bin/env bash
# Create checkpoints for all placements used in bench_throughput.sh.
# Each placement is [attention]*n_a + [gdn]*n_g + [kda]*n_k + [sw]*n_s (47 layers).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CREATE_SCRIPT="${SCRIPT_DIR}/create_placement_checkpoint.sh"

SUPERNET="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000"
OUTPUT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements"

make_placement() {
    local n_att=$1 n_gdn=$2 n_kda=$3 n_sw=$4
    python3 -c "
p = ['attention']*${n_att} + ['gdn']*${n_gdn} + ['kda']*${n_kda} + ['sliding_window']*${n_sw}
print(','.join(p))
"
}

create_ckpt() {
    local label=$1 n_att=$2 n_gdn=$3 n_kda=$4 n_sw=$5
    local placement
    placement="$(make_placement "$n_att" "$n_gdn" "$n_kda" "$n_sw")"
    "$CREATE_SCRIPT" "$SUPERNET" "${OUTPUT_BASE}/${label}" "$placement"
}

# Pure placements (4)
# create_ckpt "pure_att"       48  0  0  0
# create_ckpt "pure_gdn"        0 48  0  0
# create_ckpt "pure_kda"        0  0 48  0
# create_ckpt "pure_sw"         0  0  0 48

# # Binary mixes — attention vs each other (6)
# create_ckpt "att36_gdn12"    36 12  0  0
# create_ckpt "att24_gdn24"    24 24  0  0
# create_ckpt "att12_gdn36"    12 36  0  0
# create_ckpt "att36_kda12"    36  0 12  0
# create_ckpt "att24_kda24"    24  0 24  0
# create_ckpt "att36_sw12"     36  0  0 12

# # Binary mixes — non-attention pairs (3)
# create_ckpt "gdn24_kda24"     0 24 24  0
# create_ckpt "gdn24_sw24"      0 24  0 24
# create_ckpt "kda24_sw24"      0  0 24 24

# # Ternary mixes — attention-heavy (4)
# create_ckpt "att30_gdn9_kda9"     30  9  9  0
# create_ckpt "att30_gdn9_sw9"      30  9  0  9
# create_ckpt "att30_kda9_sw9"      30  0  9  9
# create_ckpt "att20_gdn14_kda14"   20 14 14  0

# # Ternary mixes — balanced (4)
# create_ckpt "att16_gdn16_kda16"   16 16 16  0
# create_ckpt "att16_gdn16_sw16"    16 16  0 16
# create_ckpt "att16_kda16_sw16"    16  0 16 16
# create_ckpt "gdn16_kda16_sw16"     0 16 16 16

# # Quaternary mixes (9)
# create_ckpt "equal_12_12_12_12"   12 12 12 12
# create_ckpt "att_heavy_30_6_6_6"  30  6  6  6
# create_ckpt "att_heavy_24_8_8_8"  24  8  8  8
# create_ckpt "att_heavy_20_10_9_9" 20 10  9  9
# create_ckpt "gdn_heavy_8_24_8_8"   8 24  8  8
# create_ckpt "kda_heavy_8_8_24_8"   8  8 24  8
# create_ckpt "sw_heavy_8_8_8_24"    8  8  8 24
# create_ckpt "att_low_6_14_14_14"   6 14 14 14
# create_ckpt "no_att_0_16_16_16"    0 16 16 16

# # Additional binary mixes — fill gaps (3)
# create_ckpt "att12_kda36"          12  0 36  0
# create_ckpt "att24_sw24"           24  0  0 24
# create_ckpt "att12_sw36"           12  0  0 36

# # Additional binary mixes — non-attention asymmetric (3)
# create_ckpt "gdn36_kda12"          0 36 12  0
# create_ckpt "gdn36_sw12"           0 36  0 12
# create_ckpt "kda36_sw12"           0  0 36 12

# # Additional ternary mixes (3)
# create_ckpt "gdn20_kda14_sw14"     0 20 14 14
# create_ckpt "att20_gdn14_sw14"    20 14  0 14
# create_ckpt "att20_kda14_sw14"    20  0 14 14

# # Additional quaternary mix (1)
create_ckpt "att_med_17_11_10_10" 17 11 10 10

echo ""
echo "All checkpoints created in ${OUTPUT_BASE}/"
ls -1d "${OUTPUT_BASE}"/*/ 2>/dev/null | wc -l
echo "checkpoint directories."
