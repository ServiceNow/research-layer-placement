#!/usr/bin/env bash
# Throughput benchmark across ~30 mixer count combinations on 47 layers.
# Each placement runs on its own dedicated server (1 server per eval).
# Evals run concurrently with different server offsets.
#
# Mixer types: attention, gdn, kda, sliding_window
# Each placement is constructed as: [attention]*n_a + [gdn]*n_g + [kda]*n_k + [sw]*n_s
# where n_a + n_g + n_k + n_s = 47

set -euo pipefail

DNS_PREFIX="dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f"
OUTPUT_BASE="outputs/throughput_10_requests"
LOG_DIR="logs/throughput_10_requests"
PLACEMENT_DIR="${OUTPUT_BASE}/_placement_configs/placement"
mkdir -p "$LOG_DIR" "$PLACEMENT_DIR"

# Resolve absolute path for hydra searchpath
PLACEMENT_SEARCHPATH="$(cd "${OUTPUT_BASE}/_placement_configs" && pwd)"

# Per-server settings
MAX_CONCURRENT=1000
TIMEOUT=80000
INPUT_SEQ_LEN=1
OUTPUT_SEQ_LEN=16000
NUM_REQUESTS=10
NUM_WARMUP=5

# Starting server offset — each eval gets its own server at BASE_OFFSET + index
BASE_OFFSET=300

# Generate placement YAML config file and return its name
make_placement_config() {
    local label=$1 n_att=$2 n_gdn=$3 n_kda=$4 n_sw=$5
    local cfg_file="${PLACEMENT_DIR}/${label}.yaml"

    if [[ ! -f "$cfg_file" ]]; then
        N_ATT="$n_att" N_GDN="$n_gdn" N_KDA="$n_kda" N_SW="$n_sw" \
            python3 -c '
import json, os
p = (["attention"] * int(os.environ["N_ATT"]) +
     ["gdn"] * int(os.environ["N_GDN"]) +
     ["kda"] * int(os.environ["N_KDA"]) +
     ["sliding_window"] * int(os.environ["N_SW"]))
print("# @package _global_")
print(f"placement: {json.dumps(p)}")
' > "$cfg_file"
    fi

    echo "$label"
}

# Track server index for assigning each eval a unique server
SERVER_IDX=0
PIDS=()
LABELS=()

run_bench() {
    local label=$1 n_att=$2 n_gdn=$3 n_kda=$4 n_sw=$5
    local cfg_name
    cfg_name="$(make_placement_config "$label" "$n_att" "$n_gdn" "$n_kda" "$n_sw")"

    local offset=$((BASE_OFFSET + SERVER_IDX))
    SERVER_IDX=$((SERVER_IDX + 1))

    echo "=== [$label] att=$n_att gdn=$n_gdn kda=$n_kda sw=$n_sw (server offset=$offset) ==="

    env OUTPUT_BASE="$OUTPUT_BASE" python -m place_layers \
        +command=evaluate \
        +evaluator=throughput \
        +backend=rest_dns \
        backend.dns_prefix="$DNS_PREFIX" \
        backend.num_servers=1 \
        backend.server_id_offset="$offset" \
        backend.timeout_seconds="$TIMEOUT" \
        backend.max_concurrent_per_server="$MAX_CONCURRENT" \
        evaluator.input_seq_len="$INPUT_SEQ_LEN" \
        evaluator.output_seq_len="$OUTPUT_SEQ_LEN" \
        evaluator.num_requests="$NUM_REQUESTS" \
        evaluator.num_warmup_requests="$NUM_WARMUP" \
        "hydra.searchpath=[file://${PLACEMENT_SEARCHPATH}]" \
        "+placement=${cfg_name}" \
        > "$LOG_DIR/${label}.log" 2>&1 &

    PIDS+=($!)
    LABELS+=("$label")
}

# --------------------------------------------------------------------------
# ~30 combos: (attention, gdn, kda, sliding_window) summing to 47
# --------------------------------------------------------------------------

# Pure placements (4)
run_bench "pure_att"       47  0  0  0
run_bench "pure_gdn"        0 47  0  0
run_bench "pure_kda"        0  0 47  0
run_bench "pure_sw"         0  0  0 47

# Binary mixes — attention vs each other (6)
run_bench "att35_gdn12"    35 12  0  0
run_bench "att24_gdn23"    24 23  0  0
run_bench "att12_gdn35"    12 35  0  0
run_bench "att35_kda12"    35  0 12  0
run_bench "att24_kda23"    24  0 23  0
run_bench "att35_sw12"     35  0  0 12

# Binary mixes — non-attention pairs (3)
run_bench "gdn24_kda23"     0 24 23  0
run_bench "gdn24_sw23"      0 24  0 23
run_bench "kda24_sw23"      0  0 24 23

# Ternary mixes — attention-heavy (4)
run_bench "att30_gdn9_kda8"     30  9  8  0
run_bench "att30_gdn9_sw8"      30  9  0  8
run_bench "att30_kda9_sw8"      30  0  9  8
run_bench "att20_gdn14_kda13"   20 14 13  0

# Ternary mixes — balanced (4)
run_bench "att16_gdn16_kda15"   16 16 15  0
run_bench "att16_gdn16_sw15"    16 16  0 15
run_bench "att16_kda16_sw15"    16  0 16 15
run_bench "gdn16_kda16_sw15"     0 16 16 15

# # Quaternary mixes (9)
run_bench "equal_12_12_12_11"   12 12 12 11
run_bench "att_heavy_30_6_6_5"  30  6  6  5
run_bench "att_heavy_24_8_8_7"  24  8  8  7
run_bench "att_heavy_20_10_9_8" 20 10  9  8
run_bench "gdn_heavy_8_23_8_8"   8 23  8  8
run_bench "kda_heavy_8_8_23_8"   8  8 23  8
run_bench "sw_heavy_8_8_8_23"    8  8  8 23
run_bench "att_low_5_14_14_14"   5 14 14 14
run_bench "no_att_0_16_16_15"    0 16 16 15

# Additional binary mixes — fill gaps (3)
run_bench "att12_kda35"          12  0 35  0
run_bench "att24_sw23"           24  0  0 23
run_bench "att12_sw35"           12  0  0 35

# Additional binary mixes — non-attention asymmetric (3)
run_bench "gdn35_kda12"          0 35 12  0
run_bench "gdn35_sw12"           0 35  0 12
run_bench "kda35_sw12"           0  0 35 12

# Additional ternary mixes (3)
run_bench "gdn20_kda14_sw13"     0 20 14 13
run_bench "att20_gdn14_sw13"    20 14  0 13
run_bench "att20_kda14_sw13"    20  0 14 13

# # Additional quaternary mix (1)
# run_bench "att_med_16_11_10_10" 16 11 10 10

echo ""
echo "Launched ${#PIDS[@]} benchmarks concurrently (servers $BASE_OFFSET-$((BASE_OFFSET + SERVER_IDX - 1)))"
echo "Logs in $LOG_DIR/"
echo ""
echo "Waiting for all to complete..."

# Wait for all and report results
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  [DONE] ${LABELS[$i]}"
    else
        echo "  [FAIL] ${LABELS[$i]} (see $LOG_DIR/${LABELS[$i]}.log)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [[ $FAILED -eq 0 ]]; then
    echo "All ${#PIDS[@]} benchmarks completed successfully."
else
    echo "${FAILED}/${#PIDS[@]} benchmarks failed."
fi
