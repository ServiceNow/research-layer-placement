#!/usr/bin/env bash
# Submit throughput benchmark jobs to the EAI cluster.
# Each placement gets one job per length combination.
# Checkpoints must be created first via create_bench_checkpoints.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements"

# All placement names (must match create_bench_checkpoints.sh)
PLACEMENTS=(
    # Pure placements (4)
    pure_att
    pure_gdn
    pure_kda
    pure_sw

    # Binary mixes — attention vs each other (6)
    att36_gdn12
    att24_gdn24
    att12_gdn36
    att36_kda12
    att24_kda24
    att36_sw12

    # Binary mixes — non-attention pairs (3)
    gdn24_kda24
    gdn24_sw24
    kda24_sw24

    # Ternary mixes — attention-heavy (4)
    att30_gdn9_kda9
    att30_gdn9_sw9
    att30_kda9_sw9
    att20_gdn14_kda14

    # Ternary mixes — balanced (4)
    att16_gdn16_kda16
    att16_gdn16_sw16
    att16_kda16_sw16
    gdn16_kda16_sw16

    # Quaternary mixes (9)
    equal_12_12_12_12
    att_heavy_30_6_6_6
    att_heavy_24_8_8_8
    att_heavy_20_10_9_9
    gdn_heavy_8_24_8_8
    kda_heavy_8_8_24_8
    sw_heavy_8_8_8_24
    att_low_6_14_14_14
    no_att_0_16_16_16

    # Additional binary mixes — fill gaps (3)
    att12_kda36
    att24_sw24
    att12_sw36

    # Additional binary mixes — non-attention asymmetric (3)
    gdn36_kda12
    gdn36_sw12
    kda36_sw12

    # Additional ternary mixes (3)
    gdn20_kda14_sw14
    att20_gdn14_sw14
    att20_kda14_sw14
    att_med_17_11_10_10
)

TOKENIZER="${CHECKPOINT_BASE}/pure_att"
MAX_MODEL_LEN="65536"

declare -a length_combinations=(
    "1 16000 1-16k"
)

# vLLM settings
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_FLASHINFER_SAMPLER=1
NUM_PROMPTS=1000
GPU_MEM_UTILIZATION=0.9
TP=1

OUTPUT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/throughput_results"

SUBMITTED=0

for placement in "${PLACEMENTS[@]}"; do
    MODEL="${CHECKPOINT_BASE}/${placement}"

    if [[ ! -d "$MODEL" ]]; then
        echo "WARNING: checkpoint not found: $MODEL — skipping"
        continue
    fi

    for combo in "${length_combinations[@]}"; do
        read -r INPUT_LEN OUTPUT_LEN SUFFIX <<< "$combo"

        OUTPUT_JSON="${OUTPUT_BASE}/${placement}/${SUFFIX}_${NUM_PROMPTS}-prompts.json"

        RUN_NAME="vllm_bench_tp_${placement}_${SUFFIX}"
        RUN_NAME=$(echo "$RUN_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_]/_/g')

        echo "=== Submitting: ${placement} (${SUFFIX}) ==="

        GPU=1 TP="$TP" \
        NUM_PROMPTS="$NUM_PROMPTS" \
        MODEL="$MODEL" \
        TOKENIZER="$TOKENIZER" \
        GPU_MEM_UTILIZATION="$GPU_MEM_UTILIZATION" \
        MAX_MODEL_LEN="$MAX_MODEL_LEN" \
        INPUT_LEN="$INPUT_LEN" \
        OUTPUT_LEN="$OUTPUT_LEN" \
        OUTPUT_JSON="$OUTPUT_JSON" \
        VLLM_ATTENTION_BACKEND="$VLLM_ATTENTION_BACKEND" \
        VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
        EXP_ARGS__RUN_NAME="$RUN_NAME" \
            make -C "$SCRIPT_DIR" run-vllm-throughput

        SUBMITTED=$((SUBMITTED + 1))
        sleep 5
    done
done

echo ""
echo "Submitted ${SUBMITTED} jobs. Results will appear in ${OUTPUT_BASE}/"
