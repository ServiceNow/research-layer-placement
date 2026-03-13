#!/usr/bin/env bash
# Submit throughput benchmark jobs to the EAI cluster.
# Each placement gets one job per length combination.
# Checkpoints must be created first via create_bench_checkpoints.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_BASE="/mnt/checkpoints/upstream/apriel2-0.5b-dev-fixed-placemenbts"
# CHECKPOINT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements"
# CHECKPOINT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_pure_placements"
# CHECKPOINT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_pure_placements_no_supernet"
# CHECKPOINT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements"

# All placement names (must match create_bench_checkpoints.sh)
# PLACEMENTS=(
#     # Pure placements (4)
#     # pure_att
#     # pure_gdn
#     # pure_kda
#     pure_sw
#     # a12_g1_k11
#     # a9_g14_s1
#     # a8_g10_s6
# )

PLACEMENTS=(
    "pure_att"
    "pure_gdn"
    "pure_kda"
    "pure_sw"
    "a12_g1_k11"
    "a9_g14_s1"
    "a8_g10_s6"
    "att18_gdn6"
    "att12_gdn12"
    "att6_gdn18"
    "att18_kda6"
    "att12_kda12"
    "att18_sw6"
    "gdn12_kda12"
    "gdn12_sw12"
    "kda12_sw12"
    "att16_gdn4_kda4"
    "att16_gdn4_sw4"
    "att16_kda4_sw4"
    "att12_gdn6_kda6"
    "att8_gdn8_kda8"
    "att8_gdn8_sw8"
    "att8_kda8_sw8"
    "gdn8_kda8_sw8"
    "equal_6_6_6_6"
    "att_heavy_12_4_4_4"
    "att_heavy_9_5_5_5"
    "att_heavy_8_6_5_5"
    "gdn_heavy_4_12_4_4"
    "kda_heavy_4_4_12_4"
    "sw_heavy_4_4_4_12"
    "att_low_3_7_7_7"
    "no_att_0_8_8_8"
    "att6_kda18"
    "att12_sw12"
    "att6_sw18"
    "gdn18_kda6"
    "gdn18_sw6"
    "kda18_sw6"
    "gdn10_kda7_sw7"
    "att10_gdn7_sw7"
    "att10_kda7_sw7"
    "att8_gdn6_kda5_sw5"
    "att21_gdn3"
    "att15_gdn9"
    "att9_gdn15"
    "att3_gdn21"
    "att21_kda3"
    "att15_kda9"
    "att9_kda15"
    "att3_kda21"
    "att21_sw3"
    "att15_sw9"
    "att9_sw15"
    "att3_sw21"
    "gdn18_kda6"
    "gdn6_kda18"
    "gdn21_kda3"
    "gdn3_kda21"
    "gdn6_sw18"
    "gdn18_sw6"
    "gdn21_sw3"
    "gdn3_sw21"
    "kda6_sw18"
    "kda18_sw6"
    "kda21_sw3"
    "kda3_sw21"
    "att4_gdn16_kda4"
    "att4_gdn4_kda16"
    "att4_gdn16_sw4"
    "att4_gdn4_sw16"
    "att4_kda16_sw4"
    "att4_kda4_sw16"
    "gdn16_kda4_sw4"
    "gdn4_kda16_sw4"
    "gdn4_kda4_sw16"
    "att16_gdn4_kda2_sw2"
    "att2_gdn16_kda4_sw2"
    "att2_gdn4_kda16_sw2"
    "att2_gdn4_kda2_sw16"
    "att23_gdn1"       
    "att22_gdn2"       
    "att23_kda1"       
    "att22_kda2"       
    "att23_sw1"        
    "att22_sw2"        
    "gdn23_att1"        
    "gdn22_att2"        
    "gdn23_kda1"        
    "gdn22_kda2"        
    "gdn23_sw1"         
    "gdn22_sw2"         
    "kda23_att1"        
    "kda22_att2"        
    "kda23_gdn1"        
    "kda22_gdn2"        
    "kda23_sw1"         
    "kda22_sw2"         
    "sw23_att1"         
    "sw22_att2"         
    "sw23_gdn1"         
    "sw22_gdn2"         
    "sw23_kda1"         
    "sw22_kda2"
    "att21_gdn1_kda2"  
    "att21_gdn1_sw2"   
    "att21_kda1_sw2"   
    "att20_gdn2_kda2"  
    "att20_gdn2_sw2"   
    "att20_kda2_sw2"   
    "gdn21_att1_kda2"   
    "gdn21_att1_sw2"    
    "gdn20_att2_kda2"   
    "gdn20_att2_sw2"    
    "kda21_att1_gdn2"   
    "kda21_att1_sw2"    
    "kda20_att2_gdn2"   
    "kda20_att2_sw2"    
    "att20_gdn2_kda1_sw1"
    "att19_gdn2_kda2_sw1"
    "gdn20_att2_kda1_sw1"
    "gdn19_att2_kda2_sw1"
    "kda20_att2_gdn1_sw1"
    "sw20_att2_gdn1_kda1"
)


TOKENIZER="${CHECKPOINT_BASE}/pure_att"  # Same tokenizer for all placements
MAX_MODEL_LEN="65536"

declare -a length_combinations=(
    "1 16000 1-16k"
)

# vLLM settings
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_FLASHINFER_SAMPLER=1
NUM_PROMPTS=100
GPU_MEM_UTILIZATION=0.9
TP=1
# COMPILATION_CONFIG='{\"cudagraph_mode\": \"PIECEWISE\"}'

OUTPUT_BASE="/mnt/checkpoints/fast_llm_exp/apriel2-0.5b-dev/throughput_results_np${NUM_PROMPTS}_gpu_util${GPU_MEM_UTILIZATION}_tp${TP}_atnn${VLLM_ATTENTION_BACKEND}_flashinfer_sampler${VLLM_USE_FLASHINFER_SAMPLER}"

SUBMITTED=0

for placement in "${PLACEMENTS[@]}"; do
    MODEL="${CHECKPOINT_BASE}/${placement}"
    # placement="pure_att"
    # MODEL="${CHECKPOINT_BASE}"
    for combo in "${length_combinations[@]}"; do
        read -r INPUT_LEN OUTPUT_LEN SUFFIX <<< "$combo"

        OUTPUT_JSON="${OUTPUT_BASE}/${placement}/${SUFFIX}_${NUM_PROMPTS}-prompts.json"
        # OUTPUT_JSON="${OUTPUT_BASE}/${placement}/${SUFFIX}_${NUM_PROMPTS}-prompts.json"

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

        # COMPILATION_CONFIG="$COMPILATION_CONFIG" \
echo ""
echo "Submitted ${SUBMITTED} jobs. Results will appear in ${OUTPUT_BASE}/"
