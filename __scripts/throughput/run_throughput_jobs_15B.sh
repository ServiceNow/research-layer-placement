#!/usr/bin/env bash
# Submit throughput benchmark jobs to the EAI cluster.
# Each placement gets one job per length combination.
# Checkpoints must be created first via create_bench_checkpoints.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_fixed_placements"

# All placement names (must match create_bench_checkpoints.sh)
# PLACEMENTS=(
#     10000_optimized_bayesian_mix_18
#     10000_optimized_bayesian_mix_6
#     10000_surogate_search_lieklihood_6p4cost
#     10000_optimized_bayesian_mix_7
#     10000_pure_placements/pure_attention
#     10000_every_2nd_gdn_att
# )

PLACEMENTS=(
#   pure_att                                                                                                                                                                                                                                             
#   pure_gdn                                                                                                                                                                                                                                             
#   pure_kda                                                                                                                                                                                                                                             
  pure_sw                                                                                                                                                                                                                                              
  a24_g2_k22      
  a18_g28_s2                                                                                                                                                                                                                                           
  a16_g20_s12                                                                                                                                                                                                                                          
  att36_gdn12                                                                                                                                                                                                                                          
  att24_gdn24                                                                                                                                                                                                                                          
  att12_gdn36     
  att36_kda12                                                                                                                                                                                                                                          
  att24_kda24     
  att36_sw12
  gdn24_kda24
  gdn24_sw24                                                                                                                                                                                                                                           
  kda24_sw24
  att32_gdn8_kda8                                                                                                                                                                                                                                      
  att32_gdn8_sw8  
  att32_kda8_sw8
  att24_gdn12_kda12                                                                                                                                                                                                                                    
  att16_gdn16_kda16
  att16_gdn16_sw16                                                                                                                                                                                                                                     
  att16_kda16_sw16                                                                                                                                                                                                                                     
  gdn16_kda16_sw16
  equal_12_12_12_12                                                                                                                                                                                                                                    
  att_heavy_24_8_8_8
  att_heavy_18_10_10_10                                                                                                                                                                                                                                
  att_heavy_16_12_10_10
  gdn_heavy_8_24_8_8                                                                                                                                                                                                                                   
  kda_heavy_8_8_24_8
  sw_heavy_8_8_8_24                                                                                                                                                                                                                                    
  att_low_6_14_14_14
  no_att_0_16_16_16                                                                                                                                                                                                                                    
  att12_kda36     
  att24_sw24
  att12_sw36
  gdn36_kda12
  gdn36_sw12                                                                                                                                                                                                                                           
  kda36_sw12
  gdn20_kda14_sw14                                                                                                                                                                                                                                     
  att20_gdn14_sw14
  att20_kda14_sw14                                                                                                                                                                                                                                     
  att16_gdn12_kda10_sw10
  att42_gdn6                                                                                                                                                                                                                                           
  att30_gdn18                                                                                                                                                                                                                                          
  att18_gdn30                                                                                                                                                                                                                                          
  att6_gdn42                                                                                                                                                                                                                                           
  att42_kda6      
  att30_kda18
  att18_kda30                                                                                                                                                                                                                                          
  att6_kda42
  att42_sw6                                                                                                                                                                                                                                            
  att30_sw18      
  att18_sw30
  att6_sw42
  gdn36_kda12b
  gdn12_kda36                                                                                                                                                                                                                                          
  gdn42_kda6
  gdn6_kda42                                                                                                                                                                                                                                           
  gdn12_sw36                                                                                                                                                                                                                                           
  gdn36_sw12b                                                                                                                                                                                                                                          
  gdn42_sw6                                                                                                                                                                                                                                            
  gdn6_sw42       
  kda12_sw36
  kda36_sw12b                                                                                                                                                                                                                                          
  kda42_sw6
  kda6_sw42                                                                                                                                                                                                                                            
  att8_gdn32_kda8 
  att8_gdn8_kda32                                                                                                                                                                                                                                      
  att8_gdn32_sw8  
  att8_gdn8_sw32
  att8_kda32_sw8                                                                                                                                                                                                                                       
  att8_kda8_sw32
  gdn32_kda8_sw8                                                                                                                                                                                                                                       
  gdn8_kda32_sw8  
  gdn8_kda8_sw32                                                                                                                                                                                                                                       
  att32_gdn8_kda4_sw4
  att4_gdn32_kda8_sw4                                                                                                                                                                                                                                  
  att4_gdn8_kda32_sw4                                                                                                                                                                                                                                  
  att4_gdn8_kda4_sw32                                                                                                                                                                                                                                  
  att47_gdn1                                                                                                                                                                                                                                           
  att46_gdn2      
  att47_kda1                                                                                                                                                                                                                                           
  att46_kda2
  att47_sw1                                                                                                                                                                                                                                            
  att46_sw2       
  gdn47_att1
  gdn46_att2                                                                                                                                                                                                                                           
  gdn47_kda1
  gdn46_kda2                                                                                                                                                                                                                                           
  gdn47_sw1       
  gdn46_sw2                                                                                                                                                                                                                                            
  kda47_att1                                                                                                                                                                                                                                           
  kda46_att2                                                                                                                                                                                                                                           
  kda47_gdn1                                                                                                                                                                                                                                           
  kda46_gdn2      
  kda47_sw1
  kda46_sw2
  sw47_att1
  sw46_att2
  sw47_gdn1                                                                                                                                                                                                                                            
  sw46_gdn2
  sw47_kda1                                                                                                                                                                                                                                            
  sw46_kda2       
  att44_gdn2_kda2
  att44_gdn2_sw2
  att44_kda2_sw2                                                                                                                                                                                                                                       
  att40_gdn4_kda4
  att40_gdn4_sw4                                                                                                                                                                                                                                       
  att40_kda4_sw4  
  gdn44_att2_kda2                                                                                                                                                                                                                                      
  gdn44_att2_sw2  
  gdn40_att4_kda4
  gdn40_att4_sw4
  kda44_att2_gdn2                                                                                                                                                                                                                                      
  kda44_att2_sw2
  kda40_att4_gdn4                                                                                                                                                                                                                                      
  kda40_att4_sw4  
  att40_gdn4_kda2_sw2
  att38_gdn4_kda4_sw2                                                                                                                                                                                                                                  
  gdn40_att4_kda2_sw2
  gdn38_att4_kda4_sw2                                                                                                                                                                                                                                  
  kda40_att4_gdn2_sw2
  sw40_att4_gdn2_kda2
)

TOKENIZER="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000"
MAX_MODEL_LEN="65536"

declare -a length_combinations=(
    "1 16000 1-16k"
)

# vLLM settings
VLLM_ATTENTION_BACKEND=FLASHINFER
VLLM_USE_FLASHINFER_SAMPLER=1
NUM_PROMPTS=500
GPU_MEM_UTILIZATION=0.8
TP=1
COMPILATION_CONFIG='{\"cudagraph_mode\": \"FULL_AND_PIECEWISE\"}'

OUTPUT_BASE="/mnt/checkpoints/fast_llm_exp/apriel_train/throughput_results_15b1p6/throughput_results_np${NUM_PROMPTS}_gpu_util${GPU_MEM_UTILIZATION}_tp${TP}_atnn${VLLM_ATTENTION_BACKEND}_flashinfer_sampler${VLLM_USE_FLASHINFER_SAMPLER}_COMPILATION_CONFIG_$(echo "$COMPILATION_CONFIG" | tr -d '\"{}: ')"

SUBMITTED=0

for placement in "${PLACEMENTS[@]}"; do
    MODEL="${CHECKPOINT_BASE}/${placement}"
    # placement="pure_att"
    # MODEL="${CHECKPOINT_BASE}"
    for combo in "${length_combinations[@]}"; do
        read -r INPUT_LEN OUTPUT_LEN SUFFIX <<< "$combo"

        OUTPUT_JSON="${OUTPUT_BASE}/${placement}/${SUFFIX}_${NUM_PROMPTS}-prompts.json"
        # OUTPUT_JSON="${OUTPUT_BASE}/${placement}/${SUFFIX}_${NUM_PROMPTS}-prompts.json"

        RUN_NAME="vllm_bench_np${NUM_PROMPTS}_${placement}_${SUFFIX}"
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
        COMPILATION_CONFIG="$COMPILATION_CONFIG" \
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
