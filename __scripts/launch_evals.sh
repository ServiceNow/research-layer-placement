
DNS=http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id
IDS=(300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329)
############
#### EVAL WITH RANDOM PLACEMENT
############
# TODO
# nohup env \
#   OUTPUT_BASE=outputs/generative_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime24_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_eval_rand_placement_aime24.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/generative_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="325 326 327 328 329" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_eval_rand_placement_aime25.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/generative_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_eval_rand_placement_math.out 2>&1 &


# SQAD
# nohup env \
#   OUTPUT_BASE=outputs/ruller_qa_squad_random_archs \
#   DNS="${DNS}" \
#   IDS=" 306 307 308 309 310 311 312 313 314 315 316 317 318 319" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=ruler_qa_squad EVALUATOR_CFG=generative_extraction_ruller.yaml GENERATION_CFG=greedy_very_short \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_ruller_qad_eval_set1.out" 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/ruller_qa_squad_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=ruler_qa_squad EVALUATOR_CFG=generative_extraction_ruller.yaml GENERATION_CFG=greedy_very_short \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_ruller_qad_eval_set2.out" 2>&1 &

# NIAH
# nohup env \
#   OUTPUT_BASE=outputs/ruller_niah_single_random_archs \
#   DNS="${DNS}" \
#   IDS=" 340 341 342 343 344" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=niah_single_2 EVALUATOR_CFG=generative_extraction_ruller_long.yaml GENERATION_CFG=greedy_very_short \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_ruller_niah_eval_set1.out" 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/ruller_niah_single_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="345 346 347 348 349" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=niah_single_2 EVALUATOR_CFG=generative_extraction_ruller_long.yaml GENERATION_CFG=greedy_very_short \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_ruller_niah_eval_set2.out" 2>&1 &

# MMLU
# nohup env \
#   OUTPUT_BASE=outputs/mmlu_random_archs \
#   DNS="${DNS}" \
#   IDS="300 301 302 303 304 305 306 307 308 309" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10  \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood GENERATION_CFG=greedy_long MAX_PROBLEMS_PER_TASK=10 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_mmlu_eval_rand_placement.out" 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/mmlu_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="310 311 312 313 314 315 316 317 318 319" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood GENERATION_CFG=greedy_long MAX_PROBLEMS_PER_TASK=10 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_mmlu_eval_rand_placement_2.out" 2>&1 &
# Extraction

# nohup env \
#   OUTPUT_BASE=outputs/extraction_random_archs \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324 325" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=swde,fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_extraction_eval_rand_placement.out" 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/extraction_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="326 327 328 329 330" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=swde,fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_extraction_eval_rand_placement_2.out" 2>&1 &
  
# # GSM
# nohup env \
#   OUTPUT_BASE=outputs/gsm_random_archs \
#   DNS="${DNS}" \
#   IDS="331 332 333 334 335 336 337 338 339 340" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=gsm8k_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_gsm_eval_rand_placement.out" 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/gsm_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="341 342 343 344 345 346 347 348 349" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=gsm8k_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_gsm_eval_rand_placement_2.out" 2>&1 &


######
# EVAL with one leave out
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-200:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-201:8080\"" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=100 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG="gsm8k_strip_end" EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_gsm.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-202:8080\"" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG="swde" EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_swde.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-203:8080\"" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_fda.out 2>&1 &


# MATH
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-202:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-201:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-203:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-204:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-205:8080\"" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=500 MAX_CONCURRENT_PER_SERVER=1 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_loing \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_math.out 2>&1 &

# # AIME 25
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_long \
#   DNS="${DNS}" \
#   IDS="311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329" \
#   PORT="8080" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=30 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_eval_aime.out 2>&1 &

# # MMLU
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
#   DNS="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id\"" \
#   IDS="300 301" \
#   PORT="8080" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=10 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=mmlu VALUATOR_CFG=likelihood GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_mmlu.out 2>&1 &
  
################################################################################
################################# KDA to Attn
################################################################################
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_kda_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-300:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-301:8080\"" \
#   BASE_MIXER=kda REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=100 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG="gsm8k_strip_end" EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_gsm_kdatoat.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_kda_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-302:8080\"" \
#   BASE_MIXER=kda REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG="swde" EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_swde_kdatoat.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_kda_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-303:8080\"" \
#   BASE_MIXER=kda REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_fda_kdatoat.out 2>&1 &

# # MATH
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_kda_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-304:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-305:8080\"" \
#   BASE_MIXER=kda REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=500 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_math_kdatoat.out 2>&1 &

# # AIME 25
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_kda_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-307:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-308:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-309:8080\"" \
#   BASE_MIXER=kda REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=30 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_aime_kdatoat.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_kda_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-311:8080\"" \
#   BASE_MIXER=kda REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=10 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=mmlu VALUATOR_CFG=likelihood GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_mmlu_kdatoat.out 2>&1 &


  
################################################################################
################################# SWA to Attn
################################################################################
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-300:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-301:8080\"" \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=100 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG="gsm8k_strip_end" EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_gsm_swa_fa.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-302:8080\"" \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG="swde" EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_swde_swa_fa.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-303:8080\"" \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_fda_swa_fa.out 2>&1 &

# # MATH
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-304:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-305:8080\"" \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=500 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_math_swa_fa.out 2>&1 &

# # AIME 25
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-307:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-308:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-309:8080\"" \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=30 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_aime_swa_fa.out 2>&1 &

# # MMLU
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_3 \
#   SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-311:8080\"" \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=10 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=mmlu VALUATOR_CFG=likelihood GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_mmlu_swa_fa.out 2>&1 &

# TODO!
# GENERATIVE SWA to ATTN
# nohup env \
#   OUTPUT_BASE=outputs/generative_sweep_swa_to_attn_long \
#   DNS="${DNS}" \
#   IDS="300 301 302 303 304 305 306 307 308 309" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   BASE_MIXER=sliding_window REPLACE_MIXER=attention \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end,aime24_strip_end,math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_eval_generative_swa_fa.out 2>&1 &


################################################################################
################ PRECOMPUTE TRACES
################################################################################
# # TODO
# nohup env \
#   OUTPUT_BASE=outputs/math_traces_very_long \
#   DNS="${DNS}" \
#   IDS="300 301 302 303 304 305 306 307" \
#   PORT="8080" \
#   COMMAND="precompute_traces" \
#   TIMEOUT_SECONDS=1800 \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   START_LAYER=0 NUM_LAYERS=1 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=30 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end,aime24_strip_end EVALUATOR_CFG=trace_precompute GENERATION_CFG=greedy_very_very_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_precompute_all_aime_very_long.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/math_traces_long \
#   DNS="${DNS}" \
#   IDS="308 309 310 311 312 313 314 315 316 317 318 319" \
#   PORT="8080" \
#   COMMAND="precompute_traces" \
#   TIMEOUT_SECONDS=1800 \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   START_LAYER=0 NUM_LAYERS=1 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=500 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=trace_precompute GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_precompute_all_math.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/aime_traces_long \
#   DNS="${DNS}" \
#   IDS="330 331 332 333 334 335 336 337 338 339" \
#   PORT="8080" \
#   COMMAND="precompute_traces" \
#   TIMEOUT_SECONDS=1800 \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   START_LAYER=0 NUM_LAYERS=1 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=30 MAX_CONCURRENT_PER_SERVER=1 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=trace_precompute GENERATION_CFG=greedy_long_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_generative_aime.out 2>&1 &


################################################################################
### IS evaluator
################################################################################
# math generations: outputs/math_traces_long/+backend\=rest\,+command\=precompute_traces\,+evaluator\=trace_precompute\,+generation\=greedy_long\,+placement\=layer_0\,+task\=ma__06aaccefec24/a48/
# nohup env \
#   OUTPUT_BASE=outputs/math_is_eval_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="300 301 302 303 304 305 306 307 308 309" \
#   PORT="8080" \
#   COMMAND="evaluate" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=8 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=importance_weighted \
#   TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_long/+backend\=rest\,+command\=precompute_traces\,+evaluator\=trace_precompute\,+generation\=greedy_long\,+placement\=layer_0\,+task\=ma__06aaccefec24/a48/ \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_is_math_ra2.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/math_is_eval_random_archs \
#   DNS="${DNS}" \
#   IDS="310 311 312 313 314 315 316 317 318 319" \
#   PORT="8080" \
#   COMMAND="evaluate" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=8 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=importance_weighted \
#   TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_long/+backend\=rest\,+command\=precompute_traces\,+evaluator\=trace_precompute\,+generation\=greedy_long\,+placement\=layer_0\,+task\=ma__06aaccefec24/a48/ \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_is_math_ra.out 2>&1 &



# nohup env \
#   OUTPUT_BASE=outputs/aime25_is_eval_random_archs \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324 325 326 327 328 329" \
#   PORT="8080" \
#   COMMAND="evaluate" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=8 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=importance_weighted \
#   TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_very_long/+backend=rest,+command=precompute_traces,+evaluator=trace_precompute,+generation=greedy_very_very_long,+placement=layer___6c6925a94744/a48/ \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_is_aime25_ra.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/aime24_is_eval_random_archs \
#   DNS="${DNS}" \
#   IDS="330 331 332 333 334 335 336 337 338 339" \
#   PORT="8080" \
#   COMMAND="evaluate" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=8 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures \
#   TASK_CFG=aime24_strip_end EVALUATOR_CFG=importance_weighted \
#   TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_very_long/+backend=rest,+command=precompute_traces,+evaluator=trace_precompute,+generation=greedy_very_very_long,+placement=layer___238038b71fe0/a48/ \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_is_aime24_ra.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/aime25_is_eval_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="300 301 302 303 304 305 306 307 308 309" \
#   PORT="8080" \
#   COMMAND="evaluate" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=8 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=importance_weighted \
#   TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_very_long/+backend=rest,+command=precompute_traces,+evaluator=trace_precompute,+generation=greedy_very_very_long,+placement=layer___6c6925a94744/a48/ \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_is_aime25_ra_2.out 2>&1 &


# nohup env \
#   OUTPUT_BASE=outputs/aime24_is_eval_random_archs_2 \
#   DNS="${DNS}" \
#   IDS="310 311 312 313 314 315 316 317 318 319" \
#   PORT="8080" \
#   COMMAND="evaluate" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=49 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=8 \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_2 \
#   TASK_CFG=aime24_strip_end EVALUATOR_CFG=importance_weighted \
#   TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_very_long/+backend=rest,+command=precompute_traces,+evaluator=trace_precompute,+generation=greedy_very_very_long,+placement=layer___238038b71fe0/a48/ \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_is_aime24_ra_2.out 2>&1 &


################################################################################
################ ALL-ATTENTION BASELINE
################################################################################
# NUM_LAYERS=1 + no LAYOUTS_PATH → make_placement(0) produces ["attention"]
# (broadcasts to all layers). i=1 produces ["kda"] as a bonus KDA baseline.

# # GSM8K
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="300 301 303 304 305" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=gsm8k_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_gsm8k.out 2>&1 &

# # MATH 500
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="306 307 308 309 310" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_math500.out 2>&1 &

# # AIME 24
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="311 312 313 314 315" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime24_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_aime24.out 2>&1 &

# # AIME 25
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="316 317 318 319 320" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_aime25.out 2>&1 &

# # RULER QA SQUAD
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="321 322 323 324 325" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=ruler_qa_squad EVALUATOR_CFG=generative_extraction_ruller.yaml GENERATION_CFG=greedy_very_short \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_ruler_qa_squad.out 2>&1 &

# # NIAH
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="326 327 328 329" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=niah_single_2 EVALUATOR_CFG=generative_extraction_ruller_long.yaml GENERATION_CFG=greedy_very_short \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_niah.out 2>&1 &

# # MMLU
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="330 331 332 333 334" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 MAX_PROBLEMS_PER_TASK=10 \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_mmlu.out 2>&1 &

# # Extraction (SWDE + FDA)
# nohup env \
#   OUTPUT_BASE=outputs/baseline_all_attention \
#   DNS="${DNS}" \
#   IDS="336 337 338 339 340" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=swde,fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_extraction.out 2>&1 &

# SEARCH RESULTS

# GSM8K
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="300 301 303 304 305" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=gsm8k_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_gsm8k.out 2>&1 &

# MATH 500
nohup env \
  OUTPUT_BASE=outputs/evals/optimal_ss_10 \
  DNS="${DNS}" \
  IDS="306 307 308 309 310" \
  PORT="8080" \
  TIMEOUT_SECONDS=1800 \
  PLACEMENT_CFG=surogate_search_lieklihood_10cost \
  NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
  TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
  bash __scripts/eval_all_layers.sh \
  > nohup_baseline_math500.out 2>&1 &

# AIME 24
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="311 312 313 314 315" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime24_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_aime24.out 2>&1 &

# # AIME 25
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="316 317 318 319 320" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_aime25.out 2>&1 &

# # RULER QA SQUAD
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="321 322 323 324 325" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=ruler_qa_squad EVALUATOR_CFG=generative_extraction_ruller.yaml GENERATION_CFG=greedy_very_short \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_ruler_qa_squad.out 2>&1 &

# # NIAH
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="326 327 328 329" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=niah_single_2 EVALUATOR_CFG=generative_extraction_ruller_long.yaml GENERATION_CFG=greedy_very_short \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_niah.out 2>&1 &

# # MMLU
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="330 331 332 333 334" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 MAX_PROBLEMS_PER_TASK=10 \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood GENERATION_CFG=greedy_long \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_mmlu.out 2>&1 &

# # Extraction (SWDE + FDA)
# nohup env \
#   OUTPUT_BASE=outputs/evals/optimal_ss_10 \
#   DNS="${DNS}" \
#   IDS="336 337 338 339" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   PLACEMENT_CFG=surogate_search_lieklihood_10cost \
#   NUM_LAYERS=1 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=swde,fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   bash __scripts/eval_all_layers.sh \
#   > nohup_baseline_extraction.out 2>&1 &