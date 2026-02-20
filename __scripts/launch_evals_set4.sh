
DNS=http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id
LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_3

############
#### SET 3: Leave-one-out + bookend architectures (72 placements, indices 0-71)
############

# MMLU (likelihood, MAX_PROBLEMS_PER_TASK=10)
# nohup env \
#   OUTPUT_BASE=outputs/mmlu_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="300 301 302 303 304" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood GENERATION_CFG=greedy_long MAX_PROBLEMS_PER_TASK=10 \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_mmlu_eval_set3.out" 2>&1 &


# GSM8K (generative, greedy)
# nohup env \
#   OUTPUT_BASE=outputs/gsm_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="310 311 312 313 314 315" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=gsm8k_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_gsm_eval_set3.out" 2>&1 &

# # Extraction (swde + fda)
# nohup env \
#   OUTPUT_BASE=outputs/extraction_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324 325" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=swde,fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_extraction_eval_set3.out" 2>&1 &


# AIME 24 (generative, greedy_very_long)
# nohup env \
#   OUTPUT_BASE=outputs/generative_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="326 327 328 329 330" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime24_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_aime24_eval_set3.out" 2>&1 &

# AIME 25 (generative, greedy_very_long)
# nohup env \
#   OUTPUT_BASE=outputs/generative_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324 325" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_very_long \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_aime25_eval_set3.out" 2>&1 &

# # MATH-500 (generative, greedy_long)
# nohup env \
#   OUTPUT_BASE=outputs/generative_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="305 306 307 308 309 316 317 318 319" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
#   TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_math_eval_set3.out" 2>&1 &

# SQAD
# nohup env \
#   OUTPUT_BASE=outputs/ruller_qa_squad_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="301 302 303 304 305 306 307 308" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=ruler_qa_squad EVALUATOR_CFG=generative_extraction_ruller.yaml GENERATION_CFG=greedy_very_short \
#   LAYOUTS_PATH="${LAYOUTS_PATH}" \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_ruller_qad_eval_set3.out" 2>&1 &

# NIAH
# nohup env \
#   OUTPUT_BASE=outputs/ruller_niah_single_random_archs_3 \
#   DNS="${DNS}" \
#   IDS="320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339" \
#   PORT="8080" \
#   TIMEOUT_SECONDS=1800 \
#   NUM_LAYERS=71 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=15 \
#   TASK_CFG=niah_single_2 EVALUATOR_CFG=generative_extraction_ruller_long.yaml GENERATION_CFG=greedy_very_short \
#   LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_3 \
#   bash __scripts/eval_all_layers.sh \
#   > "nohup_ruller_niah_eval_set3.out" 2>&1 &

# IS
nohup env \
  OUTPUT_BASE=outputs/aime25_is_eval_archs_4 \
  DNS="${DNS}" \
  IDS="300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315" \
  PORT="8080" \
  COMMAND="evaluate" \
  TIMEOUT_SECONDS=1800 \
  NUM_LAYERS=500 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 \
  LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_4 \
  TASK_CFG=aime25_strip_end EVALUATOR_CFG=importance_weighted \
  TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_very_long/+backend=rest,+command=precompute_traces,+evaluator=trace_precompute,+generation=greedy_very_very_long,+placement=layer___6c6925a94744/a48/ \
  bash __scripts/eval_all_layers.sh \
  > nohup_is_aime25_ra_4.out 2>&1 &


nohup env \
  OUTPUT_BASE=outputs/aime24_is_eval_archs_4 \
  DNS="${DNS}" \
  IDS="316 317 318 319 320 321 322 323 324 325 326 327 328" \
  PORT="8080" \
  COMMAND="evaluate" \
  TIMEOUT_SECONDS=1800 \
  NUM_LAYERS=500 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 \
  LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_4 \
  TASK_CFG=aime24_strip_end EVALUATOR_CFG=importance_weighted \
  TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_very_long/+backend=rest,+command=precompute_traces,+evaluator=trace_precompute,+generation=greedy_very_very_long,+placement=layer___238038b71fe0/a48/ \
  bash __scripts/eval_all_layers.sh \
  > nohup_is_aime24_ra_4.out 2>&1 &

nohup env \
  OUTPUT_BASE=outputs/math_is_eval_archs_4 \
  DNS="${DNS}" \
  IDS="329 330 331 332 333 334 335 336 337 338 339 340" \
  PORT="8080" \
  COMMAND="evaluate" \
  TIMEOUT_SECONDS=1800 \
  NUM_LAYERS=500 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=10 \
  LAYOUTS_PATH=/home/toolkit/dev/research-layer-placement/assets/__oo/layer_placement_random_architectures_4 \
  TASK_CFG=math_500_strip_end EVALUATOR_CFG=importance_weighted \
  TRACE_PATH=/home/toolkit/dev/research-layer-placement/outputs/math_traces_long/+backend\=rest\,+command\=precompute_traces\,+evaluator\=trace_precompute\,+generation\=greedy_long\,+placement\=layer_0\,+task\=ma__06aaccefec24/a48/ \
  bash __scripts/eval_all_layers.sh \
  > nohup_is_math_ra_4.out 2>&1 &
