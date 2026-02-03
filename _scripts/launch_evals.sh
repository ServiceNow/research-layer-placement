# nohup env \
#   OUTPUT_BASE=outputs/mmlu_sweep_attention_to_kda \
#   SERVER_URL="http://0.0.0.0:8080" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=10 \
#   USE_CHAT_TEMPLATE=true \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood \
#   bash _scripts/eval_all_layers.sh \
#   > nohup_mmlu.out 2>&1 &

nohup env \
  OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
  SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-200:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-201:8080\"" \
  BASE_MIXER=attention REPLACE_MIXER=kda \
  NUM_LAYERS=48 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=100 MAX_CONCURRENT_PER_SERVER=4 \
  TASK_CFG="gsm8k_strip_end" EVALUATOR_CFG=generative GENERATION_CFG=greedy \
  bash _scripts/eval_all_layers.sh \
  > nohup_generative_gsm.out 2>&1 &


nohup env \
  OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
  SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-202:8080\"" \
  BASE_MIXER=attention REPLACE_MIXER=kda \
  NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
  TASK_CFG="swde" EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
  bash _scripts/eval_all_layers.sh \
  > nohup_generative_swde.out 2>&1 &

nohup env \
  OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
  SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-203:8080\"" \
  BASE_MIXER=attention REPLACE_MIXER=kda \
  NUM_LAYERS=48 MAX_JOBS=1 MAX_CONCURRENT_PER_SERVER=4 \
  TASK_CFG=fda EVALUATOR_CFG=extraction GENERATION_CFG=extraction \
  bash _scripts/eval_all_layers.sh \
  > nohup_generative_fda.out 2>&1 &


# MATH
nohup env \
  OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
  SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-204:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-205:8080\"" \
  BASE_MIXER=attention REPLACE_MIXER=kda \
  NUM_LAYERS=2 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=100 MAX_CONCURRENT_PER_SERVER=4 \
  TASK_CFG=math_500_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy \
  bash _scripts/eval_all_layers.sh \
  > nohup_generative_math.out 2>&1 &

# AIME 25
nohup env \
  OUTPUT_BASE=outputs/generative_sweep_attention_to_kda_3 \
  SERVER_URL="\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-206:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-207:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-208:8080\",\"http://dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f-id-209:8080\"" \
  BASE_MIXER=attention REPLACE_MIXER=kda \
  NUM_LAYERS=2 MAX_JOBS=1 MAX_PROBLEMS_PER_TASK=10 MAX_CONCURRENT_PER_SERVER=4 \
  TASK_CFG=aime25_strip_end EVALUATOR_CFG=generative GENERATION_CFG=greedy_long \
  bash _scripts/eval_all_layers.sh \
  > nohup_generative_aime.out 2>&1 &

# nohup env \
#   OUTPUT_BASE=outputs/sweep_attention_to_kda \
#   SERVER_URL="http://0.0.0.0:8080" \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 MAX_JOBS=1  MAX_PROBLEMS_PER_TASK=100 \
#   USE_CHAT_TEMPLATE=true \
#   TASK_CFG=swde EVALUATOR_CFG=generative \
#   bash _scripts/eval_all_layers.sh \
#   > nohup_swde.out 2>&1 &



#   bash eval_all_layers.sh \
#   OUTPUT_BASE=outputs/mmlu_kda_sweep_attention_to_kda \
#   SERVER_URL="http:// \
#   TASK_CFG=mmlu EVALUATOR_CFG=likelihood \
#   BASE_MIXER=attention REPLACE_MIXER=kda \
#   NUM_LAYERS=48 \
#   MAX_JOBS=2 MAX_PROBLEMS_PER_TASK=10 \
#   USE_CHAT_TEMPLATE=true \
#   RUN_PREFIX=mmlu_likelihood_attention_to_kda \