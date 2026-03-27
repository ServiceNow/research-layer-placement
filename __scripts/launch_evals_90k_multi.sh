#!/usr/bin/env bash

OUT="outputs/evals_90k_fixed"
export NOHUP_DIR="nohup_outs_90k"
export EMPTY_GENERATION_MAX_RETRIES=10
# nohup bash __scripts/launch_eval_batch.sh \
#     --name ap1p6_frm1p5_distil_anld_sftv2_15k_8pls \
#     --placements surogate_search_lieklihood_6.4cost,optimized-bayesian-mix-6,optimized-bayesian-mix-7,all-attention,new-cost-10,new-cost-13,new-cost-18,new-cost-26 \
#     --ids 300-349 \
#     --timeout 28000 \
#     --output-root $OUT \
#     > "${NOHUP_DIR}/nohup_batch_ap1p6_frm1p5_distil_anld_sftv2_15k_8pls.out" 2>&1 &

# echo "Launched batch eval. PID: $!"
# echo "Log: ${NOHUP_DIR}/nohup_batch_ap1p6_frm1p5_distil_anld_sftv2_15k_8pls.out"

# --- New tasks (gpqa, mmlu_pro, ifeval) on separate model/servers ---
# eval on S1 model with all placements and new tasks (gpqa, mmlu_pro, ifeval)
# TODO: later launch also S1 model for the missing placements
nohup bash __scripts/launch_eval_batch.sh \
    --name ap1p6_frm1p5_distil_anld \
    --placements surogate_search_lieklihood_6.4cost,optimized-bayesian-mix-6,optimized-bayesian-mix-7,all-attention,new-cost-10,new-cost-13,new-cost-18,new-cost-26 \
    --ids 350-379 \
    --tasks gpqa,mmlu_pro,ifeval \
    --timeout 28000 \
    --output-root $OUT \
    > "${NOHUP_DIR}/ap1p6_frm1p5_distil_anld.out" 2>&1 &

echo "Launched new tasks eval. PID: $!"
echo "Log: ${NOHUP_DIR}/ap1p6_frm1p5_distil_anld.out"
