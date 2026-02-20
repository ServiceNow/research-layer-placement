# explore
# nohup env OUTPUT_BASE=outputs/placement_explore_math_aime MAX_CONCURRENT_PER_SERVER=4 TIMEOUT_SECONDS=800 \
#  python -m place_layers +experiments=explore_is_aime_math \
#  backend=rest_dns \
#  backend.dns_prefix=dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f \
#  backend.num_servers=39 \
#  backend.server_id_offset=300 \
#  > "nohup_math_aime_placement_exploration.out" 2>&1 &


# place
# nohup env OUTPUT_BASE=outputs/placement_optimize_math_aime_cost10 MAX_CONCURRENT_PER_SERVER=4 TIMEOUT_SECONDS=800 \
#  python -m place_layers +experiments=optimize_is_viterbi_aime_math \
#  backend=rest_dns \
#  backend.dns_prefix=dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f \
#  backend.num_servers=39 \
#  problem.cost_budget=10.0 \
#  backend.server_id_offset=300 \
#  search_strategy.exploration_dataset_path="'/home/toolkit/dev/research-layer-placement/outputs/placement_explore_math_aime/+experiments=explore_is_aime_math,backend.dns_prefix=dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f,backend.num_servers=39,bac__96eda3543813/exploration_results.json'" \
#  > "nohup_math_aime_placement_optimize.out" 2>&1 &