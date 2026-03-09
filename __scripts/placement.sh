# explore
# nohup env OUTPUT_BASE=outputs/placement/ap1p6_frm1p5_distil_annealed_cpt10k_mixo_gs/placement_explore_math_aime MAX_CONCURRENT_PER_SERVER=4 TIMEOUT_SECONDS=12000 \
#  python -m place_layers +experiments=explore_is_aime_math_cpt10_mixo_gs \
#  backend=rest_dns \
#  backend.dns_prefix=dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f \
#  backend.num_servers=20 \
#  backend.server_id_offset=340 \
#  > "nohup_math_aime_placement_exploration.out" 2>&1 &


# place
nohup env OUTPUT_BASE=outputs/placement/ap1p6_frm1p5_distil_annealed_cpt10k_mixo_gs/placement_optimize_math_aime_cost6_4 MAX_CONCURRENT_PER_SERVER=4 TIMEOUT_SECONDS=12000 \
 python -m place_layers +experiments=optimize_is_viterbi_aime_math_cpt10_mixo_gs \
 backend=rest_dns \
 backend.dns_prefix=dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f \
 backend.num_servers=40 \
 problem.cost_budget=6.4 \
 backend.server_id_offset=300 \
 search_strategy.exploration_dataset_path="'/home/toolkit/dev/research-layer-placement/outputs/placement/ap1p6_frm1p5_distil_annealed_cpt10k_mixo_gs/placement_explore_math_aime/+experiments=explore_is_aime_math_cpt10_mixo_gs,backend.dns_prefix=dns-b680aca4-03a8-4fe6-bbc1-3a659786fd8f,backend.num___19e3ef100fc8/exploration_results_weighted_average.json'" \
 > "nohup_math_aime_placement_optimize.out" 2>&1 &