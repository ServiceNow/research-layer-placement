# OUTPUT_BASE=outputs/evals_vllm_serve_test_vllm16 nohup python -m place_layers \
#   +command=evaluate \
#   +evaluator@evaluators.configs.aime=generative \
#   +backend=vllm \
#   backend.client.client_type=remote \
#   backend.client.base_url=http://0.0.0.0:8000 \
#   backend.client.model="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement" \
#   +task=aime25_strip_end \
#   +generation=greedy_very_long \
#   generation.include_logprobs=true \
#   backend.client.max_concurrent_requests=2 \
#   backend.client.timeout_seconds=1800 > logs/evals_vllm_serve.log 2>&1 &



# OUTPUT_BASE=outputs/evals_vllm_new_checkpoint nohup python -m place_layers \
#   +command=evaluate \
#   +evaluator@evaluators.configs.aime=generative \
#   +backend=rest \
#   'backend.servers=["http://0.0.0.0:8000"]' \
#   +task=aime25_strip_end \
#   +generation=greedy_very_long \
#   +placement=surogate_search_lieklihood_6.4cost \
#   backend.max_concurrent_per_server=4 \
#   backend.timeout_seconds=1800 > logs/evals_vllm.log 2>&1 &


OUTPUT_BASE=outputs/evals_vllm_new_checkpoint nohup python -m place_layers \
  +command=evaluate \
  +evaluator@evaluators.configs.aime=generative \
  +backend=rest \
  'backend.servers=["http://0.0.0.0:8000"]' \
  +task=aime25_strip_end \
  +generation=greedy_very_long \
  +placement=surogate_search_lieklihood_6.4cost \
  backend.max_concurrent_per_server=4 \
  backend.timeout_seconds=1800 > logs/evals_vllm.log 2>&1 &