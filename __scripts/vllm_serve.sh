#!/usr/bin/env bash
# Serve an extracted Apriel2 checkpoint (att12_gdn36) with vLLM.
set -euo pipefail

# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/pure_att"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/pure_gdn"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/pure_kda"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/att36_gdn12"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/att36_kda12"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/att_low_6_14_14_14"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/no_att_0_16_16_16"
MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement"
# MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement_interleaved"
# /usr/local/bin/vllm serve -- 14.0 rc1
# APRIEL2_DEBUG_STATE_INDICES=1 nohup vllm serve "$MODEL" \
#     --trust-remote-code \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 \
#     --host 0.0.0.0 \
#     --max-model-len 65536 \
#     --no-enable-prefix-caching \
# 	--max-num-seqs 256 \
#     --tokenizer /mnt/checkpoints/upstream/Apriel-1.6-15b-Thinker/ \
#     --port 8000 > logs/vllm_serve.log 2>&1 &

    # --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \ <-- works
    # --compilation-config '{"cudagraph_mode": "NONE"}' \ <- works
     # '{"pass_config": {"eliminate_noops": true}, "inductor_compile_config": {"triton.cudagraphs": false, "allow_buffer_reuse": false, "coordinate_descent_tuning": true}}' \
 # '{"pass_config": {"eliminate_noops": true}, "inductor_compile_config": {"triton.cudagraphs": false, "allow_buffer_reuse": false, "coordinate_descent_tuning": true}}' \
#   1. Test the grouped/interleaved checkpoints first — tells us if ANY interleaving is broken or just that exact pattern                                                                          
#   2. Disable inductor combo kernels for that model — add to the serve command:                                                                                                                     
#   --compilation-config '{"inductor_compile_config": {"combo_kernels": false, "benchmark_combo_kernel": false}}'
#   2. The ComboKernels warnings in the compile logs (1 long reduction nodes are separated) suggest inductor is fusing kernels aggressively, which could be the source of the corruption.


APRIEL2_OFFLOAD_INACTIVE_MIXERS=0 APRIEL2_FULL_CUDA_GRAPHS=0 APRIEL2_MIXER_CUDA_GRAPHS=0 nohup /usr/bin/python3 -m place_layers \
    +command=serve \
    +backend=local_vllm \
    +backend.client.max_model_len=65536 \
    +model=apriel2_1p6_frm1p5_distil_annealed_cpt40k \
    host="0.0.0.0" \
    port=8000 > logs/vllm_serve.log 2>&1 &

# --- Experiment: cudagraph_copy_inputs (H5 hypothesis test) ---
# Tests whether CUDA graph input address mismatch causes NCR>1 corruption.
# If outputs become correct, the bug is a tensor address changing between capture and replay.
# APRIEL2_OFFLOAD_INACTIVE_MIXERS=0 APRIEL2_FULL_CUDA_GRAPHS=0 APRIEL2_MIXER_CUDA_GRAPHS=0 nohup /usr/bin/python3 -m place_layers \
#     +command=serve \
#     +backend=local_vllm \
#     +backend.client.max_model_len=65536 \
#     '+backend.client.compilation_config={cudagraph_copy_inputs: true}' \
#     +model=apriel2_1p6_frm1p5_distil_annealed_cpt40k \
#     host="0.0.0.0" \
#     port=8000 > logs/vllm_serve.log 2>&1 &

# apriel2_1p6_frm1p5_distil_annealed_cpt40k

# CUDA_VISIBLE_DEVICES=0 APRIEL2_OFFLOAD_INACTIVE_MIXERS=0 APRIEL2_FULL_CUDA_GRAPHS=0 APRIEL2_MIXER_CUDA_GRAPHS=0 nohup /usr/bin/python3 -m place_layers \
#     +command=serve \
#     +backend=local_vllm \
#     +backend.client.max_model_len=65536 \
#     +model=apriel2_1p6_frm1p5_distil_annealed \
#     host="0.0.0.0" \
#     port=8000 > logs/vllm_serve.log 2>&1 &
#
# APRIEL2_FULL_CUDA_GRAPHS=1 APRIEL2_OFFLOAD_INACTIVE_MIXERS=1 /usr/bin/python3 -m place_layers \
#     +command=serve \
#     +backend=local_vllm \
#     +backend.client.max_model_len=65536 \
#     +model=apriel2_1p6_frm1p5_distil_annealed \
#     host="0.0.0.0" \
#     port=8000 \