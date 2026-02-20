#!/usr/bin/env bash
# Serve an extracted Apriel2 checkpoint (att12_gdn36) with vLLM.
set -euo pipefail

MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_random_placements/att12_gdn36"

# /usr/local/bin/vllm serve "$MODEL" \
#     --trust-remote-code \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 \
#     --host 0.0.0.0 \
#     --port 8000



APRIEL2_MIXER_CUDA_GRAPHS=1 /usr/bin/python3 -m place_layers \
    +command=serve \
    +backend=local_vllm \
    +backend.client.max_model_len=65536 \
    +model=apriel2_1p6_frm1p5_distil_annealed \
    host="0.0.0.0" \
    port=8000 \