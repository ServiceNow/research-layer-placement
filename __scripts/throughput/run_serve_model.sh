make run-vllm-serve MODEL=/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement/ \
     TOKENIZER=/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement/ \
     MAX_MODEL_LEN=65536 GPU_MEM_UTILIZATION=0.9 TP=1
