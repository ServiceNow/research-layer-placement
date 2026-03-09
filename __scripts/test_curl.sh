#!/bin/bash
MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement"

echo "=== Test 1: with logprobs:null ==="
curl -s http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100,\"temperature\":0,\"top_p\":1.0,\"logprobs\":null}" | python -m json.tool

echo ""
echo "=== Test 2: with logprobs:null + return_token_ids:true ==="
curl -s http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100,\"temperature\":0,\"top_p\":1.0,\"logprobs\":null,\"return_token_ids\":true}" | python -m json.tool

echo ""
echo "=== Test 3: without logprobs field (baseline) ==="
curl -s http://0.0.0.0:8000/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2?\"}],\"max_tokens\":100,\"temperature\":0,\"top_p\":1.0}" | python -m json.tool
