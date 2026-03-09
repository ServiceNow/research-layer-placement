#!/bin/bash
# Test AIME problems 2 and 3 via curl to check for NaN/truncation issues
MODEL="/mnt/checkpoints/fast_llm_exp/apriel_train/ap1p6_dist_dmx1p5_i5anl1_lr1e-05_lfmse0.5_rkl0.9_fkl0.1_sl16384_iters10000/export/apriel2/10000_surogate_search_lieklihood_6.4cost_placement"
MAX_TOKENS="${1:-15000}"

echo "Using max_tokens=$MAX_TOKENS"
echo ""

for PROB in 2 3; do
    MESSAGES=$(cat /tmp/aime_problem_${PROB}.json)
    PAYLOAD=$(python3 -c "
import json
messages = json.load(open('/tmp/aime_problem_${PROB}.json'))
payload = {
    'model': '$MODEL',
    'messages': messages,
    'max_tokens': $MAX_TOKENS,
    'temperature': 0,
    'top_p': 1.0,
}
print(json.dumps(payload))
")

    echo "=== Problem $PROB ==="
    RESPONSE=$(curl -s http://0.0.0.0:8000/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "$PAYLOAD")

    # Extract key fields
    echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
choice = data['choices'][0]
content = choice['message']['content']
finish = choice['finish_reason']
usage = data['usage']
print(f'  finish_reason: {finish}')
print(f'  completion_tokens: {usage[\"completion_tokens\"]}')
print(f'  content length: {len(content)} chars')
print(f'  content preview: {content[:200]!r}')
print(f'  content tail: {content[-100:]!r}' if len(content) > 100 else '')
"
    echo ""
done
