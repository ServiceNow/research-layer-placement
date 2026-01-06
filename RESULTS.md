# Evaluation Results

## Apriel2-0.5B-dev on GSM8K

Generative evaluation with greedy decoding on the full GSM8K test set (1319 problems).

| Placement | Accuracy | GDN/KDA Layers |
|-----------|----------|----------------|
| all_attention | 32.25% | 0 |
| every4th_gdn | 14.44% | 6 |
| every4th_kda | 8.42% | 6 |
| every3rd_gdn | 7.52% | 8 |
| every2nd_gdn | 0.28% | 12 |
| every3rd_kda | 0.00% | 8 |
| every2nd_kda | 0.00% | 12 |

### Key Findings

1. **Baseline**: All-attention achieves 32.25% accuracy on GSM8K.

2. **GDN degrades gracefully**: Replacing every 4th layer with GDN retains ~45% of baseline accuracy (14.44% vs 32.25%). Denser replacements cause steeper drops.

3. **KDA performs worse than GDN**: At the same replacement frequency, KDA consistently underperforms GDN (8.42% vs 14.44% at every-4th).

4. **Dense replacement breaks the model**: Replacing every 2nd layer with either GDN or KDA essentially breaks math reasoning (<1% accuracy).

### Placement Patterns

```
all_attention:  [A A A A A A A A A A A A A A A A A A A A A A A A]
every4th_gdn:   [A A A G A A A G A A A G A A A G A A A G A A A G]
every3rd_gdn:   [A A G A A G A A G A A G A A G A A G A A G A A G]
every2nd_gdn:   [A G A G A G A G A G A G A G A G A G A G A G A G]
```

Where: A = attention, G = GDN/KDA

### Commands

```bash
# Reproduce these results
python -m place_layers +experiments=eval_gsm8k_attention
python -m place_layers +experiments=eval_gsm8k_every4th_gdn
python -m place_layers +experiments=eval_gsm8k_every4th_kda
# ... etc
```

### Notes

- Model: `ServiceNow-AI/apriel2-0.5b-dev` (24-layer supernet)
- Task: GSM8K test split (1319 problems)
- Generation: Greedy decoding, max 1024 tokens
- Scoring: Exact match on final numeric answer
