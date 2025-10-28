# Analysis Report Figures

This directory contains key figures used in `ANALYSIS_REPORT.md`.

## Figure Inventory

### Mock Evaluator Validation (Section 4.2)

**fig1_landscape_distribution.png**
- Source: `outputs/command=landscape_analysis,evaluator=small,problem=small/`
- Shows: Score distribution across all 29,070 configurations (exhaustive enumeration)
- Key insight: Near-optimal solutions are extremely sparse (0.05% within 10% of optimum)

**fig2_percentile_analysis.png**
- Source: `outputs/command=landscape_analysis,evaluator=small,problem=small/`
- Shows: Beam search performance vs ground truth (99.98th percentile)
- Key insight: Systematic search vastly outperforms random sampling

### Beam Search Performance (Section 4.1)

**fig3_convergence_analysis.png**
- Source: `outputs/command=convergence_analysis,evaluator=apriel_h_1_5_15b,problem=apriel_h_1_5_15b/`
- Shows: Convergence trajectories across 1000 random seeds (Apriel 15B)
- Key insight: Consistent convergence despite 16% score variability

**fig4_convergence_rate.png**
- Source: `outputs/command=convergence_analysis,evaluator=apriel_h_1_5_15b,problem=apriel_h_1_5_15b/`
- Shows: Convergence rate across model sizes
- Key insight: All models reach 90% at ~80-85% of evaluations

### Landscape Heterogeneity (Section 4.5)

**fig5_heatmap_seed42_easy.png**
- Source: `outputs/command=hyperparameter_analysis,evaluator=apriel_h_1_5_15b,problem=apriel_h_1_5_15b/` (seed=42)
- Shows: Hyperparameter sensitivity on easy landscape
- Key insight: Greedy search (width=2) succeeds - strong gradient structure

**fig6_heatmap_seed200_hard.png**
- Source: `outputs/command=hyperparameter_analysis,evaluator.seed=200,evaluator=apriel_h_1_5_15b,problem=apriel_h_1_5_15b/`
- Shows: Hyperparameter sensitivity on hard landscape
- Key insight: Requires width≥12 - narrow beams fail catastrophically (16% worse)

**fig7_heatmap_seed300_medium.png**
- Source: `outputs/command=hyperparameter_analysis,evaluator.seed=300,evaluator=apriel_h_1_5_15b,problem=apriel_h_1_5_15b/`
- Shows: Hyperparameter sensitivity on medium landscape
- Key insight: Width=3 sufficient - illustrates no universal optimum exists

## Regenerating Figures

To regenerate any figure, run the corresponding command from the project root:

```bash
# Landscape analysis (figs 1-2)
python -m place_layers command=landscape_analysis evaluator=small problem=small

# Convergence analysis (figs 3-4)
python -m place_layers command=convergence_analysis evaluator=apriel_h_1_5_15b problem=apriel_h_1_5_15b

# Hyperparameter analysis (figs 5-7)
python -m place_layers command=hyperparameter_analysis evaluator=apriel_h_1_5_15b problem=apriel_h_1_5_15b
python -m place_layers command=hyperparameter_analysis evaluator=apriel_h_1_5_15b problem=apriel_h_1_5_15b evaluator.seed=200
python -m place_layers command=hyperparameter_analysis evaluator=apriel_h_1_5_15b problem=apriel_h_1_5_15b evaluator.seed=300
```

Then copy from `outputs/` to `assets/` using the naming convention above.
