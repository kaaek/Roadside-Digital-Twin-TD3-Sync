# Skill: Thesis-Style Plotting

Refer to this document to improve plot aesthetics, enforce thesis-quality sensitivity plots, create PDF figures, change plot styling defaults, or replot saved sensitivity results.

## Relevant Files

- `leader_dt/plotting/thesis_style.py`
- `leader_dt/plotting/sensitivity_plots.py`
- `scripts/run_sensitivity.py`
- `scripts/replot_sensitivity.py`
- `requirements.txt`

The thesis plotting style is not optional. Running `scripts/run_sensitivity.py` should automatically produce thesis-style plots without a CLI flag.

## Required Dependency

```bash
pip install SciencePlots
grep -q "SciencePlots" requirements.txt || echo "SciencePlots>=2.1" >> requirements.txt
```

## Mandatory Style Features

- `plt.style.use(["science", "grid"])`.
- Serif fonts with `text.usetex=False` unless a full LaTeX install is available and explicitly enabled.
- Colorblind-friendly palette.
- Distinct markers and linestyles for Greedy, TD3, and PPO.
- Tight bounding boxes in `savefig`.
- PDF export beside PNG export.
- Human-readable axis labels.

## Validation Command

```bash
python scripts/run_sensitivity.py \
  --parameter accuracy_threshold \
  --values 0.4,0.6,0.8,1.0 \
  --trials 10 \
  --seed-start 50000 \
  --td3-model-path results/td3_zone_2km_3m_seed1_gpu/models/best_td3_exact_pair_zone_b.zip \
  --ppo-model-path results/ppo_zone_2km_3m_seed1_cpu/models/best_ppo_exact_pair_zone_b.zip \
  --output-dir results/test_thesis_style_sensitivity
```

Expected outputs:

```text
results/test_thesis_style_sensitivity/plots/sensitivity_accuracy_threshold.png
results/test_thesis_style_sensitivity/plots/sensitivity_accuracy_threshold.pdf
```
