# Skill: Academic-Style Plotting

Refer to this document to improve plot aesthetics, enforce academic-quality sensitivity plots, create PDF figures, change plot styling defaults, or replot saved sensitivity results.

## Relevant Files

- `leader_dt/plotting/academic_style.py`
- `leader_dt/plotting/sensitivity_plots.py`
- `scripts/run_sensitivity.py`
- `scripts/replot_sensitivity.py`
- `requirements.txt`

The academic plotting style is not optional. Running `scripts/run_sensitivity.py` should automatically produce academic-style plots without a CLI flag.

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
