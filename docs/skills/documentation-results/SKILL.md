# Skill: Documentation and Results Reporting

## Trigger

Use this skill when the user asks for README updates, thesis-style explanations, work logs, professor emails, report text, result summaries, or artifact/documentation organization.

## Relevant Files

- `README.md`
- `Context.md`
- `Instruction.md`
- `skills/*/SKILL.md`
- `results/**/monte_carlo_metrics.json`
- `results/**/td3_convergence_history.json`
- `results/**/td3_convergence_training_report.json`
- `leader_dt/evaluation/reporting.py`

## Reporting Style

Use clear, technical, evidence-based language.

Preferred wording:

```text
TD3 learns a non-random CPU-aware scheduling policy, but under the tested formulation it does not outperform Greedy on the primary AoI/freshness metrics. Its main advantage is lower CPU backlog, while Greedy remains stronger in freshness preservation.
```

Avoid overclaiming:

```text
TD3 is optimal.
TD3 converged perfectly.
Greedy is unfair so results do not matter.
```

## README Rules

The user-facing `README.md` should include:

1. Project title.
2. Two-sentence description.
3. Installation.
4. Usage commands for training, Monte Carlo, and sensitivity sweeps.
5. MIT License.

Keep `README.md` concise but command-complete.

## Agent Context Documentation Rules

- Put stable facts in `Context.md`.
- Put always-loaded coding and research rules in `Instruction.md`.
- Put task-specific operational knowledge in `skills/<task>/SKILL.md`.
- Do not duplicate very long command lists in every skill; use task-specific commands only.
- Update skill files when new scripts or CLI flags are added.

## Work Log Style

Use concise entries of this form:

```text
Refactored the TD3 experiment pipeline by adding configurable target-policy smoothing, smaller action-noise controls, larger replay-buffer/batch-size options, TensorBoard/Monitor logging, and periodic checkpoint saving.
```

## Results Summary Checklist

When summarizing a run, include:

- Training timesteps.
- Number of seeds.
- Evaluation frequency.
- Evaluation episode count.
- Best checkpoint selection criterion.
- Monte Carlo trial count.
- TD3 vs Greedy on AoI/freshness.
- TD3 vs Greedy on CPU backlog.
- Whether reward visually plateaued.
