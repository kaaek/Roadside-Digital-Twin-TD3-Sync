# Instruction: Always-Loaded Project Rules for `td3-rsu`

## Primary Rule

Preserve the research model and modular codebase. Do not simplify the simulator, TD3 action semantics, CPU model, or evaluation methodology unless the user explicitly requests a modeling change.

## Coding Standards

1. Use Python `>=3.10` syntax.
2. Use clear snake_case names for variables, functions, and modules.
3. Use PascalCase for classes and dataclasses.
4. Add or update docstrings for every new module, public class, and public method.
5. Prefer type hints for function arguments and return values.
6. Keep configuration values in `leader_dt/constants.py` and `leader_dt/config.py`; do not scatter hardcoded experiment values across modules.
7. Keep CLI overrides in scripts when a value is likely to vary experimentally.
8. Keep simulator logic separate from RL training logic.
9. Keep plotting/reporting separate from metric computation.
10. Avoid large notebook-only changes; implement reusable code in the package and call it from scripts.

## Modeling Rules

1. The TD3 action is pair-level: one score per padded `(vehicle, sensor_type)` pair plus one requested accuracy fraction.
2. AoI/freshness/objective are sensor-type-level.
3. Communication, data availability, accuracy, and CPU cost are pair-level.
4. One feasible pair is scheduled per time slot.
5. Do not revert to the older sensor-type-only action model unless explicitly asked.
6. Do not count freshness at pair level; pair-level freshness is structurally too strict for the default `40`-slot/`160`-pair setting.
7. Preserve unit consistency. CPU backlog and CPU cost are measured in cycles; normalized CPU backlog divides cycles by per-slot CPU capacity.
8. Preserve deterministic seed controls when writing experiments.
9. Keep action-space padding stable across vehicle-count sweeps by using `max_vehicle_count_for_action_space` and `max_sensors_per_vehicle_for_action_space`.

## TD3 Rules

1. Use Stable-Baselines3 `TD3` through `leader_dt/rl/td3_agent.py`.
2. Configure action noise through `Td3TrainingConfig.action_noise_sigma`.
3. Configure target policy smoothing through `target_policy_noise` and `target_noise_clip`.
4. Keep TensorBoard, Monitor, and checkpoint settings configurable.
5. For tuning experiments, prefer small action-noise tests such as `0.05` and `0.10` rather than returning to `0.20` by default.
6. Use larger replay buffers and batch sizes for tuned runs when compute permits: `buffer_size=1_000_000`, `batch_size=256` or `512`.
7. Do not interpret jagged reward as complete failure without checking moving averages, best-so-far reward, AoI, freshness, CPU backlog, and Monte Carlo results.
8. Do not claim visual convergence unless reward and/or objective curves plateau under smoothed and multi-seed diagnostics.

## Baseline Rules

1. Keep Greedy as a strong heuristic baseline.
2. CPU-aware Greedy score must use:

   ```text
   score(i,t) = priority_weight(i) * AoI(i,t)
                - lambda_cpu * max(0, current_backlog + estimated_CPU_cycles(i,t) - slot_CPU_capacity) / slot_CPU_capacity
   ```

3. `lambda_cpu` is a weighted-score penalty coefficient, not a physical unit.
4. Estimated CPU cycles must be derived from requested/collected bits times CPU cycles per bit.
5. Greedy-specific parameters should be CLI overrides and constants, not sensitivity-sweep environment variables.
6. Preserve the ability to recover old CPU-blind Greedy by setting `lambda_cpu=0` and requested accuracy to `1.0`.

## Evaluation Rules

1. Use Monte Carlo evaluation for final claims.
2. Use the same seed ranges when comparing policies.
3. Always report TD3 against relevant baselines: Greedy, Random, and No-refresh when available.
4. Report at least AoI, freshness violations, terminal CPU violations, final CPU backlog, total collected bits, reward, and penalized score.
5. Do not rely only on TD3 training reward curves for conclusions.
6. For final comparison, prefer `500` or `1000` trials when compute permits.
7. For smoke tests, `50` to `100` trials is acceptable.

## Sensitivity Rules

1. Sweep one environment parameter at a time.
2. Keep TD3 model fixed during a sensitivity sweep unless the experiment explicitly studies retraining.
3. Save outputs in parameter-specific directories.
4. Useful sweeps include:
   - `vehicle_count`
   - `zone_size_meter`
   - `freshness_threshold_slots`
   - `data_size_high_multiplier`
   - `vehicle_speed_meter_per_second`
   - `sensors_per_vehicle`
   - `uplink_bandwidth_hz`
   - `accuracy_threshold`
   - `time_horizon_slots`
5. Do not present Greedy `lambda_cpu` as an environment sensitivity parameter; it is a baseline design parameter.

## CLI and Script Rules

1. When adding a new experiment parameter, add it in this order:
   - `leader_dt/constants.py`
   - `leader_dt/config.py`
   - relevant script CLI parser
   - report JSON/config output when applicable
2. Preserve backwards-compatible script defaults when possible.
3. Every script should have a module-level docstring and a concise `main()` docstring.
4. Write generated outputs to `results/` by default.
5. Avoid overwriting unrelated output directories.

## Testing and Validation Rules

After code changes, run:

```bash
python -m compileall -q leader_dt scripts tests
pytest -q
```

For CLI-level smoke validation, run at least:

```bash
python scripts/train_td3_until_convergence.py --maximum-timesteps 200000 --minimum-timesteps 100000 --eval-frequency-steps 50000 --evaluation-episodes 10 --output-dir results/smoke_test
python scripts/run_monte_carlo.py --model-path results/smoke_test/models/best_td3_exact_pair_zone_b.zip --trials 50 --output-dir results/smoke_test/final_monte_carlo
python scripts/run_sensitivity.py --parameter data_size_high_multiplier --values 1.0,2.0 --trials 25 --model-path results/smoke_test/models/best_td3_exact_pair_zone_b.zip --output-dir results/smoke_test/sensitivity_data
```

If dependencies are unavailable in the current runtime, state that only compile/static checks were run and do not claim full execution.

## Documentation Rules

1. Keep `README.md` user-facing and command-oriented.
2. Keep `Context.md` agent-facing and factual.
3. Keep `Instruction.md` rule-based and concise.
4. Keep `SKILL.md` files task-specific and triggerable.
5. Update documentation when changing CLI flags, defaults, result paths, model assumptions, or baseline definitions.
6. Use inline code formatting for paths, commands, classes, and functions.

## Research Reporting Rules

1. Be honest about TD3 performance.
2. Distinguish between “trained extensively,” “stabilized in a noisy band,” and “converged.”
3. Mention that Greedy can outperform TD3 on AoI/freshness while TD3 can reduce CPU backlog.
4. Do not present one seed as robust proof.
5. Use multi-seed summaries when available.
6. Explain reward erraticness using stochastic evaluation and penalty-heavy reward terms, but do not use that as proof of convergence.
