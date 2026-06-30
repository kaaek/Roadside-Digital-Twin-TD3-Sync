# Skill: Z3 Feasibility Diagnostics

## Trigger

Use this skill when the user asks about feasibility, constraint consistency, pair-level vs sensor-type-level freshness, or Z3 results.

## Relevant Files

- `leader_dt/verification/z3_feasibility.py`
- `scripts/check_z3_feasibility.py`
- `tests/test_z3_feasibility.py`

## Command Template

```bash
python scripts/check_z3_feasibility.py \
  --vehicle-count 40 \
  --sensors-per-vehicle 4 \
  --time-horizon-slots 40 \
  --freshness-threshold-slots 10 \
  --timeout-ms 60000
```

## Interpretation Rules

- Pair-level freshness is structurally too strict in the default setting because there can be up to `160` provider-sensor pairs and only one scheduled pair per slot.
- Sensor-type-level freshness is the corrected formulation because the Digital Twin needs fresh sensor-type information, not every vehicle-sensor pair refreshed within the threshold.
- Z3 `unknown` under full constraints does not prove infeasibility; inspect relaxed constraints and diagnostic counts.
- If removing freshness constraints gives `sat`, freshness is likely the binding constraint.

## Known Diagnostic Pattern

A prior diagnostic showed:

- Pair count around `160`.
- Sensor type count `8`.
- Freshness threshold `10`.
- Pair-level persistent capacity violation was true.
- Sensor-type-level capacity violation was false.

This supports maintaining pair-level scheduling with sensor-type-level AoI/freshness.

## Validation

```bash
pytest -q tests/test_z3_feasibility.py
```
