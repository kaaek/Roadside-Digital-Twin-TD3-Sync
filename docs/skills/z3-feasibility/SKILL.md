# Skill: Z3 Feasibility Diagnostics

Refer to this document to learn about feasibility, constraint consistency, pair-level vs sensor-type-level freshness, or Z3 diagnostics.

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

- Z3 `unknown` does not prove infeasibility.
- If removing freshness gives `sat`, freshness is likely the binding constraint.
- Pair-level freshness is structurally too strict when many provider pairs exist but only one can be scheduled per slot.
- Sensor-type-level freshness matches the Digital Twin objective: keep sensor information fresh, not every provider pair fresh.
