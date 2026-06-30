# Skill: Simulator and Model Changes

## Trigger

Use this skill when modifying the environment, dynamics, action decoder, observation builder, reward, CPU model, communication model, sensor model, or scenario generation.

## Relevant Files

- `leader_dt/simulator/environment.py`
- `leader_dt/simulator/action.py`
- `leader_dt/simulator/dynamics.py`
- `leader_dt/simulator/state.py`
- `leader_dt/simulator/recorder.py`
- `leader_dt/models/aoi.py`
- `leader_dt/models/accuracy.py`
- `leader_dt/models/communication.py`
- `leader_dt/models/cpu.py`
- `leader_dt/models/objective.py`
- `leader_dt/domain/scenario.py`
- `leader_dt/domain/sensor_pairs.py`
- `leader_dt/domain/topology.py`
- `leader_dt/rl/observation.py`
- `leader_dt/rl/reward.py`

## Core Invariants

1. TD3 selects pair-level scores.
2. AoI/freshness/objective remain sensor-type-level.
3. CPU, data, accuracy, and communication constraints remain pair-level.
4. One pair is scheduled per slot.
5. The episode horizon is finite and defaults to `40` slots.
6. The action and observation spaces must keep stable shapes across sweeps.
7. Do not silently change default physical units.

## Unit Conventions

- Time slot duration: seconds.
- CPU frequency: cycles/second.
- CPU backlog: cycles.
- CPU cost: cycles.
- Data size: bits.
- Uplink bandwidth: Hz.
- AoI/freshness: slots.
- Accuracy: fraction in `[0, 1]`.

## Implementation Checklist

Before changing simulator logic:

1. Identify whether the change is pair-level or sensor-type-level.
2. Check if a config value already exists.
3. Add new constants to `leader_dt/constants.py` only if needed.
4. Add dataclass fields to `leader_dt/config.py` for configurable parameters.
5. Preserve deterministic seed behavior.
6. Update tests if behavior changes.
7. Run compile and tests.

## Validation Commands

```bash
python -m compileall -q leader_dt scripts tests
pytest -q tests/test_environment.py
pytest -q tests/test_penalized_objective.py
```

## Common Pitfalls

- Accidentally counting freshness violations at pair level.
- Breaking action-space dimensions during vehicle-count sweeps.
- Mixing cycles and bits in CPU formulas.
- Adding randomness without using the configured random seed.
- Updating simulator behavior without updating Monte Carlo and sensitivity reports.
