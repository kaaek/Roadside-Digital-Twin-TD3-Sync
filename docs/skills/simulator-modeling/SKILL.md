# Skill: Simulator and Model Changes

Refer to this document when modifying the environment, action decoder, dynamics, state, observation builder, reward, CPU model, communication model, sensor model, or scenario generation.

## Relevant Files

- `leader_dt/simulator/environment.py`
- `leader_dt/simulator/action.py`
- `leader_dt/simulator/dynamics.py`
- `leader_dt/simulator/state.py`
- `leader_dt/models/aoi.py`
- `leader_dt/models/accuracy.py`
- `leader_dt/models/communication.py`
- `leader_dt/models/cpu.py`
- `leader_dt/models/objective.py`
- `leader_dt/domain/scenario.py`
- `leader_dt/domain/sensor_pairs.py`
- `leader_dt/rl/observation.py`
- `leader_dt/rl/reward.py`

## Core Invariants

- RL scores provider pairs, not sensor types alone.
- Sensor-type AoI/freshness is the main Digital Twin objective.
- Pair-level constraints handle data, communication, CPU, and accuracy.
- One pair is scheduled per slot.
- Stable action and observation shapes must be preserved across sweeps.
- Evaluation should use deterministic action decoding.

## Unit Conventions

- Time: seconds or slots.
- CPU frequency: cycles/second.
- CPU backlog and CPU cost: cycles.
- Data size: bits.
- Bandwidth: Hz.
- AoI/freshness: slots.
- Accuracy: fraction in `[0, 1]`.

## Validation Commands

```bash
python -m compileall -q leader_dt scripts tests
pytest -q tests/test_environment.py
```
