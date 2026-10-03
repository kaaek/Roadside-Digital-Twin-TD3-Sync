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
- `leader_dt/rl/pair_features.py`
- `leader_dt/rl/reward.py`

## Core Invariants

- RL schedules provider pairs, not sensor types alone. The actor outputs one weight per pair feature (`leader_dt/rl/pair_features.py`) plus the requested accuracy; every feasible pair is scored with the same weights. Keep features normalized and computed identically for every pair, and never give the actor per-slot outputs: slot identities change every episode.
- Heuristic baselines return `PairSchedulingRequest`, never RL action vectors.
- Sensor-type AoI/freshness is the main Digital Twin objective.
- Pair-level constraints handle data, communication, CPU, and accuracy.
- One pair is scheduled per slot.
- A pair is feasible only if its carrier is in Zone B and it holds a pending sample. Read data sizes through `LeaderSynchronizationDynamics.get_available_data_size_bits_array`, not `scenario.available_data_size_bits_matrix`, which holds sizes of samples at arrival time.
- Each pair keeps only its latest unsent sample; an upload consumes it. On a successful refresh, AoI = sample age + sensing delay + transmission delay.
- With `sample_arrival_rate_per_slot=None` every pair has a fresh sample in every slot. Arrival counts are drawn after sample sizes, so a seed produces the same sizes at every arrival rate.
- Stable action and observation shapes must be preserved across sweeps: a 7-value action and an 80-value observation for any vehicle count, sensor-type count (up to 16), or arrival rate. Adding a pair feature changes both shapes and requires retraining.
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
.venv/bin/python -m compileall -q leader_dt scripts tests
.venv/bin/python -m pytest -q tests/test_environment.py
```
