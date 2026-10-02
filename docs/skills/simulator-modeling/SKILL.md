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
- A pair is feasible only if its carrier is in Zone B and it holds a pending sample. Read data sizes through `LeaderSynchronizationDynamics.get_available_data_size_bits_array`, not `scenario.available_data_size_bits_matrix`, which holds sizes of samples at arrival time.
- Each pair keeps only its latest unsent sample; an upload consumes it. On a successful refresh, AoI = sample age + sensing delay + transmission delay.
- With `sample_arrival_rate_per_slot=None` the simulator must reproduce the legacy model (fresh sample every slot) exactly; arrival counts are drawn after sample sizes to keep the legacy random stream.
- Stable action and observation shapes must be preserved across sweeps. The observation has a sample-age block only when the arrival process is enabled (1284 vs. 964 values), so do not mix legacy and arrival-mode models.
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
