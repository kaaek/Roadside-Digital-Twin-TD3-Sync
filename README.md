# Leader-Assisted DT Synchronization for RSU Failure: Simulator

<!--
This package implements the System Model and Problem Formulation from the paper
around a Zone B defective-RSU simulation.


## Key modeling decisions

- The scheduled entity is the exact sensor-vehicle pair `i ∈ I`.
- Available data size `δ_i(t)` is sampled per pair and per slot.
- Vehicles physically move through Zone B every slot.
- Only Zone B is modeled in code.
- The leader vehicle is currently included as a possible sensor provider.
- TD3 receives penalties for freshness, accuracy, and terminal CPU violations;
  no safety override is applied.
- MILP is excluded from the active execution path for now and kept as a future
  interface only.

-->

## Run

```bash
pip install -e .
python scripts/run_smoke_tests.py
```

## Main package layers

- `domain/`: vehicles, sensor types, exact pair indexing, scenario generation.
- `models/`: communication, AoI, CPU, accuracy, and weighted objective equations.
- `simulator/`: state, action decoding, transition dynamics, Gym environment.
- `baselines/`: greedy/no-refresh/random policies.
- `evaluation/`: rollout and Monte Carlo interfaces.
- `rl/`: TD3 wrappers and training helpers.
