# Context: TD3-RSU Leader-Assisted Digital Twin Synchronization

## Purpose

This document explains the current research project at a high level, it is not intended as a developer manual. It explains the system model, the decision process, the learning formulation, the heuristic baselines, the evaluation protocol, and an analysis of the current results.

---

## 1. Research Problem Summary

Vehicular Digital Twins require frequent updates from physical vehicles and road-side infrastructure in order to maintain a fresh virtual representation of traffic, vehicle states, and sensing conditions. A Roadside Unit (RSU) normally helps collect and forward these updates. When an RSU fails, the Digital Twin may continue to receive stale or incomplete information, especially if there is no temporary replacement mechanism.

This project studies a recovery strategy in which a selected leader vehicle temporarily substitutes for a defective RSU inside the failed coverage zone, referred to as Zone B (Zone A precedes the defective zone B, and zone C succeeds it). In zone B, the leader vehicle collects sensor data from nearby vehicles and forwards the collected updates to preserve the freshness of the Digital Twin.

The research question is:

> How should the leader vehicle schedule sensor updates from surrounding vehicles under mobility, bandwidth, CPU, accuracy, and freshness constraints?

The project compares four policy families: TD3, PPO, CPU-aware Greedy, and proximity greedy.

The purpose is to evaluate whether learned policies can make resource-aware scheduling decisions in a stochastic vehicular environment, and to compare them against heuristics.

---

## 2. System-Level View

The system is modeled as a slotted decision process. Time is divided into discrete slots. At each slot, the leader vehicle observes the current episode state and chooses one provider-sensor pair to serve.

A provider-sensor pair means:

$$
i = (v, s)
$$

where (v) is a vehicle and (s) is a sensor type carried by that vehicle.

This distinction is important. Scheduling is pair-level i.e. the leader chooses a specific vehicle-sensor pair, whereas freshness is sensor-type-level i.e. the Digital Twin tracks the freshness of each sensor type, not every individual vehicle-sensor pair separately.

For example, if several vehicles carry a radar sensor, the leader may choose any feasible radar-providing vehicle. A successful radar update refreshes the Digital Twin’s radar information, regardless of which vehicle supplied it.

At every time slot, the simulator performs the following sequence:

1. Identify which vehicles are currently inside the defective zone.
2. Determine which provider-sensor pairs are feasible: the carrier vehicle must be inside Zone B and the pair must hold a pending sensor sample.
3. Build the observation seen by TD3/PPO.
4. Let the selected policy choose a pair and an accuracy request.
5. Compute the achievable communication rate from the 3GPP TR 37.885 highway pathloss, the slot's shadowing, and Shannon capacity.
6. Determine how much data is actually collected.
7. Check whether the achieved accuracy satisfies the threshold.
8. Update the sensor-type Age of Information / Age of Digital Twin.
9. Update CPU backlog and resource usage.
10. Advance vehicle positions, deliver the sensor samples generated for the next slot, and move to the next slot.

---

## 3. Main Components

### 3.1 Domain Layer: Vehicles, Sensors, and Feasible Pairs

The domain layer defines the scenario entities:

* Vehicles moving through the defective RSU zone.
* Sensor types available in the system.
* Sensor assignments to vehicles.
* Road and zone geometry.
* Provider-sensor pairs that can be scheduled.
* Whether the leader itself is allowed to act as a provider.

The current nominal configuration uses:

* 20 vehicles.
* 8 active sensor types.
* 4 sensors per vehicle.
* 40 time slots.
* A defective zone from 0 m to 2000 m.
* A 1-second slot duration.
* Sample sizes drawn uniformly between 0.5× and 2.0× each sensor type's nominal payload.
* All vehicles, the leader included, drive at speeds drawn from N(23, 4) m/s; the leader starts at the zone entry.
* A sample arrival rate of $\lambda = 0.10$ new samples per pair per slot (Section 3.2).

With 20 vehicles and 4 sensors per vehicle, the nominal number of real provider-sensor pairs is:

$$
20 \times 4 = 80
$$

The reinforcement learning interface does not depend on the number of pairs (Section 5), so the same trained network can be evaluated at any vehicle count.

---

### 3.2 Physical and Resource Models

The model layer converts the scheduling decisions described above into physical and computational consequences/effects.

#### Communication Model

The current system does not use a fixed number of transmitted bits per slot. Instead, the available uplink rate depends on the distance between the selected provider vehicle and the leader vehicle.

The communication model follows a Shannon-style capacity expression:

$$
R_i(t) = B \log_2(1 + \mathrm{SNR}_i(t))
$$

where:

* $R_i(t)$ is the uplink rate for pair $i$ at time slot $t$.
* $B$ is the uplink bandwidth.
* $\mathrm{SNR}_i(t) = P\,10^{(-\mathrm{PL}(d) + X_v(t))/10} / (N_0 B)$.

The pathloss follows the 3GPP TR 37.885 highway line-of-sight model at $f_c = 5.9$ GHz:

$$
\mathrm{PL}(d) = 32.4 + 20\log_{10} f_c[\mathrm{GHz}] + 20\log_{10} d[\mathrm{m}] \quad (\mathrm{dB})
$$

$X_v(t) \sim \mathcal{N}(0, 3^2)$ dB is log-normal shadowing, drawn per vehicle and slot (i.i.d.; spatial correlation is ignored) and shared by all sensors of that vehicle. The nominal link uses $P = 23$ dBm, a noise PSD of $-174$ dBm/Hz plus a 9 dB noise figure, and $B = 1.8$ MHz, i.e. one LTE-V2X sub-channel (10 resource blocks) of the shared 10 MHz ITS channel. Without shadowing this gives about 22.5 Mbit/s at 100 m, 10.6 Mbit/s at 1 km and 7.1 Mbit/s at 2 km, so a 10 Mbit LiDAR cloud may not reach the 0.8 accuracy threshold in one slot from far away.

As distance increases, pathloss increases, SNR decreases, and the achievable data rate decreases. A far vehicle may carry an urgent sensor, but if the communication channel is weak, the leader may collect too little data to meet the required accuracy. In that case, the update may fail to refresh the Digital Twin.

#### Sensor Sample Generation

Sensors do not have fresh data at every slot. Each provider-sensor pair generates new samples according to a Poisson process:

$$
N_i(t) \sim \mathrm{Poisson}(\lambda)
$$

where $N_i(t)$ is the number of samples pair $i$ generates in slot $t$. A pair keeps only its latest unsent sample: a newer sample overwrites an older one, and an upload consumes the pending sample whether or not it meets the accuracy threshold. A pair is schedulable only while it holds a pending sample, so in a given slot it has a fresh sample with probability

$$
1 - e^{-\lambda}
$$

With the nominal $\lambda = 0.10$, each pair produces a new sample in about 9.5% of slots, roughly one every 10 seconds. Across the 80 nominal pairs, a few new samples still appear in the zone every slot, but a particular sensor on a particular vehicle usually has nothing new to offer. This makes data availability, not only geometry, a scheduling constraint. Waiting is also costly, because a pending sample keeps ageing until it is uploaded.

Each sample's size $\delta_i(t)$ is drawn uniformly between 0.5× and 2.0× its sensor type's nominal payload. Nominal payloads are one digital-twin update: a compressed camera frame (3 Mbit) or LiDAR point cloud (10 Mbit), a radar object list (0.4 Mbit), or a short telemetry message (0.25–16 kbit). Arrivals and sizes are pre-generated per episode from the scenario seed. Setting the rate to `None` gives every pair a fresh sample in every slot (no arrival process).

The rate is set by `DEFAULT_SAMPLE_ARRIVAL_RATE_PER_SLOT` in `leader_dt/constants.py`, by `--sample-arrival-rate` on the convergence training scripts, or swept with the `sample_arrival_rate` sensitivity parameter.

#### Data Size and Accuracy

Each sensor type has an associated data size. The policy does not simply decide whether to update or not, but it also enforces a data quality fraction $\eta$.

The achieved accuracy is determined by the ratio:

$$
\alpha_i(t) = \frac{\text{collected bits from pair } i}{\text{size of pair } i\text{'s pending sample}}
$$

The update is considered feasible only if:

$$
\alpha_i(t) \geq \alpha_{\min}
$$

where $\alpha_{\min}$ is the minimum required accuracy threshold.

The current nominal threshold is:

$$
\alpha_{\min} = 0.80
$$

This means that collecting a small fraction of the available data is not always useful. The leader may spend communication and CPU resources but still fail to produce an acceptable Digital Twin update.

#### CPU Backlog

The leader vehicle has limited processing capacity. Collected data creates CPU work, and this work accumulates as backlog if it cannot be processed immediately.

For a selected provider-sensor pair (i), the processing demand is modeled as:

$$
C_i(t) = b_i(t)\rho_i
$$

where:

* $C_i(t)$ is the added CPU work.
* $b_i(t)$ is the number of collected bits.
* $\rho_i$ is the CPU cycles per bit for the selected sensor type: hundreds for perception data (camera 500, LiDAR 800, radar 100), tens for telemetry, following common MEC values.

At each slot, the leader can process a limited number of CPU cycles:

$$
C_{\text{slot}} = f_{\text{CPU}} \Delta t
$$

where:

* $f_{\text{CPU}}$ is the leader CPU frequency.
* $\Delta t$ is the slot duration.

The current nominal CPU frequency is:

$$
f_{\text{CPU}} = 2.0 \times 10^9 \text{ cycles/s}
$$

with $\Delta t = 1 \text{ s}$, i.e. one 2 GHz application core of an automotive SoC. A nominal camera frame (3 Mbit × 500 cycles/bit) takes 0.75 s of CPU, and a LiDAR point cloud about 4 s, so heavy perception uploads build a backlog.

CPU backlog is important because a policy that aggressively collects high-volume sensor data may reduce AoI in the short term but leave excessive processing work unfinished. This creates a tradeoff between freshness improvement and computational feasibility.

---

## 4. Simulator Layer: Episode Evolution

The simulator is the episode engine. It stores the current state of the system and updates it after every scheduling decision.

The state includes:

* Current time slot.
* Vehicle positions.
* Active vehicles inside Zone B.
* Feasible provider-sensor pairs.
* Sensor-type AoI values.
* Each pair's pending sensor sample: its size and the slot it was generated in.
* CPU backlog.
* Recent CPU usage.
* Accuracy and freshness status.

At each step, the simulator receives a decision from a policy. TD3 and PPO send an action vector (Section 5.2); heuristic baselines name a pair directly with a `PairSchedulingRequest` (pair index and requested accuracy). Either way, the simulator ends up with one feasible pair or an idle slot. It then computes how many bits can be collected from that pair, whether the accuracy threshold is met, and whether the corresponding sensor-type AoI should be refreshed.

If a sensor type is successfully refreshed, its AoI resets to the age of the uploaded sample plus the update delay:

$$
A_s(t+1) = \underbrace{(t - g_i)}_{\text{sample age}} + d^{\text{sense}}_s + d^{\text{tx}}_i(t)
$$

where $g_i$ is the slot in which pair $i$'s pending sample was generated. Uploading a sample that has waited in the buffer therefore refreshes the Digital Twin less than uploading a fresh one. If a sensor type is not refreshed, its AoI increases by one slot.

The simulator also updates the CPU backlog: data collection increases CPU work, per-slot processing reduces backlog, unprocessed work remains as backlog into future slots.

The output of each step is then used to compute both: the training reward used by TD3/PPO, and evaluation metrics used for reporting and comparison.

---

## 5. Reinforcement Learning Formulation

The project formulates leader-assisted Digital Twin synchronization as a continuous-control reinforcement learning problem.

Section 9.1 explains why the policy weights pair features instead of scoring each pair slot directly.

### 5.1 Pair Features

Every vehicle-sensor pair $i$ is described by the same six features, computed in `leader_dt/rl/pair_features.py`:

| Feature | Definition |
|---|---|
| Weighted urgency | $w_s A_s(t) / (\max_s w_s \cdot \tau)$, clipped to $[0, 3]$ |
| Reachable accuracy | $\min(1, \text{uplink capacity}_i / \text{pending sample size}_i)$; 0 without a sample |
| Proximity | $1 - d_i / \text{zone length}$ |
| CPU cost | cycles needed for the collectable bits / per-slot CPU capacity, clipped to $[0, 3]$ |
| Sample age | pending sample age $/ \tau$, clipped to $[0, 1]$ |
| Dwell | time until the carrier leaves Zone B / horizon duration |

Here $w_s$ is the priority weight of the pair's sensor type, $A_s(t)$ its AoI, and $\tau$ the freshness threshold.

### 5.2 Action

The action has 7 values in $[0, 1]$: six feature weights and the requested accuracy fraction. Each weight action $a_k$ is mapped to $\omega_k = 2a_k - 1 \in [-1, 1]$. Every feasible pair is scored with the same weights,

$$
\text{score}_i = \sum_{k=1}^{6} \omega_k f_k(i),
$$

and the highest-scoring feasible pair is scheduled with the requested accuracy.

TD3 and PPO therefore learn a state-dependent scheduling rule: at every slot they choose how much to value urgency, link quality, distance, CPU cost, sample age, and remaining dwell time. The action size does not depend on the number of vehicles, and the Greedy baselines are approximately special cases (Proximity Greedy weights proximity; CPU-aware Greedy weights urgency against CPU cost), so the learned policy can represent them and adapt beyond them.

### 5.3 Observation

TD3 and PPO receive an 80-value normalized observation that does not depend on the vehicle count:

* Per sensor type, padded to 16 types: AoI, priority weight, whether any feasible pair provides that type, and the best reachable accuracy among those providers (64 values).
* The mean and maximum of each pair feature over feasible pairs (12 values).
* Four global values: CPU backlog, time progress, recent CPU load, and the fraction of active sensor types near the freshness threshold.

### 5.4 Reward

The reward combines several operational objectives:

* Reduce weighted sensor-type AoI.
* Avoid freshness violations.
* Avoid accuracy violations.
* Avoid excessive CPU backlog.
* Avoid terminal CPU overload.
* Encourage useful, feasible updates.

The reward is intentionally different from raw reporting metrics. Reporting metrics remain interpretable and count-based, while the reward is shaped to provide a smoother learning signal: freshness and accuracy penalties grow with how far a value is past its threshold (squared slack) instead of jumping at the threshold, and CPU backlog is penalized through a normalized $\log(1 + \cdot)$ term.

This distinction matters because a reward curve should not be interpreted as the final scientific result by itself. It is primarily evidence about training behavior. Final policy quality is assessed through Monte Carlo evaluation metrics.

---

## 6. Heuristic Baselines

### 6.1 CPU-Aware Greedy

The CPU-aware Greedy heuristic scores feasible provider-sensor pairs using urgency and computational pressure.

A pair receives a high score if it is associated with a high-priority stale sensor type. It receives a lower score if serving it is expected to increase CPU backlog too much.

### 6.2 Proximity Greedy

Proximity Greedy adds an important physical assumption: nearby vehicles are often better providers because the uplink channel is stronger.

Its decision logic can be summarized as:

1. Find the closest eligible provider vehicle.
2. Consider the sensors carried by that vehicle.
3. Apply the same urgency and CPU-aware scoring among those candidate sensors.
4. Select the best provider-sensor pair.

Only pairs that hold a pending sample are eligible, so the closest vehicle is the closest one with something to upload. Neither Greedy variant looks at sample age.

This can perform very well because the communication model strongly depends on distance. A nearby vehicle often provides higher uplink rate, more collected data within the slot, higher achieved accuracy, and a lower risk of failed refresh. Therefore, it encodes a physically meaningful scheduling rule that is closely aligned with the wireless channel model.

---

## 7. Evaluation Layer

The evaluation layer turns policies into comparable research results.

A single episode is not enough because the environment is stochastic. Vehicle placement, available data, feasible pairs, and service conditions can vary across random seeds. Therefore, the project uses Monte Carlo evaluation.

The evaluation protocol runs each policy over many independently generated scenarios and aggregates the results.

The primary metrics include:

* Average weighted AoI.
* Maximum AoI.
* Freshness violation count.
* Accuracy violation count.
* Final CPU backlog.
* Terminal CPU violation.
* Total collected data.
* Mean achieved accuracy.
* Episode return.
* Penalized score.

For the current paper direction, the main sensitivity curves focus on average AoI as the primary metric. Other metrics remain useful for diagnosing why one policy performs better or worse.

---

## 8. Experiment Types

### 8.1 Reward Convergence

TD3 and PPO training scripts produce reward convergence curves. These curves show how the learned policy’s evaluation reward changes during training. These plots show if the learned policy improves during training.

A reward curve can be noisy even when the trained policy is useful, especially when the environment is stochastic and the reward contains penalties for threshold violations.

### 8.2 Vehicle-Count Sensitivity

This experiment varies the number of vehicles from 10 to 80. The purpose is to test how performance changes as the number of potential providers increases.

More vehicles can help because there are more possible data sources. However, more vehicles also increase the size of the scheduling space and may introduce more competition among possible updates. Since only one pair can be served per slot, additional providers do not automatically guarantee better freshness.

### 8.3 Data-Size-High-Multiplier Sensitivity

This experiment varies the upper multiplier controlling sensor data size from 1 till 4 (the nominal value is 2.0). The purpose is to test what happens as sensor updates become heavier. Larger data sizes can make the problem harder because they require more upload time, increase CPU processing demand, make it harder to satisfy the accuracy threshold, and increase the chance that urgent sensors remain stale.

### 8.4 Sensor-Type Scalability

The sensor-type scalability experiment varies the number of active sensor types from 4 till 16. This experiment is exceptional because TD3 and PPO are retrained at each sensor-type count but no results were generated using this mode so far because the earlier expirements were not satisfactory.

---

### 8.5 Sample-Arrival-Rate Sensitivity

This experiment varies the sample arrival rate $\lambda$ (parameter `sample_arrival_rate`). Lower rates make data scarce. Policies can then no longer assume that the nearest vehicle has the sensor they want, and must weigh distance against sample availability and sample age. A model trained at one arrival rate can be evaluated at any other, since the observation and action do not depend on it.

---

## 9. Why Proximity Greedy Can Outperform

The strong performance of Proximity Greedy is plausible under the current model.

The main reason is that distance is a dominant physical factor. Since the uplink rate depends on distance, selecting a nearby provider can improve the probability of a successful update. A close provider usually allows the leader to collect more bits within the slot, making it more likely that the achieved accuracy exceeds the required threshold.

Therefore, Proximity Greedy outperforming TD3 or PPO does not automatically indicate a bug. It may indicate that the dominant scheduling factor in the current environment is geometric proximity.

This advantage depends on data being abundant. Without an arrival process, every in-zone pair has fresh data in every slot, so the nearest vehicle always offered the sensor Proximity Greedy wanted. The sample arrival process was introduced to test this. A preliminary baseline-only check (20 seeds, before retraining TD3/PPO) gave the following average weighted AoI:

| Arrival rate $\lambda$ | CPU-aware Greedy | Proximity Greedy | Max-AoI Greedy |
|---|---|---|---|
| `None` (fresh sample every slot) | 7.15 | **5.74** | 7.30 |
| 1.0 | 7.38 | **6.14** | 7.34 |
| 0.3 | 7.91 | **7.29** | 7.48 |
| 0.1 | 8.41 | 8.81 | **8.00** |
| 0.05 | 9.11 | 9.18 | **8.85** |

Proximity Greedy's AoI lead shrinks as samples become scarcer and disappears around $\lambda = 0.1$. Its training-reward return also degrades sharply, from about −291 without an arrival process to −911 at $\lambda = 0.05$. Section 9.1 shows how TD3 and PPO compare at the nominal rate.

---

### 9.1 Why the Policy Weights Pair Features

An earlier version of the RL interface had the actor output one score per pair slot (320 padded slots plus the accuracy), with an observation of per-slot AoI, feasibility, sample size and sample age. Models trained that way reached an average weighted AoI of about 10.9 at every vehicle count, against 8.6–9.5 for the Greedy baselines, and their training curves stopped improving after about 100k steps. A diagnosis on 30 evaluation seeds (20 vehicles, $\lambda = 0.10$, `scripts/diagnose_policy.py`) showed why:

| Policy | AoI | Failed uploads | Distance rank of the pick (0 = closest, 0.5 = random) |
|---|---|---|---|
| Proximity Greedy | 8.87 | 0% | 0.01 |
| CPU-aware Greedy | 8.73 | 36% | 0.57 |
| TD3 (per-slot scores) | 10.91 | 24% | 0.49 |
| PPO (per-slot scores) | 11.00 | 25% | 0.49 |
| Random feasible pair, full accuracy | 12.02 | – | 0.50 |

The RL policies had learned to request full accuracy, but chose pairs essentially at random with respect to distance, so a quarter of their uploads went over links too weak to deliver 80% of the sample. Two design problems caused this:

1. The observation contained no distance, link capacity, reachable accuracy, priority weight, or CPU cost, which are exactly the quantities the heuristics use.
2. Pair slot $i$ is a different (vehicle, sensor) combination in every episode, so an unshared network had to learn the same scoring rule separately for each of 320 output slots.

Weighting shared pair features addresses both: the features carry the missing information, and one weight vector is applied to every pair. As a check of what this action space can express, a fixed weight vector with no learning (weighted urgency + reachable accuracy − 0.3 × CPU cost − 0.3 × sample age, full accuracy) reaches an AoI of 5.78 with 1% failed uploads on the same seeds, well below both Greedy baselines.

A first short training check (200k steps each, models trained at 20 vehicles, evaluated on the same 30 seeds) shows the learned policies beating both heuristics:

| Policy | AoI, 20 vehicles | Failed uploads | AoI, 80 vehicles | Failed uploads |
|---|---|---|---|---|
| CPU-aware Greedy | 8.73 | 36% | 9.33 | 42% |
| Proximity Greedy | 8.87 | 0% | 8.60 | 0% |
| TD3 | 8.44 | 3% | 6.17 | 0% |
| PPO | 7.83 | 2% | 6.11 | 0% |

These are preliminary. The fixed-weight result of 5.78 suggests there is still room for improvement with full-length training, and the final comparison should come from the Monte Carlo and sensitivity runs.

---

## 10. Why TD3 Reward Can Be Noisy Without a Hidden Bug

TD3 reward curves can look noisy for several reasons.

First, the environment is stochastic. Even if the policy is fixed, different random seeds can produce different vehicle positions, sample arrivals and sizes, feasible pairs, and service outcomes.

Second, the reward contains competing objectives. A decision can improve AoI but increase CPU backlog. Another decision can reduce CPU pressure but leave important sensor types stale. The reward compresses these tradeoffs into a single scalar value.

Third, threshold effects can create abrupt changes. A small action change can determine whether achieved accuracy is slightly above or slightly below the target. The smooth slack-based reward reduces these jumps in the reward itself, but the refresh outcome is still all-or-nothing, so the AoI trajectory can still change abruptly.

Fourth, TD3 is a deterministic actor-critic method trained with function approximation. In high-dimensional continuous action spaces, it can show unstable or jagged learning curves, especially when the action is later decoded into a discrete feasible scheduling decision.

---