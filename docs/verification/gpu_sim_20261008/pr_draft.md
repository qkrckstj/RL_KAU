# Add experimental batched CUDA simulation and GPU PPO

## Why

Original neural training collects each flight through CPU JSBSim, leaving little
work for the GPU. Add a separate batched simulator to remove that collection
bottleneck while preserving the official task as the reference evaluator.

## What changed

Integrate the pinned published F-16 CUDA physics port with tensor autopilot,
two-seat combat, observations, reset/terminal handling and GPU-native PPO.
Provide a checkpoint adapter for original JSBSim evaluation and a reproducible
audit. Existing application sources, policies and results are preserved.

## How validated

136 tests passed; pip check passed. Actual PPO collected 2,621,440 samples and
performed 1,280 updates in 10.897 s. Checkpoint loading ran both seats in JSBSim
(0 wins, 2 losses). Timed simulation reached 1,998,842 env steps/s at 4,096 games
versus about 4,510 serial CPU env steps/s, with workload limitations recorded.
Both float32 and float64 trajectory audits completed and failed acceptance.

## Risks

Maximum 120-second trajectory error is about 258 m in float32 and 265 m in
float64. This backend is experimental and cannot replace official evaluation.
No equal-quality training-speed or policy-strength claim is made. Optional
physics dependency is GPL-3.0-or-later. No DQN port or resumable PPO is included.

## Reviewer focus

Control/rule parity, preserved masked state, invalid and terminal boundaries,
CUDA graph capture, GAE episode separation and faithful reporting of failed
fidelity. This is a local draft; no remote PR or merge was performed.
