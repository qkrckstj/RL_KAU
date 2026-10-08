# GPU batched simulation — implementation and measured limits

The optional GPU backend now runs thousands of two-aircraft games in parallel,
including physics, autopilot, weapons, observations and PPO learning. **It is
experimental: the trajectory-equivalence gate failed, so official evaluation
still uses unchanged JSBSim.** Existing policies and experiment sources remain
unchanged. No GPU-trained policy was promoted as a champion.

## What changed

New code lives in `aircombat-rl/experiments/gpu_sim/`:

- `env.py`: batched two-aircraft state, six 120 Hz physics/control updates per
  20 Hz decision, weapon bookkeeping, terminal handling and masked resets.
- `control.py`, `geometry.py`: tensor translations of existing control, gun,
  observations, 31 neural features and the 17-coefficient CEM controller.
- `train.py`, `policy.py`: GPU PPO collection/optimization and checkpoint loading
  back into the original CPU JSBSim evaluator.
- `audit.py`: original-vs-GPU trajectory/game checks and separately timed full
  environment stepping. All output paths must be fresh.

The physics dependency is the published [JSBSim F-16 CUDA port](https://github.com/juyoung020/jsbsim-f16-cuda)
at commit `1462017f6a5c724f9780c5d65b0c9552538ef426` (0.2.1,
GPL-3.0-or-later). It ports F-16 model equations to a CUDA kernel; this work adds
the RL_KAU control/combat/training integration. No claim is made that the
upstream physics engine was written in this change. Its documented local-Earth
approximation and trim interpolation remain fidelity limits.

## Full-environment throughput

RTX 5080, Ryzen 7 9800X3D, Torch 2.14.1+cu130, JSBSim 1.3.0, one Torch CPU
thread. Three repeats, median elapsed time, 256 decision steps per run after
warmup. CUDA synchronized around timing; graph/setup excluded and separately
recorded. Each environment step includes two aircraft × six physics frames,
autopilot feedback every frame, geometry, weapon bookkeeping and observations.
Both aircraft hold their targets; all slots use the same development IC for
this controlled throughput workload. Policy inference and learning are excluded.

| Backend | Parallel games | Environment steps/s | Relative to serial JSBSim |
|---|---:|---:|---:|
| Original JSBSim CPU | 1 | about 4,510 | 1× |
| GPU | 64 | 35,171 | 7.8× |
| GPU | 256 | 119,240 | 26.4× |
| GPU | 1,024 | 514,503 | 114.1× |
| GPU | 4,096 | 1,998,842 | 443.2× |

These are amortized batch throughput gains, not single-flight latency gains,
whole-CPU-machine comparisons, equal-fidelity guarantees, or convergence gains.
GPU setup/capture after the kernel was cached took 0.14–0.17 s in these cases;
first installation/kernel compilation has an additional one-time cost.

## Actual GPU PPO run

`runs/gpu_ppo_smoke_20261008`: 1,024 parallel games, 64 local steps/rollout,
40 iterations, four epochs, minibatch 8,192, 64×64 actor/critic, gamma 0.999.
Learner seats alternate across the batch; opponent is tensor horizontal Ace.

- 2,621,440 environment samples in **10.897 s** of measured collection/update
  iterations, **240,566 samples/s**; setup 0.878 s separately.
- 1,280 actual optimizer steps; finite changed parameters verified on CUDA.
- 1,068 completed training episodes, one win, zero invalid episodes.
- Peak Torch allocation about 155 MiB. Checkpoint and optimizer retained.
- The saved model loaded through the original policy loader and completed both
  physical seats against original Ace in JSBSim: **0 wins / 2 losses**. This is
  an execution check on consumed development conditions, not a strength estimate.

The rollout horizon, batch size, initialization and update/sample ratio differ
from the prior SB3 CPU audit. No direct end-to-end training speedup ratio is
claimed across those configurations. High throughput has not yet produced a
strong policy in this short demonstration.

## Fidelity checks and failed acceptance

The installed upstream `fdm_verify --check --fused --lat 37.5665` passed its
one-frame float64 checks. The largest measured vertical acceleration residual
was 2.168e-4 ft/s², below its 1e-3 threshold. That tests local physics terms;
it does not prove long-horizon task equivalence.

This integration additionally drove original `Aircraft` and GPU aircraft with
identical open-loop commands in three programs (hold, sustained right/accelerate,
cycling all nine actions every two seconds), both aircraft, actual FairFight
initial conditions at 20,000 ft. No premature end or invalid physics occurred.

Maximum position error over these six aircraft:

| Simulated horizon | GPU float64 | GPU float32 (training precision) |
|---|---:|---:|
| 0.05 s | 0.011 m | 0.012 m |
| 1 s | 0.253 m | 0.251 m |
| 10 s | 7.167 m | 7.273 m |
| 60 s | 79.406 m | 75.870 m |
| 120 s | 264.891 m | 257.931 m |

The predeclared drop-in thresholds were position ≤10 m, speed ≤1 kt and
attitude ≤1 degree across measured horizons. **Both precisions failed.**
The steady-flight northing error also accumulates, consistent with the known
coordinate-model difference; the audit does not isolate every contribution
from trim, frame transformation and closed-loop numerical divergence.

Two consumed development seeds, both seats, frozen final CEM versus original
CEM gave matching verdicts in **4/4 games at each precision**. In float64,
kill times differed by 2–4 decisions (0.10–0.20 s). This tiny sample cannot
override the trajectory failure or establish general policy-ranking equivalence.
In float32 the difference reached 21 decisions (1.05 s).
No final test band was opened and no model was selected using these games.

The practical boundary is therefore clear: GPU simulation is available for
separate experiments, with original JSBSim verification of learned policies.
Eliminating the remaining flight-model differences is further physics-port work,
not a CUDA configuration switch. The official default was not redirected.

## Reproduce and inspect

See [GPU backend usage](../aircombat-rl/experiments/gpu_sim/README.md) for install,
training, CPU evaluation and direct batched CEM-policy inference.

```bash
cd /home/hoseon/Desktop/박찬서/RL_KAU/aircombat-rl
source .venv/bin/activate
python -m experiments.gpu_sim.train --out runs/new_gpu_ppo \
  --envs 1024 --rollout 64 --iterations 40 --epochs 4 --batch-size 8192
python -m experiments.gpu_sim.audit --out runs/new_gpu_audit --dtype float32
```

Evidence is in [verification/gpu_sim_20261008](verification/gpu_sim_20261008/):
raw audit JSONs, source snapshots, full process logs, checkpoint, dependency
metadata, official-loader games, and test/review records. The first float64
audit source is preserved because a later review added explicit dtype selection.
The checkpoint and exact run trees are retained in `run_artifacts.tar.gz`;
extract from `aircombat-rl` only where those output directories do not exist.

Validation: **136 tests passed**, four existing warnings, 26.35 s; `pip check`
passed. Independent review covered scalar parity, episode boundaries, graph
capture, reset masks, invalid-state isolation and evidence scope. Review caught
missing skips for the optional dependency; these were fixed, and graph capture
state preservation plus CPU checkpoint loading received additional tests.

Changes are local on `feat/gpu-batched-simulation`. GitHub authentication remains
invalid; no remote issue, push, PR or merge is claimed.
