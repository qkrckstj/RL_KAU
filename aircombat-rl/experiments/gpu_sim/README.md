# Experimental GPU simulation and PPO

This backend runs batched aircraft physics, 120 Hz autopilot control, gun geometry,
match scoring, observations and PPO rollouts on CUDA. It is a separate training
environment. **It failed the original-JSBSim trajectory-equivalence gate and must
not replace the official evaluator.** Read the [measured report](../../../docs/GPU_SIMULATION_20261008.md).

## Install and run

Run from `aircombat-rl` in the intended Python environment:

```bash
source .venv/bin/activate
python -m pip install --no-deps -r requirements-gpu-sim.txt

# New experiment: 1024 parallel games, two F-16s each, GPU PPO.
python -m experiments.gpu_sim.train --out runs/my_gpu_ppo \
  --envs 1024 --rollout 64 --iterations 40 --epochs 4 --batch-size 8192

# Simulator fidelity and throughput; this reports gate failures honestly.
python -m experiments.gpu_sim.audit --out runs/my_gpu_sim_audit --dtype float32

# Evaluate the saved neural policy in the unchanged JSBSim task, development only.
python -m tools.grade templates/project_04_fair --design experiments/gpu_sim \
  --policy runs/my_gpu_ppo/policy.pt --band 33030000 --n 2 \
  --out runs/my_gpu_ppo_jsbsim_check.json
```

Always use new output paths. The short run is an execution demonstration, not a
recommended convergence budget. The 64-step per-environment PPO rollout, large
batch, actor initialization and update/sample ratio differ from archived SB3
experiments; increased sample throughput is not an equal-quality speedup.
The checkpoint is saved after successful completion. The PPO entrypoint does
not resume interrupted training. Separate native DQN and CEM entrypoints are
available below; archived DQN/CEM trainers remain unchanged.

## GPU DQN and CEM

Both entrypoints use this same experimental GPU physics and retain its failed
trajectory-equivalence gate. See [DQN/CEM measurements and checks](../../../docs/GPU_DQN_CEM_20261008.md).

```bash
# DQN: 1024 simultaneous games; GPU Q network, target network and replay buffer.
# --steps counts vector decisions: 4096 * 1024 = 4,194,304 collected transitions.
python -m experiments.gpu_sim.dqn --out runs/my_gpu_dqn \
  --envs 1024 --steps 4096

# CEM: 64 candidates * 4 shared ICs * 2 seats * 2 opponents = 1024 games/generation.
# Starts from the archived final controller, preserving the source policy.
python -m experiments.gpu_sim.cem --out runs/my_gpu_cem \
  --population 64 --elites 8 --ics 4 --generations 3

# Original JSBSim execution checks; these are consumed development conditions.
python -m tools.grade templates/project_04_fair --design experiments/gpu_sim/dqn \
  --policy runs/my_gpu_dqn/policy.pt --band 33030000 --n 2 \
  --out runs/my_gpu_dqn_jsbsim_check.json
python -m tools.grade templates/project_04_fair --design experiments/gpu_sim/cem \
  --policy runs/my_gpu_cem/policy_net.json --band 33030000 --n 2 \
  --out runs/my_gpu_cem_jsbsim_check.json
```

DQN uses 31 features, a 64x64 ReLU Q network, gamma 0.999, uniform replay,
terminal reward +1/-0.2, Huber TD loss, and epsilon 1.0 to 0.05 over 1,048,576
transitions. Default replay capacity is 262,144 transitions; learning starts at
32,768 collected transitions, with one 4,096-sample update per four vector
decisions. `--target-updates 100` copies the network every 100 optimizer steps.
Those rates are explicit new settings, not an equal-work SB3 comparison. Seats
alternate; the opponent is the same tensor horizontal Ace used by GPU PPO.

CEM searches all 17 coefficients within the original controller bounds. The
default parent is `experiments/league/bundle/models/final/policy_net.json`;
`--parent` accepts another valid 17-coefficient JSON or ZIP. Every generation
uses the same recorded training ICs, both seats, tensor Ace and original CEM.
The objective is mean game score (win 1/draw 0.5/loss 0) plus 0.001 times mean
final health difference. The incumbent remains in each population; elites fit
a normalized diagonal Gaussian with 0.5 smoothing and a 0.035 standard-deviation
floor. Candidates all complete their games; invalid games reject the run.

Each fresh output contains configuration, source snapshots, progress, results and
the saved policy. CEM additionally records every population, paired verdicts,
ICs and distribution state. Checkpoints are saved on successful completion.
These new entrypoints do not resume runs; DQN replay is not saved. Short examples
prove execution, not convergence or general competitive strength. Existing PPO
checkpoints and commands remain compatible.

## Environment API

```python
import torch
from experiments.gpu_sim.env import BatchedFairFight
from experiments.gpu_sim.geometry import features, reactive_actions

env = BatchedFairFight(1024, seed=700)
env.capture()  # optional CUDA graph; preserves current state
actions = torch.full((1024, 2), 4, dtype=torch.int64, device="cuda")
raw, done, outcome = env.step(actions)
network_inputs = features(raw)  # (1024, 2, 31), stays on GPU
# Copy terminal observations and flags before resetting completed slots.
terminal_raw = raw.clone()
terminal_done = done.clone()
env.reset_done()
```

Actions are nine discrete heading×speed commands per physical seat. Zero delta
holds the prior autopilot target. Each decision performs six physics frames and
six autopilot updates. The raw `(N,2,39)` contract is seat-relative; outputs are
live buffers and are overwritten on the next step/reset.

`outcome`: 0 active, 1 red wins, -1 blue wins, 2 mutual, 3 timeout, 4 invalid.
`invalid` must be checked: invalid dynamics are quarantined to avoid poisoning
the CUDA context and flagged, not scored as wins. The trainer rejects any run
with an invalid episode. Done slots preserve terminal observations until reset.

`reset(initial=...)` accepts `(N,2,4)` tensors with east/north position in metres,
heading in radians and speed in knots. Use actual `FairFightEnv.sample()` values
for paired comparisons. Default random sampling has the same declared range and
shared speed per pair, but uses Torch RNG, not NumPy's seed stream. It is intended
for the official 20,000 ft, 405–495 kt initial band; arbitrary flight ICs are not
validated by this wrapper.

For batched CEM inference, load validated 17-coefficient policies and provide
`parameters` shaped `(17,)` or `(N,2,17)` to `reactive_actions(raw, parameters)`.
The tensor implementation is tested against the existing controller. The CEM
entrypoint above provides separate experimental search, not the archived league
selection and champion-admission protocol.

## Boundaries

- Physics dependency: [jsbsim-f16-cuda](https://github.com/juyoung020/jsbsim-f16-cuda),
  GPL-3.0-or-later, exact commit pinned in `requirements-gpu-sim.txt`.
- Upstream approximates Earth coordinates and uses an interpolated trim table.
  The wrapper uses Seoul latitude and enables the original refuel setting.
  It does not silently substitute a learned dynamics model.
- GPU resets start each episode at the documented initial fuel. The original
  reused JSBSim backend may retain fuel across resets; exact historical-run
  continuation is not supported.
- The training opponent is the tensor Ace horizontal controller, with altitude
  actions closed. Original Ace can command altitude corrections if separation
  becomes large. Official evaluation uses the original bot.
- CPU scalar rule parity, CUDA state handling, source fidelity, physics
  throughput and trained-policy strength are separate checks.
