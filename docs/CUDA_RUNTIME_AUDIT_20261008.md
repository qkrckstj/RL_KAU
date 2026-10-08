# CUDA and training throughput audit — 2026-10-08

CUDA works on this Linux machine. The existing small DQN/PPO networks are faster
on CPU; the current CEM policy has no neural network to accelerate with CUDA.
The useful improvements measured here are CPU simulation parallelism and, if
GPU PPO updates are required, CPU action collection. These are throughput
results, not evidence of improved policy quality or time to convergence.

## Machine and scope

- Ryzen 7 9800X3D, RTX 5080 16 GB, NVIDIA driver 580.159.03.
- Project-local `aircombat-rl/.venv`: Python 3.12.3, PyTorch 2.14.1+cu130,
  SB3 2.9.0, Gymnasium 1.3.0, NumPy 2.5.3, JSBSim 1.3.0.
- Exact portable requirements installed; `pip check` passes. The prior Windows
  DLL problem does not describe this Linux runtime.
- Base commit: `461284b1b046e890dab8436570ef34b71904e36b`.
  All previously tracked application sources, physics, policies, checkpoints,
  and results remain unchanged. New code and outputs are separate.
- No league search, model selection, or final-test evaluation was performed.

## Controlled CPU/CUDA measurement

Each cell is the median of three fresh processes, seeds 0/1/2, 24,576 environment
steps after a discarded warmup. CPU/CUDA execution order alternates. CPU Torch
threads are capped at one. Initialization and warmup are outside the timer;
CUDA is synchronized at measured phase boundaries. Identical initial weights,
step counts, and actual optimizer-step counts were verified across devices.
Finite changed weights were verified on the requested device, so these are
actual CUDA learning runs, not GPU detection alone.

| Learner | CPU seconds | CUDA seconds | CUDA time increase | Optimizer steps/run |
|---|---:|---:|---:|---:|
| DQN | 12.066 | 15.538 | 28.8% | 5,894 |
| Double DQN | 12.164 | 15.810 | 30.0% | 5,894 |
| Default PPO | 14.124 | 23.535 | 66.6% | 3,840 |
| Archived warm-start PPO | 11.550 | 18.704 | 61.9% | 480 |

DQN/PPO use the actual `experiments/plan_a/core.py` factories: 31 observations,
64×64 hidden layers, and the preserved training settings. Warm PPO loads the
actual transferred actor/fresh critic from
`runs/autolab_20261005/ppo_pilot/s700/checkpoints/step_0/policy_net.zip`
(SHA-256 `d408f8395b9e5bfac7909a52b2f3dd114965a07dc448454c3c92495737b78eb9`).
Its settings are rollout 2048, batch 256, five epochs, target KL 0.01. This avoids
recreating actor transfer from an unavailable historical teacher replay buffer.

## Why the GPU loses

1. JSBSim and environment logic run on CPU. Selecting `cuda` moves the neural
   learner; it does not move flight simulation onto the GPU.
2. Each single-environment action passes a tiny observation to the GPU and
   returns an action to CPU. Small matrix operations cannot amortize launch,
   transfer, and synchronization costs.
3. DQN uses minibatches of 32; even gradient computation is too small here.
   Median DQN update time was 3.173 s on CPU versus 5.282 s on CUDA.
4. Warm PPO spends most time collecting experience: CPU rollout 11.044 s,
   update 0.498 s; CUDA rollout 17.982 s, update 0.720 s. Optimizing just the
   GPU backward pass cannot remove the dominant collection time.
5. The archived Plan A worker and warm-start PPO worker select CPU internally.
   Installing CUDA does not override those choices. The existing
   `tools.resume_dqn --device cuda` and `templates/train.py --device cuda`
   do expose device selection. Frozen experiment scripts were not rewritten.

The phase timings support this explanation but are not a kernel-level trace.
Do not sum independently calculated medians as though they were one run.
This agrees with [SB3's PPO guidance](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#example),
which recommends CPU and subprocess environments for non-CNN PPO workloads.

## Improvements measured separately

### PPO CPU collection with CUDA updates

The optional `tools/ppo_cpu_rollouts.py` context manager moves the policy to CPU
for a whole rollout, then back to CUDA for optimization. Adam state remains on
CUDA and parameter identities are preserved. SB3's rollout buffer stores CPU
NumPy arrays and converts minibatches to the training device.

Warm PPO took **11.773 s**, versus 18.704 s with full CUDA: **37.1% less time**.
All three runs executed 24,576 steps and 480 optimizer steps. This remains about
1.9% slower than entirely CPU training at 11.550 s. CPU remains the measured
choice for the current model. The helper is useful when GPU updates are needed,
not a reason to force this small model onto CUDA.

The context restores devices and methods on exceptions. Regression tests cover
GPU Adam state, actual weight updates, an interrupted rollout, and a checkpoint
saved inside a callback followed by CUDA reload and training. gSDE is rejected.
Do not force temporary methods into `model.save(include=...)`. Device changes
alter numerical/RNG paths, so identical learning trajectories are not promised.

Example in a **new** training script, after constructing/loading a PPO model
with its environment and `device="cuda"`:

```python
from tools.ppo_cpu_rollouts import cpu_rollouts

with cpu_rollouts(model):
    model.learn(total_timesteps=24576, callback=callback)
model.save(new_output_directory / "policy_net.zip")
```

This helper is opt-in; archived trainers do not automatically enable it.

### Two PPO environments

Two subprocess environments, with global rollout size kept at 2048 and local
rollout size 1024, gave CPU median **7.536 s**, versus single-env 11.550 s:
**34.8% less time** for the same 24,576 steps and 480 optimizer steps.
This is an exploratory sampling change: horizon, seeds, and sample correlations
change. Equal sample counts do not establish equal training quality.

The two-env CUDA runs took median 10.881 s, but performed **473/479/480** optimizer
steps versus CPU **480/480/480** because the existing KL early-stop criterion
triggered differently. The audit intentionally exited with its equal-work
guard after recording all six runs. **No equal-work CPU/CUDA speedup is claimed
for this matrix.** KL stopping was not disabled to manufacture a comparison.
The raw runs and failure log are preserved; snapshot hashes were checked
externally because this guard precedes the final in-process hash check.

For subprocess environments, `environment_seconds` measures parent `step_wait`
wall time including IPC/wait, not total simulator CPU consumption. Hybrid
policy transfers are included in total time, outside the rollout sub-timer.

### CEM simulation workers

The current controller evaluates 17 scalar coefficients with NumPy/math; its
`device` argument does not create a Torch/CUDA model. Existing league code
already maintains a process pool with default three workers.

For 24 fixed-policy games against Ace/Pursuit/Evader, both seats, previously
consumed development conditions 33030000–33030003:

| Comparison | Median seconds | Time reduction |
|---|---:|---:|
| Existing serial / 3 workers | 8.051 / 3.229 | 59.9% |
| Additional paired 3 / 6 workers | 3.249 / 2.043 | 37.1% |

Each worker count was measured three times with alternating order. All games,
32,446 simulation steps per case, and complete outcome hashes matched. Six
workers are a bounded diagnostic override; the project default remains three.
This supports trying six workers in a newly recorded CEM configuration on this
machine. It does not prove a 37% speedup for full CEM search, neural opponents,
admission/evaluation, or a machine with a different CPU. The original league
CLI has no `--workers` switch; a future full-run entrypoint must record that
setting in a new plan rather than modifying a frozen plan in place.

## Run on this machine

```bash
cd /home/hoseon/Desktop/박찬서/RL_KAU/aircombat-rl
source .venv/bin/activate
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())'

# Use a new output name for every invocation.
python -m tools.runtime_audit --out runs/my_cpu_cuda_audit --steps 24576 --repeats 3 --cem
python -m tools.runtime_audit --out runs/my_hybrid_audit --algorithms ppo_warm --devices cuda --steps 24576 --repeats 3 --cpu-rollouts
python -m tools.runtime_audit --out runs/my_parallel_audit --algorithms ppo_warm --devices cpu --steps 24576 --repeats 3 --n-envs 2

# Existing DQN resume command; a complete replay checkpoint must exist.
python -m tools.resume_dqn --run runs/plan_a_20261001/duration/s2 --checkpoint 204800 --additional-steps 128 --val-every 128 --val-n 2 --device cuda --out runs/my_cuda_resume
```

The last command was verified in `runs/runtime_cuda_resume_20261008`: step
204800 → 204928, gradient updates 50950 → 50982, replay size 50000 → 50000,
finite final weights. It restores learner/optimizer/replay and starts a fresh
episode; this is not an exact historical trajectory continuation. Two validation
games are execution checks, not a policy-strength estimate.

For a clean installation on this same class of machine, install the compatible
CUDA PyTorch wheel first, then `requirements-portable.txt` and the editable
package, as described in `REPRODUCE.md`. The installed environment and complete
dependency versions are recorded below; a local virtual environment is not a
portable artifact.

## Evidence and review

- [Evidence directory](verification/runtime_20261008/): readable raw JSONs,
  process logs, environment/versions, CUDA-resume metadata and final model.
- `raw_audits.tar.gz` retains all four audit output trees, per-case logs, raw
  CEM games, and available source/input snapshots. The original baseline has a
  narrower hash list; later hybrid/vector snapshots include physics and inputs.
- `integrity.json` checks captured hashes and confirms all originally tracked
  application files against the base commit. The callback serialization fix
  followed timing measurements; the exact measured helper is in the archive.
- `.venv/bin/python -m pytest -q`: **126 passed**, four warnings, 25.92 s.
  Warnings are SB3's small-MLP CUDA advisory and existing multiprocessing fork
  warnings. No failed or skipped tests in this environment.
- Independent code review found the callback-save issue before completion;
  the regression reproduced it and the exclusion/restoration fix passes.
  Review also checked the unequal-work and timing interpretation limits above.
- Changes are local. GitHub authentication is currently invalid; no issue,
  push, PR, or merge is claimed.
