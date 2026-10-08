# Audit CUDA throughput and support CPU PPO rollouts

## Why
The actual RL_KAU workload can be slower on CUDA despite working GPU support.
Frozen research sources and prior policies must remain reproducible.

## What changed
Add a fresh-process CPU/CUDA audit with verified actual optimizer updates,
source hashes, and repeatable CEM worker measurements. Add an opt-in PPO context
for CPU action collection and CUDA optimization, including safe callback saves.
Preserve timing results and document CPU recommendations and exploratory limits.

## How validated
126 tests passed; pip check passed. Three-repeat equal-work neural comparisons;
actual saved DQN CUDA resume; paired CEM outcomes identical across worker counts.
Independent code review completed. See CUDA_RUNTIME_AUDIT_20261008.md and evidence.

## Risks
Different devices and environment counts change learning trajectories. Timing
is not convergence evidence. Two-env CPU/CUDA optimizer work differed due to KL
stopping, so that comparison is explicitly rejected. Frozen sources unchanged.

## Reviewer focus
Temporary PPO method/device restoration, callback serialization, timing scope,
source provenance, and separation of CEM search from neural training.

Local draft only: GitHub credentials are invalid; nothing published or merged.
