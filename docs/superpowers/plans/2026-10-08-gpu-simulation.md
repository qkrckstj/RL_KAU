# Batched GPU Simulation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run many actual flight simulations on GPU and connect GPU rollouts to training.

**Architecture:** Optional pinned CUDA F-16 physics plus tensor autopilot, combat,
observation, and PPO layers. Original JSBSim remains the reference evaluator.

**Tech Stack:** Python 3.12, Torch 2.14.1+cu130, JSBSim 1.3.0, jsbsim-f16-cuda 0.2.1.

**Spec:** `docs/superpowers/specs/2026-10-08-gpu-simulation-design.md`

## Global constraints

- No edits to archived application sources, policies or experiment outputs.
- Optional physics dependency pinned to 1462017f6a5c724f9780c5d65b0c9552538ef426.
- 120 Hz physics/autopilot, 20 Hz decisions, official action/weapon/judge rules.
- New output directories; no final-test evaluation or champion selection.
- Report upstream flat-Earth/trim limitations and measured local fidelity.

## Review focus

- Zero heading/speed action must hold its previous target (task 1 test).
- Masked reset must preserve all other control and physics state (task 2 test).
- Terminal observations must precede reset and prevent GAE crossing episodes (task 3 test).
- Invalid physics must not become a kill or silently poison training (task 2 test).
- GPU timing must count aircraft/matches/physics frames distinctly (task 4 evidence).

## Task 1: Tensor control and observation contracts

Create `experiments/gpu_sim/control.py`, `geometry.py`, and package `__init__.py`.
Consume original constants and `AutopilotGains`; produce tensor control updates,
engagement geometry, 39-channel observation packing and 31-channel features.

- [x] Add `tests/test_gpu_sim.py` scalar-vs-tensor tests using seeded AircraftState
  samples, `Autopilot.update`, `engagement.look`, `State(specification(...))`.
- [x] Run `.venv/bin/python -m pytest tests/test_gpu_sim.py -q` and confirm absent
  implementation fails before creating code.
- [x] Implement tensor equivalents with float64 parity tests at tight tolerance,
  no independent re-tuning of constants or control rates.
- [x] Test repeated updates and saturating control memory, not only one input.

## Task 2: Batched dual-aircraft environment

Create `experiments/gpu_sim/env.py` and `requirements-gpu-sim.txt`.
Consume task 1 tensors and `F16Stick`; produce `BatchedFairFight`.

- [x] Add tests for discrete target persistence, simultaneous weapon damage,
  mutual tolerance, timeout, masked reset, invalid-state handling, seat symmetry.
- [x] Implement `(N,2)` match state, six autopilot+physics substeps, terminal
  observation/flags, explicit reset masks and source/version metadata.
- [x] Run actual CUDA tests, including finite nonidentical aircraft trajectories
  and untouched state for slots not reset.
- [x] Keep tensor-only stepping capture-compatible; use optional CUDA graph
  replay only after graph/eager state equality checks pass.

## Task 3: GPU PPO training entrypoint

Create `experiments/gpu_sim/train.py`, consuming env observations/features and
producing separately saved checkpoints/config/metrics, never changing archives.

- [x] Test GAE against a small hand-computed sequence with episode boundaries.
- [x] Implement batched opponent, GPU rollout tensors, clipped PPO update,
  finite checks, fresh-output guard, and reproducible checkpoint load.
- [x] Execute a bounded smoke run and verify actual changed finite GPU weights.

## Task 4: Fidelity and throughput evidence

Create `experiments/gpu_sim/audit.py`; consume original and GPU environments.

- [x] Run upstream one-frame verification at the local latitude.
- [x] Compare aircraft trajectories and paired policy matches on development
  seeds; retain errors and disagreements without adjusting acceptance after seeing them.
- [x] Measure GPU batches 64/256/1024/4096 against serial original JSBSim and
  distinguish end-to-end training from physics-only speed.
- [x] Run full suite and pip check, review code independently, then update report,
  START_HERE and current-state JSON with measured limitations and exact commands.

## Execution ledger

- Baseline commit 4364d36; isolated worktree `RL_KAU_GPU`, branch
  `feat/gpu-batched-simulation`. Shared existing project venv via ignored symlink.
- Read-only exploration found a published F-16 CUDA port; use it instead of
  silently substituting a learned surrogate. Upstream sources cloned separately.
- User authorized implementation directly; proceed with reversible optional
  backend work and retain the official evaluator while fidelity is measured.

- Tasks 1–3: scalar parity and regression tests passed; new GPU environment and
  PPO run with actual CUDA physics, reset graphs and changed finite weights.
  Red tests exposed absent modules, Ace snap-threshold mismatch, reset-mask
  aliasing and checkpoint safe-loading compatibility before their fixes.
- Task 4: upstream one-frame check passed, but local 120-second trajectory
  acceptance failed in both precisions (float32 maximum 257.931 m). This is a
  measured failed fidelity gate, not a completed equivalent-physics replacement.
- Controlled 4,096-game simulation throughput: 1,998,842 env steps/s. PPO smoke:
  2,621,440 samples, 1,280 updates, 10.897 s; no equal-quality speed claim.
- Independent review packaging finding fixed; capture preservation and actual
  float32 fidelity checks added. Full suite: 136 passed; pip check passed.
- Original JSBSim checkpoint loader ran both seats: 0 wins / 2 losses. No new
  policy selected; original source files and final evaluation bands preserved.
- Local report/state and raw evidence retained. GitHub authentication invalid;
  issue and PR drafts are local, and no merge is performed.
