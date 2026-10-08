# GPU DQN and CEM Implementation Plan

> Execute inline using superpowers:executing-plans, then independent review.

**Goal:** Add working DQN and CEM GPU simulation entrypoints and original-evaluator policy loading.
**Architecture:** Additive trainers on the unchanged BatchedFairFight tensor API.
**Tech Stack:** Existing Torch/CUDA and pinned jsbsim-f16-cuda dependency.
**Spec:** docs/superpowers/specs/2026-10-08-gpu-dqn-cem-design.md

## Global constraints

- Base 1071945; official application and existing GPU/PPO Python sources unchanged.
- New runs, source snapshots, unchanged final test bands and archived champions.
- Physics fidelity still failed; no equivalent-quality or champion claims.
- GPU tests skip only when CUDA/optional dependency is unavailable.

## Review focus

- Replay copies terminal observations before reset; no reference aliasing.
- Ring wrap neither drops nor duplicates transitions; sample only initialized slots.
- TD bootstrap stops at all actual task endings, including finite-horizon timeout.
- Every candidate sees identical ICs in both seats against each opponent.
- Invalid/unfinished CEM games are rejected, never scored as draws or victories.

## Task 1: DQN

Files: new `experiments/gpu_sim/dqn/{__init__,__main__,train,policy}.py`,
new `experiments/gpu_sim/run_utils.py`, `tests/test_gpu_dqn_cem.py`.

- [x] Write CPU replay test: capacity 5, insert rows 0..2 then 3..5, verify retained
  rows 1..5 and copied next-state values after input mutation; oversized add rejects.
- [x] Write TD test: rewards [1,2], next max [10,20], done [true,false], gamma .9
  must produce [1,20]. Run and observe missing implementation before writing it.
- [x] Implement Replay.add/sample, QNetwork, td_targets and CUDA trainer using
  `features(env.obs[row,seat])`, explicit pre-reset next state, uniform samples,
  Huber loss, gradient clipping, target copies, and fresh-output manifests.
- [x] Test real CUDA optimizer updates, target-copy cadence, finite changed weights,
  CPU Policy.act and duplicate-output guard. Retain a measured longer smoke run.

## Task 2: CEM

Files: new `experiments/gpu_sim/cem/{__init__,__main__,train,policy}.py` and tests.

- [x] Write elite-fit test: candidates [.1,.3,.9], scores [0,2,1], best two mean .6,
  std .3; with old mean .5/std .2, smoothing .5 produces mean .55/std .25.
- [x] Test layout and shared IC/seats plus score verdicts and invalid rejection.
  Run red tests before adding implementation.
- [x] Implement candidate x opponent x IC x seat batches, full episode evaluation,
  incumbent retention, elite fitting and distribution state/parameter JSON exports.
- [x] Run actual CUDA CEM generation with multiple candidates and both seats;
  verify full terminal results, distribution updates and original loader parity.

## Task 3: Integration, evidence and review

- [x] Run representative DQN collection/update and CEM search with synchronized
  CUDA timers; archive raw metrics, models, source snapshots and failed runs.
- [x] Execute each exported policy through original JSBSim in both seats on open
  consumed development conditions; report quality separately from successful execution.
- [x] Run full pytest and pip check, independent code/architecture review, address
  important findings, update README, report, START_HERE and LEAGUE_STATE.
- [x] Commit locally, make the tested branch accessible from RL_KAU, verify there.

## Execution ledger

- Reused clean detached worktree RL_KAU_GPU at 1071945, branch feat/gpu-dqn-cem.
- Ruling: implement the explicitly requested reversible extension without another
  approval round, as required by session autonomy instructions. No merge/publish.
- Pre-flight: both trainers consume unchanged BatchedFairFight/geometry; shared
  run helper produces manifests only, no shared mutable learner state.

- Task 1 complete: GPU replay and DQN learning/export passed tests. Actual run
  collected 4,194,304 samples, performed 1,017 updates/10 target copies in 16.720 s.
- Task 2 complete: 64-candidate CEM completed 3,072 paired games, 3 distribution
  updates in 18.022 s. Invalid games=0. Incumbent training win score stayed 1.0;
  only health tie breaker improved. No general-strength claim.
- Review P2 resolved with RED/GREEN regression: size-two populations retain a
  random challenger. Default-size behavior unchanged; measured sources retained.
- Real CUDA timeout test verifies DQN terminal observations/done survive reset.
- Task 3 validation: full suite 145 passed (32.35 s), pip check passed. Original
  JSBSim checks completed 8 games total, both seats. DQN: 2 losses/2 draws; CEM:
  4 wins from a strong parent. Open consumed development only, no promotion.
- New models, logs, all populations, source snapshots and checksums retained in
  docs/verification/gpu_dqn_cem_20261008. GitHub authentication rechecked invalid;
  issue and PR drafts local, no merge.
