# Add experimental GPU simulation with PPO, DQN and CEM

## Why

CPU flight simulation limits collection throughput for small neural policies.
Provide optional batched CUDA flight/combat simulation and native learners, plus
measured fidelity limits, without replacing the official JSBSim evaluator.

## What changed

Add the pinned F-16 CUDA physics integration, tensor control/combat, GPU PPO and
DQN learning, CEM population search, original-evaluator policy adapters, runtime
audits, regression tests and reproducible model/evidence archives. Preserve the
latest remote league work at f63cc95, including all existing application files,
models, split archives and paused research state. Documentation is appended and
GPU state is recorded separately. No champion is promoted or main branch merged.

## How validated

The original GPU implementation tree passed 145 tests. Integration with latest
main: 343 passed, 1 skipped, 3 failed. All three failures reproduce on unchanged
main: two missing restored residual-run artifacts and a residual-policy test's
CPU/CUDA load mismatch. Details and logs: docs/GPU_UPLOAD_20261008.md.

Actual earlier GPU runs: PPO 2,621,440 samples in 10.897 s; DQN 4,194,304 samples
and 1,017 updates in 16.720 s; CEM 3,072 complete games and three distribution
updates in 18.022 s. Checkpoints execute in original JSBSim. These are different
workloads/configurations, not equal-quality CPU speedup estimates.

## Risks

GPU physics failed original-JSBSim trajectory equivalence: maximum float32
position error about 258 m at 120 simulated seconds. This is an experimental
backend and official evaluation remains JSBSim. Native trainers have no resume
entrypoint; DQN replay is not included in its saved checkpoint. The dependency
is GPL-3.0-or-later. Short execution runs do not establish competitive strength.

## Reviewer focus

Preservation against latest main, experiment/policy isolation, masked resets,
terminal replay/GAE boundaries, candidate/IC/seat fairness, faithful reporting
of failed fidelity and existing test failures. Merge remains a manual decision.
