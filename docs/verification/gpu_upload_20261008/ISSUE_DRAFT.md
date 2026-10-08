# Publish completed experimental GPU training without replacing existing work

The user explicitly requested uploading the completed CUDA simulation and
PPO/DQN/CEM work to this repository while preserving existing work.

The proposed branch starts from latest main f63cc95 and adds GPU code, models,
source snapshots and evidence. Existing application files, policies, experiment
archives, model guidance and paused research state are retained. No new research
run, champion promotion, main-branch update or merge is authorized.

Validation: combined suite 343 passed, 1 skipped, 3 failed; all three failures
reproduce on unchanged main (two missing restored artifacts, one CPU/CUDA test
input mismatch). GPU physics still fails trajectory-equivalence acceptance.
See docs/GPU_UPLOAD_20261008.md and the preservation evidence.

@Codex: review this additive publication only; do not resume training or merge.
