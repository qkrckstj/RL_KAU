# GPU publication preserving the latest remote history

**Published:** [Draft PR #2](https://github.com/qkrckstj/RL_KAU/pull/2) and [the GPU branch](https://github.com/qkrckstj/RL_KAU/tree/feat/gpu-training-upload-20261008).
The remote branch commit was verified; `main` remains at `f63cc95`. No merge.

The user explicitly requested publishing the completed GPU work to
`qkrckstj/RL_KAU` while preserving existing work. The upload branch is
`feat/gpu-training-upload-20261008`, based on remote `main` at
`f63cc95abb593f93c9bb5e1b66b3093fc1d73618`.

This remote base contains newer paused league experiments and model-selection
guidance absent from the initial local clone. It is preserved. The completed
local GPU changes through `3352588` are added on top rather than replacing the
remote branch with the older local tree. No force push or main-branch update is
planned. The repository's research pause remains in effect.

## Preservation

- Existing application code, policies, experiment records and split artifact
  archives match the remote base; no existing application file is modified or
  deleted.
- `START_HERE.md` and `REPRODUCE.md` retain their full remote contents, followed
  by the GPU additions.
- Every existing `LEAGUE_STATE.json` value is retained, including the paused
  objective, current models and publication restrictions. GPU metadata is added
  under separate keys.
- The only other existing-file edit is three whitespace rules for immutable GPU
  evidence logs in `.gitattributes`.
- GPU model checkpoints, source snapshots, raw benchmarks and verification logs
  remain included in the committed evidence archives.

## Integration validation

The full suite on this combined tree reports **343 passed, 1 skipped, 3 failed**.
The failures are in existing remote tests and were reproduced on an unchanged
checkout of `f63cc95` in the same Python environment:

1. `test_real_checkpoint_export_relocates_preserves_weights_and_replays`: missing
   restored residual-continuation checkpoint artifact.
2. `test_analysis_of_identical_archived_records_cannot_claim_mode_gain`: missing
   restored residual-continuation plan artifact.
3. `test_initial_greedy_identity_distribution_consistency_and_save_load`: the
   archived residual-policy test passes CPU input to an automatically CUDA-loaded
   model on this machine.

These are reported, not silently skipped or fixed by changing frozen remote
sources. The earlier local GPU tree passed all 145 tests. The combined tree must
not be described as having a completely green suite. No new training experiment
or policy-selection assessment was launched for this publication step.

## Authentication and publication status

At preparation time, HTTPS/`gh` authentication was invalid. SSH authenticated as
`HOSEONGI`, but GitHub denied write access to `qkrckstj/RL_KAU`. The user subsequently authenticated as `qkrckstj`; repository ADMIN permission
was verified. Publication is prepared as a new branch and draft PR only.

The prepared PR body and preservation checks are retained under
`docs/verification/gpu_upload_20261008/`. Remote publication was confirmed by matching the uploaded branch commit
`bf6e84dd2c1e7dcb62fed47feda28be6f24483ab`; later documentation-only commits
record this confirmation. PR #2 remains draft and unmerged.
