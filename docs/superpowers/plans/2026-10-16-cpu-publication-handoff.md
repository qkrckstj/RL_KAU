# CPU continuation publication plan

> **For agentic workers:** Use superpowers:executing-plans for implementation and superpowers:requesting-code-review before publication.

**Goal:** Publish a compact, verifiable cross-computer continuation of the seven-opponent project on existing PR #4.

**Architecture:** Preserve the dirty experiment worktree. Prepare an isolated publication checkout from its published base, copy selected CPU sources and immutable evidence, and provide a new relative-path evaluation entry point. Historical frozen runners remain provenance, never silently rewritten.

**Tech stack:** Python 3.12, native CPU JSBSim 1.3.0, CPU PyTorch, existing policy loader and v15 panel scorer.

**Spec:** User's 2026-10-16 cross-computer handoff request and previously approved 224/448 protocol.

## Constraints and review focus

- Original seven, native physics, verdict and CEM remain unchanged.
- Goal remains paused; publication validation does not resume training.
- Preserve local unpublished files. Do not merge. Only this publication is authorized.
- Reject partial/duplicate evaluation cells through the existing scorer; all draws fails.
- Repository-relative paths resolve on another computer; historical absolute paths are provenance.
- Fail before overwriting an output directory and verify frozen input hashes before/after evaluation.
- New compact manifest identifies included files; omitted residue remains local.

## Tasks

- [ ] Copy selected CPU source closure, all seven-opponent textual reports, 92-candidate results/settings, four feedback candidates/screens and useful neural checkpoints.
- [ ] Add `tests/test_cpu_handoff.py` first; run to observe missing-module failure. Implement `tools/cpu_handoff.py` using existing native evaluator, loader and strict scorer; test fixed axes, relative paths, all-draw rejection and overwrite protection.
- [ ] Write `docs/CPU_CONTINUATION_HANDOFF_20261016.md` and `docs/SEVEN_OPPONENT_PROGRESS_20261016.md`, with commands, exact counts, per-opponent results, pending hypothesis, version rules and resume limitations. Link from START_HERE and current state.
- [ ] Verify manifest hashes, policy tests, CPU-only installation/loading and a clearly labeled small native reproducibility smoke. Keep smoke separate from historical performance totals.
- [ ] Obtain independent code/architecture review, address findings, commit, fast-forward push existing PR branch and verify the remote SHA. Report commit, PR, scope and CPU support in Korean.
