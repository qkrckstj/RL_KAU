# Project context for a new agent

**User paused all autonomous training/evaluation on2026-10-08. Do not resume it until explicitly requested.** Read `docs/MODEL_CHOICE_AND_PAUSE_20261008.md`. The user authorized publication of current work and that recommendation; this does not restore permission for automatic future GitHub uploads.

Read `START_HERE.md` and `docs/LEAGUE_STATE.json` before continuing experiments.
They identify the objective, current artifacts, completed work, and next branches.
Older sections of `HANDOFF.md` are historical; their PIDs and “running” labels are not live state.

Preserve official FairFight physics, initial conditions, weapon and verdict rules, and the original CEM policy.
Keep prior policies and raw results. Separate CEM parameter search from neural DQN/PPO training in reports.
Inspect actual processes and completion/failure files before launching or resuming a run. Never duplicate a live run.
On Windows, do not repeatedly open actively overwritten progress/reuse JSON while training: atomic replacement has failed with WinError 5. Use `python -m tools.league_live_status <run>` for file metadata and inspect actual process identity separately. Read immutable completed state/result artifacts for analysis. A FILE_SHARE_DELETE reader did not prevent the observed replacement failure in a real local test, so do not assume it fixes this.
Use a new output directory for changed settings or a new experiment. Frozen source/input hashes must not be bypassed.
Both seats and shared initial conditions are required for paired policy comparisons. Retain older opponents.
Do not select a model using the final test, or reuse a consumed test as unseen evidence.
Do not infer universal tournament performance from local scripted opponents or from Ace-only results.

Run commands from `aircombat-rl` with the intended Python environment. The latest `runs/league_split_benchmark_20261007/completion.json` recommends **16 total workers: 14 verified NumPy-only +2 ordinary neural/unknown workers** through `tools.league_split_pool.SplitPool`. Use `experiments/league/execution_profile.json` and set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` BEFORE Python starts. Two reversed trials of the same 216 games per configuration (1,728 total executions) matched all official episode and selected full damage-trace records. Median match time improved from ordinary8 16.305s to split16 11.217s (1.454x throughput, 31.2% less time); minimum observed commit headroom was 5.795 GiB. This measures a benchmark, not a complete training run. Unknown/neural policies must use ordinary workers; do not add source signatures without inspecting and verifying the policy code. Check current commit headroom before starting. Existing frozen runs retain their loader/count. The prior ordinary-loader benchmark selected 8 for memory headroom; its skipped 12/16 trials were not measured slower. CLI defaults remain three; measure on another machine.
Before increasing concurrency or launching auxiliary jobs, inspect Windows system commit headroom as well as available physical RAM. The 2026-10-07 01:22 sample had 6.64 GiB physical memory free but 95.75% of the commit limit used (1.90 GiB headroom). An audit waiter had a native crash and a JUnit environment query emitted E_OUTOFMEMORY; causation is not established. Preserve the frozen active run, avoid concurrent extra simulations, and include commit headroom in future worker-count decisions. Record actual process exits and Windows events before resuming; never duplicate live training.
For future assessments with about25% ordinary-policy jobs, the newer `runs/league_budget_pool_benchmark_20261008/completion.json` selects16total workers with12NumPy+4ordinary via `experiments/league/execution_profile_budget_20261008.json`. Across1,344 repeated game executions with exact episode/selected trace agreement, median match times were23.018/15.665/16.254seconds for ordinary2/4/6. This scope-specific1.469x gain over14+2 is not a training-worker recommendation. Frozen old runs retain their original profiles. The current compact PPO uses8physical/32virtual environments and48active foes; full205archive remains preserved. Read its active roster manifest and current state before selecting future opponents.

Use `REPRODUCE.md` for installation and `experiments/league/bundle/README.md` for policy loading.
An archived Windows PID, absolute runtime path, or installed virtual environment is not portable to another machine.
Published policy weights and CEM search records are included; some historical neural replay buffers are local-only.

Update the current state record and result report after meaningful completed work.
GitHub publication is excluded from automation. The user revoked prior automatic-upload authorization; push or publish only when the user explicitly requests it again. Continue training, evaluation, analysis and local artifact preservation autonomously.
When explicitly asked to upload, include model artifacts and verification evidence, and distinguish a local result from a verified remote commit.
