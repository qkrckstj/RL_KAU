# Continuous CPU rear learning and independent evaluation

The remote update at commit `1c397248b3e398de32bc8e50e1ff84a4cbc29ff2` was inspected in `.worktrees/rear-async`. The current handoff, rear budget v2, portable parent and opponent bundles, automated tactics strategy, reward code, curriculum code, and reference runners were reviewed. The previous CUDA worktree and models remain intact. Remote main removed its old GPU prototype directory, so it was not merged over the CUDA worktree.

## Measured worker choice

The user requested CPU learning and preservation of small positive gains. With the same parent, 32 environments, and PPO settings, two reversed benchmark passes gave median rates of 4,583 / 7,605 / 10,599 / 10,231 steps per second for 2 / 4 / 8 / 16 processes. All final actor hashes matched within each seed pass. Eight processes were selected. Their peak process-tree RSS was 5.86 GiB, compared with 10.84 GiB for sixteen. RSS includes shared pages; it is not total physical memory consumption. The benchmark executed 327,680 training interactions and no evaluation games. It measures short warmed rollouts and does not establish a universal worker optimum or same-setting GPU speed.

## Current run and budget

Run directory: `aircombat-rl/runs/rear_cpu_main_20261012`.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m tools.rear_async_cpu launch \
  --out runs/rear_cpu_main_20261012 --workers 8
```

A single CPU trainer runs 32 native JSBSim environments on eight processes. A separate low-priority CPU evaluator uses one process. Both continuations, seeds 42100 and 42101, restore geometry_s10101 actor, critic and Adam state. Reserved training bands are 1800400000 and 1810400000. The minimum is 5,242,880 additional steps per repeat, with a conditional maximum of 10,485,760. Checkpoints are saved every 655,360 steps and monitor evaluations are requested every 1,310,720.

The quarter curriculum fraction remains 50%, the prior bias remains 6, and the 48-opponent base distribution, coverage period 2, PPO settings and rear reward coefficients are preserved. The completed focus schedule stays disabled. Checkpoint `prefix_steps` records the saved SB3 counter, not the model's complete ancestral lifetime training. Reported additional steps count this continuation only.

## Independent evaluation and candidates

Each checkpoint is written in a staging directory. It contains model and optimizer, learner RNG, worker sampler state, exported policies and hashes. A READY manifest appears only when publication completes. The evaluator copies and verifies the immutable checkpoint; job identity includes checkpoint and configuration. Completed cells are reused on retry, and completed jobs are not repeated. Three failed attempts produce a blocked record. Evaluation failures do not stop intermediate training; required failures are surfaced at the minimum-budget decision boundary.

At the minimum endpoint, live model, optimizer and environments are retained while the three latest monitoring points and broader development assessment finish. Repeated rear skill improvement, or maintained rear skill with reduced incoming damage, can extend the same continuation to the maximum. Win-rate decline alone does not discard the rear lineage. All checkpoints and small positive gains remain selection candidates; they are not automatically statistically confirmed improvements or replacements for the frontline model.

Historical positive branch-selection records are indexed in `runs/rear_cpu_async_20261012/historical_candidates.json`. Some are recorded-data proposals requiring runtime evaluation, rather than deployable proven winners. Their original data and historical reports remain preserved.

## Stop, restart and verification

Create `runs/rear_cpu_main_20261012/STOP` for a safe stop. Remove it and add `--resume` to resume. Complete checkpoint counters and any approved extension limit are retained. Optimizer, RNG and opponent sampler states are restored; native flight episodes start fresh. Exact mid-flight restoration is not claimed. Frozen source changes fail visibly. GitHub publication requires an explicit user request.

Seven focused tests passed for atomic publication and corruption rejection, job identity, small-gain retention, minimum and bounded budgets, extension resume, and failed-evaluation cache reuse/deduplication. Real CPU integration v2 executed 40,960 training steps, 34 completed training games and 14 evaluation games: 4.70 seconds learning, 7.04 seconds evaluation, 14.95 seconds overall. These stage times overlap. Parent actor, critic and Adam restoration was exact; step increments and parameter updates were verified. Torch/NumPy export relative logit error was 1.91e-6. The earlier integration also showed training completion while its evaluator remained busy. Smoke runs are qualification records, not policy performance evidence.

The public Windows replay's terminal health results (1, 0) reproduced for both published wins, but rear damage and duration differed on this Linux runtime: 0.6535 versus 0.5215, and 0.967 versus 1.0; duration differences +0.15 and -1.05 seconds. Exact cross-platform replay qualification therefore did not pass. The cause is not established, and physics was not modified to force agreement. This run computes fresh same-machine native baselines and does not reuse those Windows outcome caches. Bundle integrity and policy loading passed for 320 files and 73 actor instances.

## CSV logging on process resume

SB3 CSV logger configuration overwrites an existing progress.csv at the same log directory. Before any manual same-run resume, archive the s*_metrics directories. The automatic local-skill extension preparation tool now archives them alongside the original budget decisions and terminal records before allowing a resume.

The first entropy extension occurred before this safeguard and overwrote both pre-extension optimizer-metric CSVs. Exact complete CSV recovery is unavailable. All learner checkpoints, READY counters, training episode JSONL, evaluation cells/results, cycle timing reports and recorded first-monitor summaries remain available. Current extension metrics were copied to the archived segment on inspection. See runs/rear_cpu_entropy01_20261012/metrics_logging_issue.json for the precise limitation; no simulations are repeated just to regenerate logging history.
