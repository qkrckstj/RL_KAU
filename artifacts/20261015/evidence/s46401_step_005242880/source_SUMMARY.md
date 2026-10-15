# Predictive aiming CPU learning

Original native JSBSim; 8 physical workers /32 environments. Fixed aiming and HP-exchange reward. All models retained.

Fixed teacher prior6.0;44learner features/39public channels. One public heading-rate cue added;current reward0.05/12km retained. No new elapsed-time tactical switch. Original columns/Adam/counters retained;new columns start0. ZIP bytes change. Initial inference verified on consumed1609frame trace;cached reference games reused.

| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Wall seconds (overlap included) |
|---|---:|---:|---:|---:|---:|---:|
| 46400 | 5,242,880 | 5,892 | 572.52 | 1,056 | 545.95 | 928.37 |
| 46401 | 5,242,880 | 6,003 | 570.37 | 1,056 | 553.01 | 926.19 |

Live: seed 46401, 5,242,880 added decisions /6,003 completed training games. Saved counters above lag live progress.

New completed evaluations: 2112; reused parent games: 672.
Elapsed wall time: 1857.18s; completed evaluation process time: 1098.96s. Training and evaluation overlap.

| Model /panel | Added steps | Wins /games | Win rate | Gain vs unchanged prior6comparator | Dealt /received HP | First-shot rate | Connected firing seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
Cached unchanged prior6comparator: 672unique matches /800panel entries. Hard monitor:8foes/4IC/both seats/both samplers;full acceptance21foes/8IC/both seats/both samplers.

| parent-prior6-comparator /development | 0 | 498 /672 | 74.11% | baseline | 0.9260 /0.5939 | 22.62% | 2.823 |
| parent-prior6-comparator /hard-monitor | 0 | 54 /128 | 42.19% | baseline | 0.8590 /0.8541 | 2.34% | 2.618 |
| 46400 /hard-monitor | 1,310,720 | 54 /128 | 42.19% | +0.00pp | 0.8572 /0.8604 | 3.12% | 2.612 |
| 46400 /hard-monitor | 2,621,440 | 55 /128 | 42.97% | +0.78pp | 0.8526 /0.8533 | 2.34% | 2.598 |
| 46400 /hard-monitor | 3,932,160 | 42 /128 | 32.81% | -9.38pp | 0.8504 /0.8639 | 1.56% | 2.591 |
| 46400 /development | 5,242,880 | 479 /672 | 71.28% | -2.83pp | 0.9304 /0.5880 | 24.85% | 2.836 |
| 46400 /hard-monitor | 5,242,880 | 47 /128 | 36.72% | -5.47pp | 0.8487 /0.8592 | 2.34% | 2.587 |
| 46401 /hard-monitor | 1,310,720 | 53 /128 | 41.41% | -0.78pp | 0.8566 /0.8579 | 2.34% | 2.611 |
| 46401 /hard-monitor | 2,621,440 | 50 /128 | 39.06% | -3.12pp | 0.8549 /0.8511 | 3.12% | 2.605 |
| 46401 /hard-monitor | 3,932,160 | 46 /128 | 35.94% | -6.25pp | 0.8538 /0.8537 | 1.56% | 2.602 |
| 46401 /development | 5,242,880 | 494 /672 | 73.51% | -0.60pp | 0.9284 /0.5799 | 25.74% | 2.830 |
| 46401 /hard-monitor | 5,242,880 | 54 /128 | 42.19% | +0.00pp | 0.8667 /0.8583 | 3.12% | 2.641 |

Roster is frozen within the study. Similar historical opponents have one representative; full artifacts and small-gain candidates remain archived.
Consumed development panels support adaptive selection, not unseen final validation. No automatic publication. Rear scores do not gate continuation.
