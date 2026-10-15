# Predictive aiming CPU learning

Original native JSBSim; 8 physical workers /32 environments. Fixed aiming and HP-exchange reward. All models retained.

Fixed teacher prior6.0;44learner features/39public channels. One public maintained-heading error cue from own actionhistory;current reward0.05/12km retained. No new elapsed-time tactical switch. Original columns/Adam/counters retained;new columns start0. ZIP bytes change. Initial inference verified on consumed1609frame trace;cached reference games reused.

| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Wall seconds (overlap included) |
|---|---:|---:|---:|---:|---:|---:|
| 46500 | 5,242,880 | 5,974 | 573.42 | 1,056 | 542.34 | 929.56 |
| 46501 | 5,242,880 | 5,902 | 575.42 | 1,056 | 546.25 | 926.84 |

Live: seed 46501, 5,242,880 added decisions /5,902 completed training games. Saved counters above lag live progress.

New completed evaluations: 2112; reused parent games: 672.
Elapsed wall time: 1858.08s; completed evaluation process time: 1088.59s. Training and evaluation overlap.

| Model /panel | Added steps | Wins /games | Win rate | Gain vs unchanged prior6comparator | Dealt /received HP | First-shot rate | Connected firing seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
Cached unchanged prior6comparator: 672unique matches /800panel entries. Hard monitor:8foes/4IC/both seats/both samplers;full acceptance21foes/8IC/both seats/both samplers.

| parent-prior6-comparator /development | 0 | 498 /672 | 74.11% | baseline | 0.9260 /0.5939 | 22.62% | 2.823 |
| parent-prior6-comparator /hard-monitor | 0 | 54 /128 | 42.19% | baseline | 0.8590 /0.8541 | 2.34% | 2.618 |
| 46500 /hard-monitor | 1,310,720 | 50 /128 | 39.06% | -3.12pp | 0.8579 /0.8619 | 2.34% | 2.614 |
| 46500 /hard-monitor | 2,621,440 | 52 /128 | 40.62% | -1.56pp | 0.8628 /0.8551 | 3.12% | 2.630 |
| 46500 /hard-monitor | 3,932,160 | 39 /128 | 30.47% | -11.72pp | 0.8479 /0.8637 | 2.34% | 2.584 |
| 46500 /development | 5,242,880 | 482 /672 | 71.73% | -2.38pp | 0.9270 /0.5859 | 26.04% | 2.826 |
| 46500 /hard-monitor | 5,242,880 | 40 /128 | 31.25% | -10.94pp | 0.8584 /0.8606 | 3.12% | 2.616 |
| 46501 /hard-monitor | 1,310,720 | 51 /128 | 39.84% | -2.34pp | 0.8611 /0.8593 | 3.12% | 2.624 |
| 46501 /hard-monitor | 2,621,440 | 44 /128 | 34.38% | -7.81pp | 0.8520 /0.8604 | 3.12% | 2.596 |
| 46501 /hard-monitor | 3,932,160 | 46 /128 | 35.94% | -6.25pp | 0.8640 /0.8589 | 2.34% | 2.632 |
| 46501 /development | 5,242,880 | 466 /672 | 69.35% | -4.76pp | 0.9266 /0.5906 | 23.66% | 2.825 |
| 46501 /hard-monitor | 5,242,880 | 44 /128 | 34.38% | -7.81pp | 0.8576 /0.8584 | 0.78% | 2.614 |

Roster is frozen within the study. Similar historical opponents have one representative; full artifacts and small-gain candidates remain archived.
Consumed development panels support adaptive selection, not unseen final validation. No automatic publication. Rear scores do not gate continuation.
