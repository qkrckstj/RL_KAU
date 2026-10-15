# Predictive aiming CPU learning

Original native JSBSim; 8 physical workers /32 environments. Fixed aiming and HP-exchange reward. All models retained.

Teacher prior:base6.0,conflict3.0under a fixed public-state rule. No elapsed-time switch. Cached cells evaluate the unchanged prior6comparator,not the gated learner at zero steps.

| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Wall seconds (overlap included) |
|---|---:|---:|---:|---:|---:|---:|
| 46200 | 5,242,880 | 5,882 | 579.54 | 1,056 | 562.03 | 943.51 |
| 46201 | 5,242,880 | 5,845 | 573.16 | 1,056 | 557.92 | 930.73 |

Live: seed 46201, 5,242,880 added decisions /5,845 completed training games. Saved counters above lag live progress.

New completed evaluations: 2112; reused parent games: 672.
Elapsed wall time: 1875.93s; completed evaluation process time: 1119.96s. Training and evaluation overlap.

| Model /panel | Added steps | Wins /games | Win rate | Gain vs unchanged prior6comparator | Dealt /received HP | First-shot rate | Connected firing seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
Cached unchanged prior6comparator: 672unique matches /800panel entries. Hard monitor:8foes/4IC/both seats/both samplers;full acceptance21foes/8IC/both seats/both samplers.

| parent-prior6-comparator /hard-monitor | 0 | 54 /128 | 42.19% | baseline | 0.8590 /0.8541 | 2.34% | 2.618 |
| parent-prior6-comparator /development | 0 | 498 /672 | 74.11% | baseline | 0.9260 /0.5939 | 22.62% | 2.823 |
| 46200 /hard-monitor | 1,310,720 | 57 /128 | 44.53% | +2.34pp | 0.8583 /0.8551 | 2.34% | 2.615 |
| 46200 /hard-monitor | 2,621,440 | 45 /128 | 35.16% | -7.03pp | 0.8535 /0.8596 | 2.34% | 2.602 |
| 46200 /hard-monitor | 3,932,160 | 56 /128 | 43.75% | +1.56pp | 0.8565 /0.8539 | 3.12% | 2.611 |
| 46200 /development | 5,242,880 | 476 /672 | 70.83% | -3.27pp | 0.9299 /0.5882 | 25.60% | 2.835 |
| 46200 /hard-monitor | 5,242,880 | 43 /128 | 33.59% | -8.59pp | 0.8653 /0.8591 | 3.12% | 2.637 |
| 46201 /hard-monitor | 1,310,720 | 53 /128 | 41.41% | -0.78pp | 0.8529 /0.8560 | 2.34% | 2.598 |
| 46201 /hard-monitor | 2,621,440 | 55 /128 | 42.97% | +0.78pp | 0.8575 /0.8591 | 2.34% | 2.614 |
| 46201 /hard-monitor | 3,932,160 | 53 /128 | 41.41% | -0.78pp | 0.8665 /0.8556 | 3.12% | 2.642 |
| 46201 /development | 5,242,880 | 495 /672 | 73.66% | -0.45pp | 0.9299 /0.5953 | 24.40% | 2.835 |
| 46201 /hard-monitor | 5,242,880 | 55 /128 | 42.97% | +0.78pp | 0.8508 /0.8594 | 3.12% | 2.593 |

Roster is frozen within the study. Similar historical opponents have one representative; full artifacts and small-gain candidates remain archived.
Consumed development panels support adaptive selection, not unseen final validation. No automatic publication. Rear scores do not gate continuation.
