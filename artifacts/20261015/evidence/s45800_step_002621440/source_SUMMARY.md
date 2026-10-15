# Predictive aiming CPU learning

Original native JSBSim; 8 physical workers /32 environments. Fixed aiming and HP-exchange reward. All models retained.

| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Wall seconds (overlap included) |
|---|---:|---:|---:|---:|---:|---:|
| 45800 | 5,242,880 | 6,028 | 558.56 | 1,184 | 581.49 | 960.44 |
| 45801 | 5,242,880 | 5,959 | 564.83 | 1,184 | 587.97 | 975.16 |

Live: seed 45801, 5,242,880 added decisions /5,959 completed training games. Saved counters above lag live progress.

New completed evaluations: 2368; reused parent games: 800.
Elapsed wall time: 1937.49s; completed evaluation process time: 1169.46s. Training and evaluation overlap.

| Model /panel | Added steps | Wins /games | Win rate | Gain vs same-roster parent | Dealt /received HP | First-shot rate | Connected firing seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| parent /development | 0 | 493 /672 | 73.36% | baseline | 0.9261 /0.5878 | not recorded | not recorded |
| parent /monitor | 0 | 109 /128 | 85.16% | baseline | 0.9954 /0.5176 | 45.31% | 3.036 |
| 45800 /monitor | 1,310,720 | 107 /128 | 83.59% | -1.56pp | 0.9956 /0.5229 | 45.31% | 3.036 |
| 45800 /monitor | 2,621,440 | 110 /128 | 85.94% | +0.78pp | 0.9906 /0.5287 | 43.75% | 3.020 |
| 45800 /monitor | 3,932,160 | 100 /128 | 78.12% | -7.03pp | 0.9935 /0.5307 | 44.53% | 3.029 |
| 45800 /development | 5,242,880 | 493 /672 | 73.36% | +0.00pp | 0.9264 /0.5890 | 23.51% | 2.824 |
| 45800 /monitor | 5,242,880 | 106 /128 | 82.81% | -2.34pp | 0.9947 /0.5193 | 45.31% | 3.033 |
| 45801 /monitor | 1,310,720 | 108 /128 | 84.38% | -0.78pp | 0.9969 /0.5312 | 43.75% | 3.040 |
| 45801 /monitor | 2,621,440 | 107 /128 | 83.59% | -1.56pp | 0.9958 /0.5258 | 44.53% | 3.037 |
| 45801 /monitor | 3,932,160 | 104 /128 | 81.25% | -3.91pp | 0.9930 /0.5105 | 46.88% | 3.028 |
| 45801 /development | 5,242,880 | 492 /672 | 73.21% | -0.15pp | 0.9303 /0.5916 | 27.38% | 2.836 |
| 45801 /monitor | 5,242,880 | 107 /128 | 83.59% | -1.56pp | 0.9945 /0.5147 | 47.66% | 3.033 |

Roster is frozen within the study. Similar historical opponents have one representative; full artifacts and small-gain candidates remain archived.
Consumed development panels support adaptive selection, not unseen final validation. No automatic publication. Rear scores do not gate continuation.
