> Latest completed-cycle selection: s47236_step_002621440, **1319W309D100L/1728** across all54 originalCPU opponents (+9wins vs previous reference). See [completed cycle review](COMPLETED_CYCLE_REVIEW.md). Next learning is being prepared; no training process is currently active. All checkpoints retained.

# Encounter actor CPU PPO progress

Frozen qualified parent actor before the first range exit; independently trainable actor afterward. The event uses public distance history, not elapsed time. Both actors start from the same s47223 weights. Original CPU JSBSim, controller, rewards and PPO settings are unchanged. 43 original features plus one history flag; 39 public observations. The 7,561 frozen opening parameters are outside Adam; original trainable tensors, Adam and counters are preserved at initialization.

Qualified parent: 1310/1728 all54; 506/672 representative panel. 8 CPU processes / 32 environments; asynchronous evaluation.

Budget: 10,485,760 additional decisions. Save every 655,360; monitor every 2,621,440. 128-game monitor / 672-game representative development; selected candidates require all54 paired comparison before promotion.

| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Overlapped wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| 47236 | 10,485,760 | 11,850 | 3424.84 | 1,600 | 810.03 | 4810.36 |

Live: 10,485,760 added decisions / 11,850 training games. Saved counters can lag live progress.

New completed evaluations: 1600. Reused parent: 672 unique games / 800 panel entries.
Overall wall: 4813.79s; completed evaluation process time: 810.03s. Stage times overlap; do not sum them.

Training starts: 50% normal / 50% existing native-prefix attempts. Successful prefixes hand over the same simulator when closing within 4,000m after the first surviving range exit. Failed attempts reset the same foe and initial condition. Prefix steps are excluded from PPO decisions, but prefix/reset execution is included in learning wall time. Prefix loop seconds below sum overlapping workers and must not be added to learning wall. The reused uniform opening bank has constant zero policy penalty under the frozen opening actor. No new bank collection.

| Seed | Prefix attempts | Surviving handoffs | Fallbacks | Extra prefix native steps | Sum prefix seconds |
|---|---:|---:|---:|---:|---:|
| 47236 | 5,908 | 5,601 | 307 | 6,551,080 | 2140.13 |

| Model / panel | Added steps | Wins / games | Win rate | Gain vs parent |
|---|---:|---:|---:|---:|
| parent / monitor | 0 | 60 / 128 | 46.88% | baseline |
| parent / development | 0 | 506 / 672 | 75.30% | baseline |
| 47236 / development | 2,621,440 | 508 / 672 | 75.60% | +0.30pp |
| 47236 / monitor | 2,621,440 | 60 / 128 | 46.88% | +0.00pp |
| 47236 / monitor | 5,242,880 | 59 / 128 | 46.09% | -0.78pp |
| 47236 / monitor | 7,864,320 | 60 / 128 | 46.88% | +0.00pp |
| 47236 / development | 10,485,760 | 508 / 672 | 75.60% | +0.30pp |
| 47236 / monitor | 10,485,760 | 60 / 128 | 46.88% | +0.00pp |

All54 foes and all checkpoints retained, including small gains and specialists. Consumed development supports adaptive selection, not unseen validation. No automatic publication.
