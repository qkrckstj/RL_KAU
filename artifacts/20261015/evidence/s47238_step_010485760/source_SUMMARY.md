# Conditional speed exploration CPU PPO progress

Continue the qualified parent with full learned/frozen tensors, Adam and counters preserved. Reference: 1319/1728 originalCPU all54 games. Single intervention: conditional speed temperature 2 after the existing range-history event. Opening behavior is unchanged; at a fixed observation each heading marginal is preserved while conditional speed choices become more diverse. Suffix actions and closed-loop trajectories intentionally can differ. All54 original opponent weights, native CPU simulator/controller, rewards, PPO settings and evaluation conditions remain. No elapsed-time gate or forced speed command.

Qualified parent: 1319/1728 all54; 508/672 representative panel. 8 CPU processes / 32 environments; asynchronous evaluation.

Budget: 10,485,760 additional decisions. Save every 655,360; monitor every 2,621,440. 128-game monitor / 672-game representative development; selected candidates require all54 paired comparison before promotion.

| Seed | Saved added steps | Training games | Learning seconds | New evaluation games | Evaluation seconds | Overlapped wall seconds |
|---|---:|---:|---:|---:|---:|---:|
| 47238 | 10,485,760 | 12,002 | 3532.67 | 1,056 | 535.49 | 3898.20 |

Live: 10,485,760 added decisions / 12,002 training games. Saved counters can lag live progress.

New completed evaluations: 1056. Reused parent: 672 unique games / 800 panel entries.
Overall wall: 3902.43s; completed evaluation process time: 535.49s. Stage times overlap; do not sum them.

Training starts: 50% normal / 50% existing native-prefix attempts. Successful prefixes hand over the same simulator when closing within 4,000m after the first surviving range exit. Failed attempts reset the same foe and initial condition. Prefix steps are excluded from PPO decisions, but prefix/reset execution is included in learning wall time. Prefix loop seconds below sum overlapping workers and must not be added to learning wall. The reused uniform opening bank has constant zero policy penalty under the frozen opening actor. No new bank collection.

| Seed | Prefix attempts | Surviving handoffs | Fallbacks | Extra prefix native steps | Sum prefix seconds |
|---|---:|---:|---:|---:|---:|
| 47238 | 6,099 | 5,795 | 304 | 6,733,030 | 2199.51 |

| Model / panel | Added steps | Wins / games | Win rate | Gain vs parent |
|---|---:|---:|---:|---:|
| parent / development | 0 | 508 / 672 | 75.60% | baseline |
| parent / monitor | 0 | 60 / 128 | 46.88% | baseline |
| 47238 / monitor | 2,621,440 | 60 / 128 | 46.88% | +0.00pp |
| 47238 / monitor | 5,242,880 | 59 / 128 | 46.09% | -0.78pp |
| 47238 / monitor | 7,864,320 | 60 / 128 | 46.88% | +0.00pp |
| 47238 / development | 10,485,760 | 507 / 672 | 75.45% | -0.15pp |
| 47238 / monitor | 10,485,760 | 61 / 128 | 47.66% | +0.78pp |

All54 foes and all checkpoints retained, including small gains and specialists. Consumed development supports adaptive selection, not unseen validation. No automatic publication.
