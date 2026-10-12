# CPU rear execution review

Code review: checkpoint publication is atomic, payload hashes are checked before copy and evaluation, and partial publication is not discoverable as a ready checkpoint. Training and evaluation hold separate process locks. Evaluator copies are independent of live learner state. A failed evaluation reuses completed cells and cannot change the learner RNG or optimizer.

Architecture review: one continuing trainer, separate low-priority evaluator, immutable checkpoint queue, and separate candidate records. Every checkpoint remains available. The minimum budget is not shortened by the legacy two-point win gate, and extensions are bounded by rear skill trends. Restart restores optimizer/RNG/opponent sampling state and avoids counting complete steps twice; it starts new physical episodes. No automatic frontline promotion or external publication is implemented.

Validation: two reversed worker benchmark passes, exact per-pass final policy hashes, real asynchronous integration, exact initial actor/critic/optimizer restoration, export agreement, and seven focused tests. Tests cover the new publication, retry and judgment boundaries; broad prior simulator suites were not repeated.

Limitations: local and archived Windows replay trajectories differ; same-machine native baselines are used. A separate laptop evaluator is not automatically connected. Async evaluation can still compete for CPU time, and the minimum-budget endpoint intentionally waits for required assessments. Worker count is optimal only among the measured configurations. Checkpoint counters do not reconstruct all ancestral training steps.
