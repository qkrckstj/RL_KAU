# GPU DQN and CEM — implemented and measured

The experimental GPU simulator now supports **DQN, CEM and the existing PPO**.
DQN collects batched transitions into CUDA replay and updates CUDA Q/target
networks. CEM evaluates entire paired games for many parameter candidates and
updates its search distribution on CUDA. Both new exports run through the
unchanged original JSBSim policy loader.

**The simulator's existing trajectory-fidelity failure remains.** These trainers
do not resolve the roughly 258 m maximum 120-second float32 trajectory error
reported in [the GPU physics audit](GPU_SIMULATION_20261008.md). Official JSBSim,
archived trainers and policies, and the prior GPU PPO implementation are unchanged.
No challenger is promoted and no final test band is opened.

## What was added

- `aircombat-rl/experiments/gpu_sim/dqn`: vanilla DQN, 31 features, nine actions,
  64x64 network, GPU replay, epsilon-greedy collection, target copies and CPU
  evaluator policy adapter. Terminal next observations are copied before reset;
  actual endings, including the finite 120-second timeout, disable bootstrap.
- `aircombat-rl/experiments/gpu_sim/cem`: bounded 17-coefficient population search,
  candidate x opponent x shared IC x physical seat evaluation, complete episode
  scoring, elite distribution updates and standard reactive-policy JSON export.
- `run_utils.py`: fresh-output protection, source snapshots, metadata and failure
  records. Source snapshots include the new trainers and their local physics,
  observation and scalar-controller dependencies.

See [installation, flags and runnable commands](../aircombat-rl/experiments/gpu_sim/README.md).
The optional physics dependency remains the pinned published GPL-3.0-or-later
[F-16 CUDA port](https://github.com/juyoung020/jsbsim-f16-cuda), commit
`1462017f6a5c724f9780c5d65b0c9552538ef426`. No additional package was required.

## Actual CUDA runs

RTX 5080, Ryzen 7 9800X3D, Torch 2.14.1+cu130. Each workload ran alone. Timing
synchronizes CUDA and separates setup from collection/optimization/search.
Final checkpoint persistence is outside the loop timer. This is execution and
throughput evidence, not an equal-quality CPU training comparison.

| Measurement | DQN | CEM |
|---|---:|---:|
| Parallel games | 1,024 | 1,024 |
| Collected / effective environment steps | 4,194,304 | 4,718,109 |
| Timed loop | 16.720 s | 18.022 s |
| Setup | 0.890 s | 0.413 s |
| Effective steps/s | 250,855 | 261,799 |
| Learning/search updates | 1,017 optimizer steps | 3 distribution updates |
| Completed games/episodes | 3,577 | 3,072 |
| Invalid games/episodes | 0 | 0 |
| Peak Torch allocation | 212,585,984 bytes | 105,771,008 bytes |

DQN used `--envs 1024 --steps 4096` and other defaults: 262,144 replay capacity,
32,768 warmup transitions, one optimizer step per four vector decisions,
4,096-sample minibatches, gamma 0.999, learning rate 3e-4, target copy every 100
updates. Ten target copies occurred and finite changed CUDA weights were checked.
The short training run recorded six wins among 3,577 completed episodes. It has
not produced a strong policy.

CEM used 64 candidates, eight elites, four shared ICs, two seats, two opponents,
three generations: 1,024 complete games/generation. The opponents were tensor
horizontal Ace and embedded original CEM. All candidates/generations used the
same recorded **training** IC set. It dispatched 7,375,872 batch slots, of which
4,718,109 were active game steps; finished slots were frozen. Only active steps
are used for the reported effective throughput.

CEM started from the frozen final archived policy. The best training game score
was already 1.0 in generation one and remained 1.0; the chosen candidate improved
only the small health tie breaker (objective 1.000665875 to 1.000727750).
Distribution parameters changed and all generations completed, but this is not
evidence of a stronger general policy or improved win rate over the parent.

## Original JSBSim execution checks

Both saved policies loaded through `tools.policies` and `tools.league_matches.duel`.
Each faced original Ace and original CEM in both seats on the already consumed
open development seed 33030000, giving four games per policy:

| Saved GPU-run policy | Against Ace | Against original CEM | Total |
|---|---|---|---|
| DQN | 0 wins, 2 losses | 2 draws | 0 wins, 2 losses, 2 draws |
| CEM | 2 wins | 2 wins | 4 wins |

These establish loading and execution compatibility. The CEM run inherited a
strong parent, the sample is tiny, and these familiar conditions are not an
unseen strength test. No policy was selected using these JSBSim results.

## Validation, review and reproducibility

Full suite: **145 passed**, four existing warnings, 32.35 s. `pip check` passed.
The nine new tests cover replay wrap and input-buffer mutation, terminal TD
targets, actual timeout collection before reset, target-copy cadence, changed
CUDA weights, CPU policy loading, paired CEM layout, elite fitting and complete
CUDA candidate games. Invalid and unfinished game codes are rejected.

Independent review found a population-size-two issue: retaining both incumbent
and mean left no random challenger. A regression reproduced the failure, and
the final sampler reserves the mean slot only for populations larger than two.
The review also requested direct terminal-collection coverage; the added test
uses actual CUDA physics at the timeout boundary and verifies replay after reset.
The follow-up review found no remaining important issue.

The measured 64-candidate run predates that narrow fix; the default-size sampling
behavior is unchanged. Exact measured source snapshots are retained, distinct
from the final source. The full suite ran on the final code.

Evidence is under [verification/gpu_dqn_cem_20261008](verification/gpu_dqn_cem_20261008/):
raw metrics, logs, tests, JSBSim games, review record and an archive containing
complete runs, policies and source snapshots. `manifest.json` records hashes.
Extract the archive from `aircombat-rl` only into fresh output paths.

Each new entrypoint saves its checkpoint at successful completion; resuming is
not implemented and DQN replay is not included in its policy checkpoint. This
does not change the archived trainers' own resume capabilities. A longer run or
larger batch is not guaranteed to improve policy quality.

Local branch: `feat/gpu-dqn-cem`. GitHub authentication was checked and is invalid;
issue/PR drafts are retained locally. No remote push, PR or merge is claimed.
