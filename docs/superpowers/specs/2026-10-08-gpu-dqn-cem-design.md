# GPU DQN and CEM design

The user requested the same GPU simulation support for DQN and CEM that PPO
already has. Success means actual collection/search and updates on CUDA, saved
policies that execute in original JSBSim, and retained reproducible evidence.
This is an additive extension of the experimental backend, not acceptance of
its known failed physics-fidelity gate. No previous policy is overwritten.

## DQN

Use the existing 31 features and nine actions with a 64x64 ReLU Q network,
uniform GPU ring replay, epsilon-greedy collection from many environments,
Huber TD loss and a separate periodically copied target network. Store next
observations before resetting; all actual episode endings suppress bootstrap.
Alternate learner seats. Use existing tensor horizontal Ace and terminal reward
(+1 win, -0.2 other ending, zero otherwise). Count collection in total transitions
and target updates in optimizer steps explicitly. This is a new vanilla DQN
configuration, not a resumed SB3 run. No Double-DQN feature or resume is required.

## CEM

Search the archived 17-parameter reactive-controller family within existing
bounds. Start from the frozen final policy; retain it unchanged on disk. Batch
candidate x opponent x shared IC x physical seat. Default opponents are tensor
horizontal Ace and embedded original CEM. All candidates and generations share
the same sampled training IC set, so incumbent comparisons are well defined.
Each game reaches its actual terminal condition or 120-second timeout. Invalid
or incomplete games reject the generation; they never score as draws.

Optimize mean game score (win 1, draw 0.5, loss 0), plus 0.001 times mean final
health difference as a tie breaker. Retain the incumbent in each population.
Fit a diagonal Gaussian in normalized parameter space to elites with 0.5
smoothing and a minimum standard deviation. Export ordinary controller JSON,
population results, distribution state and RNG state. This training selection
does not promote a league champion and does not open a final test band.

## Boundaries and evidence

Separate entrypoints/design directories under experiments/gpu_sim/dqn and cem.
Share only new run-record helpers; existing PPO, physics, rules and archives
remain byte-identical. Fresh output directories, source snapshots, dependency
metadata, CUDA synchronization around timers, failure records and checkpoints.
Report setup versus training/search and effective completed game steps versus
dispatched batch slots; never equate sample throughput with learning quality.

Tests cover replay wrap/copy, TD target termination, deterministic elite fitting,
candidate/IC/seat layout, invalid rejection, actual CUDA updates, policy loading
and duplicate output protection. Run original JSBSim checks on consumed open
development seed 33030000 in both seats. Keep all successful and failed evidence.
