# Add native GPU DQN and CEM training

## Why

The experimental batched CUDA simulator only had a PPO learner. DQN and CEM
still used archived CPU simulation paths. Add separate GPU-native entrypoints
for both requested algorithms without redirecting the official evaluator.

## What changed

DQN adds GPU replay, epsilon-greedy batched collection, a target network, Huber
TD updates and an original-evaluator policy adapter. CEM adds full paired game
evaluation across candidates/opponents/shared ICs/seats, elite fitting and
standard controller JSON exports. Both save fresh run records and exact sources.
Existing PPO, official simulation, archives and champions remain unchanged.

## How validated

145 tests passed; pip check passed. DQN collected 4,194,304 transitions and ran
1,017 updates in 16.720 s. CEM completed 3,072 games and three distribution
updates in 18.022 s. Neither run had invalid episodes. Both exports loaded in
original JSBSim in both seats against Ace and original CEM: DQN 2 losses/2 draws,
CEM 4 wins on a tiny consumed development fixture. Exact measured sources saved.

## Risks

The prior physics-equivalence gate remains failed. No equal-quality CPU speedup
is claimed. CEM starts from a strong archived parent; its small result is not
evidence of broader strength or parent win-rate improvement. Native trainer
resume is absent; DQN checkpoints do not contain replay. Checkpoints are saved
at successful completion. No new final test or champion promotion.

## Reviewer focus

Terminal transition copying before reset, replay wrap, target-copy units, shared
IC/seat layout, random challengers with population=2, invalid game rejection,
and experimental reporting boundaries. This is a local draft, not a remote PR.
