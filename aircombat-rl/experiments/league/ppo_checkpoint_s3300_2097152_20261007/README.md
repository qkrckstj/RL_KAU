# PPO development checkpoint: inference package

This is the preserved RNG3300 checkpoint after 2,097,152 additional interactions,
following a shared 1,032,200-interaction prefix. It is a development candidate,
not the final selected tournament policy. Its 24-opponent,96-game development
comparison was64 wins,16 draws,16 losses versus the teacher52/24/20.

The ZIP includes the original teacher parameters and the learned NumPy actor.
Original paths inside policy_net.json are provenance only. Keep policy.py,
wrappers.py and policy_net.zip together. The official aircombat_gym package,
NumPy and its normal simulator dependencies must be installed.

From this folder, run `python verify.py` to check hashes and fixed synthetic
actions without importing Torch or training helpers. In the course repository,
the normal grader can load this folder with its --design option and the bundled
policy_net.zip. A new grader run creates new evaluation games; this packaging
check did not run any flights.

This is an inference package. It does not contain the PPO critic/optimizer,
opponent archive or mid-episode simulator state. Continue learning using the
original run's learner.zip, frozen plan and source snapshot, and explicitly
fresh episode streams. Do not treat this ZIP alone as a full training replay.
