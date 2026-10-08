# Connect DQN and CEM to the experimental CUDA simulator

The user requested GPU simulation support for DQN and CEM after the native PPO
entrypoint was implemented. Add GPU replay/Q-network learning and paired
population search, preserve archived sources, save reproducible artifacts, and
verify exports in original JSBSim.

Implemented locally with tests and actual CUDA runs. The existing simulator
trajectory-fidelity gate remains failed; the new trainers are experimental.
See docs/GPU_DQN_CEM_20261008.md for measurements and execution results.

Remote issue creation is unavailable because GitHub authentication is invalid.
