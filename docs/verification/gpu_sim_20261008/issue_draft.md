# Run many air-combat simulations on CUDA

Original JSBSim simulation is serial within each environment and dominates small
neural-policy training. The user requested large-scale GPU simulation rather
than only moving a small neural network to CUDA.

Implement a separately selectable, pinned GPU physics backend and batched
combat interface, connect GPU rollouts to PPO, preserve archived sources and
official JSBSim, and measure trajectory fidelity and actual throughput.

Implementation and execution validation are complete locally. Fidelity failed:
the GPU physics is not accepted as an equivalent official backend. Further
coordinate/trim physics-port work is needed before changing that boundary.
See `docs/GPU_SIMULATION_20261008.md` and the retained evidence.

Remote issue creation was not possible because GitHub authentication is invalid.
