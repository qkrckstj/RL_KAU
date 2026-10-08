# Batched GPU FairFight backend

## Intent and authorization

The user explicitly requested changing simulation structure to execute many
simulations on GPU. Preserve archived research, use a separate optional backend,
and measure fidelity as well as throughput. Do not replace the official evaluator
or claim equivalent research results from a newly implemented simulator.

## Architecture

Use the optional `jsbsim-f16-cuda` physics package pinned to commit
`1462017f6a5c724f9780c5d65b0c9552538ef426`, version 0.2.1, GPL-3.0-or-later.
Source: https://github.com/juyoung020/jsbsim-f16-cuda . This ports the F-16
equations and FCS to CUDA; it is not a learned dynamics surrogate. It still has
documented flat-Earth and interpolated trim differences, so this backend is
experimental until matched against this project's trajectories and verdicts.

New code belongs in `experiments/gpu_sim/`; it must not alter `aircombat_gym/`,
archived trainer sources, or existing policies. Python/Torch owns batched state,
the CUDA package advances each aircraft, and a tensor autopilot updates every
1/120 s physics frame. Six frames remain one 20 Hz decision. Holding a single
autopilot command for six frames is forbidden because that changes control.

Represent aircraft as 2*N rows, interleaved red/blue, and matches as N rows.
Keep physics, control memory, health, tracking, observations, opponent action,
and PPO rollout tensors on CUDA. CPU is used for setup, diagnostics, checkpoint
I/O, and the official comparison only. No per-aircraft Python step loop or
per-step NumPy conversion is allowed in the batched path.

Expose `BatchedFairFight(n, device, dtype)` with reset/masked-reset, dual-seat
discrete actions, raw `(N,2,39)` observations, terminal flags and signed outcomes.
Maintain the existing nine heading/speed actions and persistent zero-action
targets. Closed vertical means altitude-holding autopilot, not fixed height.
Use the official weapon geometry, lock, simultaneous damage and mutual tolerance.
Return terminal observations before explicit reset; mask completed slots until
reset. Nonfinite physics must be reported as invalid simulation, never a win.

Tensor preprocessing produces the same 31 public features as Plan A. A bounded
GPU PPO trainer operates on those features, against a tensor scripted opponent,
with GPU rollout storage and minibatches. This is a new experiment, with its
own config/seed/hash/checkpoints, not an exact archived-run continuation.
Do not train a new champion or use consumed final test bands.

## Validation and success criteria

1. Tensor autopilot and geometry agree with scalar source on seeded states,
   including angle wrapping, clipping, persistent targets, and lock reset.
2. Seat swapping permutes observations and reverses wins; simultaneous death
   and mutual tolerance match the official judge. Partial resets cannot mutate
   other aircraft state or control memory.
3. Run installed upstream CUDA verification, then paired open-loop aircraft
   trajectories with original JSBSim at the actual latitude, altitude and speed
   band. Report position/speed/attitude errors at horizons, not just throughput.
4. Compare closed-loop policy games on shared development seeds in both seats.
   Report verdict agreement and divergence; a failure prevents calling it a
   drop-in replacement. Keep the official default unchanged regardless.
5. Benchmark fresh equal-length workloads at multiple batch sizes with CUDA
   synchronization, warmup excluded and setup costs reported separately. Compare
   simulator throughput separately from end-to-end PPO updates.
6. Demonstrate finite actual GPU PPO weight updates and checkpoint reload.
   Existing tests still pass. Save raw evidence and an independently reviewed
   report. No numerical speedup or policy-quality result is promised in advance.

## Limits and alternatives

CPU processes preserve the official backend but do not satisfy GPU simulation.
A learned or hand-simplified flight surrogate is not introduced without an
explicit change of scope. A full globe/trim-exact CUDA rewrite is substantially
larger than integrating the existing port; validation determines the residual
work. Do not hide that boundary by describing approximate trajectories as exact.
