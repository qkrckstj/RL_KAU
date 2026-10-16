# CPU rear training with asynchronous evaluation

Use `python -m tools.rear_async_cpu launch --out runs/<new-run> --workers 8` with BLAS and OpenMP thread limits set to one. The native CPU simulator and current rear-training rewards are preserved. A separate evaluator consumes hash-verified immutable snapshots, allowing training to continue inside the approved budget.

See `../../../docs/REAR_CPU_ASYNC_20261012.md` for worker measurements, budgets, candidate retention, restart limits and integration evidence. The implementation changes execution and reporting rather than automatically selecting new reward coefficients. All checkpoints remain candidates, including small win-rate gains.
