# DQN learning-rate comparison

Test band: 900000; matches/model: 2

| Learning rate | Budget | Selection | Wins by training seed | Mean |
|---|---:|---|---|---:|
| 0.001 | 2048 | validation_selected | [0] | 0.00% |
| 0.001 | 2048 | budget_endpoint | [0] | 0.00% |
| 0.0001 | 2048 | validation_selected | [0] | 0.00% |
| 0.0001 | 2048 | budget_endpoint | [0] | 0.00% |

## Final replay diagnostics

| Condition | Seed | Q max | Out-of-bound fraction | Last five validation wins |
|---|---:|---:|---:|---|
| lr_1e3 | 0 | 0.209 | 0.00% | [0, 0] |
| lr_1e4 | 0 | 0.228 | 0.00% | [0, 0] |
