# DQN learning-rate comparison

Test band: 1400000; matches/model: 40

| Learning rate | Budget | Selection | Wins by training seed | Mean |
|---|---:|---|---|---:|
| 0.001 | 204800 | validation_selected | [0, 0, 6] | 5.00% |
| 0.001 | 512000 | validation_selected | [0, 0, 6] | 5.00% |
| 0.001 | 1024000 | validation_selected | [13, 2, 6] | 17.50% |
| 0.001 | 204800 | budget_endpoint | [0, 0, 6] | 5.00% |
| 0.001 | 512000 | budget_endpoint | [0, 0, 0] | 0.00% |
| 0.001 | 1024000 | budget_endpoint | [0, 0, 0] | 0.00% |
| 0.0001 | 204800 | validation_selected | [0, 0, 1] | 0.83% |
| 0.0001 | 512000 | validation_selected | [0, 0, 1] | 0.83% |
| 0.0001 | 1024000 | validation_selected | [0, 1, 1] | 1.67% |
| 0.0001 | 204800 | budget_endpoint | [0, 0, 1] | 0.83% |
| 0.0001 | 512000 | budget_endpoint | [0, 0, 0] | 0.00% |
| 0.0001 | 1024000 | budget_endpoint | [0, 0, 0] | 0.00% |

## Final replay diagnostics

| Condition | Seed | Q max | Out-of-bound fraction | Last five validation wins |
|---|---:|---:|---:|---|
| lr_1e3 | 0 | 23.375 | 99.54% | [5, 0, 0, 0, 0] |
| lr_1e3 | 1 | 14.158 | 97.81% | [0, 0, 0, 4, 0] |
| lr_1e3 | 2 | 13.388 | 99.20% | [0, 1, 1, 0, 0] |
| lr_1e4 | 0 | 2.653 | 84.92% | [0, 0, 0, 0, 0] |
| lr_1e4 | 1 | 1.998 | 55.07% | [1, 0, 0, 0, 0] |
| lr_1e4 | 2 | 2.507 | 70.39% | [0, 0, 0, 0, 0] |
