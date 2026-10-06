# Flight diagnostics

Same 20 validation engagements; hold4 is a diagnostic intervention, not retraining.

| Model | Hold ticks | Wins | Action changes/s | Heading reversals/s | Own WEZ s | Enemy WEZ s | Longest track mean s |
|---|---:|---:|---:|---:|---:|---:|---:|
| starting_good | 1 | 9/20 | 0.05 | 0.05 | 4.62 | 4.38 | 2.86 |
| starting_good | 4 | 10/20 | 0.04 | 0.04 | 4.62 | 4.37 | 2.87 |
| continued_bad | 1 | 0/20 | 0.35 | 0.07 | 1.37 | 4.80 | 0.90 |
| continued_bad | 4 | 0/20 | 0.28 | 0.05 | 1.32 | 4.69 | 0.95 |
| large_buffer_bad | 1 | 0/20 | 1.17 | 0.15 | 0.20 | 4.61 | 0.20 |
| large_buffer_bad | 4 | 0/20 | 0.90 | 0.09 | 0.21 | 4.59 | 0.21 |
| small_buffer_recovered | 1 | 8/20 | 0.64 | 0.57 | 4.03 | 4.76 | 2.39 |
| small_buffer_recovered | 4 | 6/20 | 0.31 | 0.24 | 3.48 | 4.80 | 1.98 |
