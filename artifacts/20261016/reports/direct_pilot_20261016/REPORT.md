# Direct pilot and evasion study — 2026-10-16

Agent-designed public-observation feedback controllers. Neural training decisions: **0**. Original physics, weapons, health-round4 verdict and all seven opponents were preserved. All conditions below are consumed adaptive development evidence; no unseen confirmation or admission occurred.

## Completed comparisons

| Run | Games | W / D / L | Mean received HP | No-hit wins / draws | Rear damage games | Seconds |
|---|---:|---|---:|---|---:|---:|
| grid_000 | 96 | 2 / 9 / 85 | 0.9204 | 0 / 0 | 0 | 16.829 |
| screen_000 | 224 | 62 / 78 / 84 | 0.8023 | 0 / 0 | 0 | 54.324 |
| grid_001 | 144 | 15 / 6 / 123 | 0.8979 | 0 / 0 | 2 | 27.130 |
| grid_002 | 120 | 2 / 0 / 118 | 0.9074 | 0 / 0 | 5 | 25.814 |
| screen_001 | 224 | 26 / 0 / 198 | 0.9468 | 0 / 0 | 28 | 39.069 |
| grid_003 | 288 | 4 / 4 / 280 | 0.9233 | 0 / 4 | 4 | 44.749 |
| screen_002 | 224 | 0 / 11 / 213 | 0.5269 | 0 / 11 | 0 | 68.783 |

The final evasion grid crossed 72 combinations of left/right direction, onset distance, duration and speed, using two consumed initial conditions and both seats against original44701/action4900 (288 games). Development ranking distinguishes no-hit victories from no-hit draws. The selected controller was frozen by content hashes and assessed on all seven originals, eight shared conditions, both seats and both opponent action seeds (224 games). The own controller is deterministic; own action RNG is unused.

Paired mean received-damage change versus the prior rear candidate on the same 224 cells: **-0.4199 HP**. This is a comparison of complete controllers, not an isolated causal effect of one maneuver parameter.

## Final strict screen

| Opponent | W / D / L | Strict verdict |
|---|---|---|
| s44701_step_002621440 | 0 / 3 / 29 | loss |
| s45800_step_002621440 | 0 / 1 / 31 | loss |
| s46201_step_005242880 | 0 / 1 / 31 | loss |
| s46401_step_005242880 | 0 / 0 / 32 | loss |
| s46500_step_005242880 | 0 / 0 / 32 | loss |
| s47236_step_002621440 | 0 / 3 / 29 | loss |
| s47238_step_010485760 | 0 / 3 / 29 | loss |

Strict opponent passes: **0/7**. Any loss or all draws fails an opponent. Neither development victories nor no-hit draws establish general victory.

## Rear-position evidence and limits

grid001/agent009 won its red consumed cell after receiving 0.7755 HP damage, then held unilateral rear WEZ for 2.55 seconds and inflicted 0.858 rear damage. Its blue counterpart lost. This is an observed rear-position comeback, but does not satisfy the light-damage descriptor (received HP <=0.25). Rear damage and victory are separately reported; successful damage alone is not admission.

Total direct diagnostics: 1324 games, including one explicit manual game and three consumed trace replays. The replay instrumentation was verified after execution to match every original non-wall cell metric. Session exits were observed; the report verifies frozen dependencies, raw cell counts, all official verdicts and content digests. No model replaced and no GitHub upload.
