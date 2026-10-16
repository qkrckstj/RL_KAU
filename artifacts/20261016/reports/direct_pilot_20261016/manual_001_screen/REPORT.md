# Manual control 001 and fixed 224-game screen

The agent selected 42 commands through the native stepping bridge. The bridge repeated each requested action from the existing nine-action space. It did not select maneuvers. All 2,401 executed ticks and their public observations are retained. Session 82614 exited 0.

Manual result: official health win, own 0.9835000 versus opponent 0.9340000. First received damage occurred at 36.9 seconds. Maximum unilateral WEZ duration was 0.15 seconds; no rear attack occurred. The agent gained a small advantage during the first exchange, then disengaged.

## One frozen candidate

Only the 42 step/action pairs were copied into a fixed public-clock replay. This is not a feedback policy. It receives no opponent identity, condition seed, or outcome. The official loader reproduces all 2,401 manual actions, and the matching native screen cell reproduces terminal health, step count, and duration exactly.

The original protocol remains seven opponents, eight shared initial conditions, both seats, and opponent action seeds 4900/4901: 224 games. Any loss or all draws fails an opponent. The own controller is deterministic. No variable-size development grid or additional native qualification replay was run.

Screen session 7033 exited 0 in 68.761449 seconds: **7 wins, 30 draws, 187 losses; 0/7 opponent passes**. The eight shared screen conditions were previously consumed. This is adaptive evaluation, not unseen confirmation.

| Opponent | Wins / Draws / Losses | Verdict |
|---|---|---|
| s44701_step_002621440 | 1 / 5 / 26 | loss |
| s45800_step_002621440 | 3 / 3 / 26 | loss |
| s46201_step_005242880 | 0 / 5 / 27 | loss |
| s46401_step_005242880 | 2 / 4 / 26 | loss |
| s46500_step_005242880 | 1 / 3 / 28 | loss |
| s47236_step_002621440 | 0 / 5 / 27 | loss |
| s47238_step_010485760 | 0 / 5 / 27 | loss |

The fixed schedule reproduced the demonstrated win, but its victories did not extend to strict passes over all seven opponents. Next, pilot a failed condition and identify which observations should trigger maneuver changes. No qualified teacher, imitation learning, fresh confirmation, admission, or GitHub publication is claimed. All original physics, rules, and models are preserved. The pending 784-game comparison remains unlaunched.

This phase added one manual game and 224 evaluation games: 225 native executions, zero neural training decisions. Manual wall time was 339.946499 seconds, including agent reasoning and tool pauses; it is not a simulation throughput measurement. Eight offline replay tests passed; they do not represent eight additional games.
