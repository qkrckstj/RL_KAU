# Manual side-attack demonstration, 2026-10-16

The agent completed a winning side attack against preserved opponent s47236_step_002621440 in manual_003. It maintained unilateral weapon eligibility for 2.50 simulation seconds, dealt the opponent's remaining 0.505 HP, and received zero damage in that segment. Final health was 0.505 / 0 at 54.50 seconds. This was a side attack, not a rear attack.

Both manual_002 and manual_003 used the same consumed condition: IC 3401500000, own seat red, opponent action RNG 4900. The second episode replayed 13 previously agent-selected command intervals through 52.10 seconds (1,042 exact native transitions), then the agent inspected observations and issued three explicit left-and-accelerate commands. This is a manually corrected demonstration with a replayed prefix; it is not an autonomous feedback policy or an unseen test.

| Episode | Result | Own / opponent HP | Simulation s | Wall s including decisions | Commands | Max unilateral WEZ s |
|---|---|---|---|---|---|---|
| manual_002 | Loss | 0 / 0.0925 | 108.50 | 752.897665 | 70 | 2.20 |
| manual_003 | Win | 0.505 / 0 | 54.50 | 18.672050 | 16 | 2.50 |

The winning change was earlier turn reversal at side-attack entry, following the changing gun-lead direction instead of continuing the previous turn. In manual_002 the lock broke with opponent HP 0.0925, and subsequent reapproaches lost the health advantage. That complete loss is preserved. manual_003 kept eligibility until the opponent reached zero.

Early damage remains substantial: 0.495 HP was received in the initial mutual exchange. Zero damage applies only to the later unilateral side-attack segment, not the whole game. No rear-only requirement is imposed: the user accepts any position where the agent can hit and the opponent cannot.

This phase added two native games, zero neural training decisions, and zero standard screen games. Total completed direct diagnostics are now 3,119. Existing strict screening remains 224 games (seven opponents, eight shared initial conditions, both seats, two action RNGs). Any loss or all draws fails an opponent. All-seven passage still requires subsequent disjoint unused 448-game confirmation; neither gate has been achieved by this demonstration. Existing seven models and all prior artifacts remain preserved. No admission or GitHub publication occurred.

The requested video is a post-hoc rendering of the exact manual_003 trace, with no additional simulation. Green identifies the agent and pink the opponent. Weapon eligibility and tracking labels use recorded referee fields; cone outlines are only horizontal projections. See manual_003/video/manifest.json for hashes, frame counts and playback timing. Full replay is normal speed; the final 47.00-54.50 seconds are also shown at half speed. Each video holds the terminal frame for two extra seconds.

Evidence: MANUAL_002_003_AUDIT.json, manual_002/{commands,trace}.jsonl and completion.json, manual_003/{commands,trace}.jsonl and completion.json. Native sessions 55716 and 34276 were observed exiting zero.
