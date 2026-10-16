# Signed-gate feedback001: fixed224screen

The signed-only gate revision completed the unchanged224screen in 50.121004s: **189wins,4draws,31losses;0/7 opponent passes**. Every opponent still has a loss. No unused448confirmation, qualified v1, admission or publication occurred. This is automated controller evaluation, not interactive agent piloting or neural training.

| Opponent | Wins | Draws | Losses | Opponent verdict |
|---|---:|---:|---:|---|
| s44701_step_002621440 | 25 | 0 | 7 | loss |
| s45800_step_002621440 | 27 | 0 | 5 | loss |
| s46201_step_005242880 | 21 | 1 | 10 | loss |
| s46401_step_005242880 | 28 | 1 | 3 | loss |
| s46500_step_005242880 | 28 | 1 | 3 | loss |
| s47236_step_002621440 | 30 | 0 | 2 | loss |
| s47238_step_010485760 | 30 | 1 | 1 | loss |

## Paired change

Both candidates used the exact same224opponent/IC/seat/action-RNG combinations and unchanged original physics/weapons/health verdict. Only abs(lead_error)<=12 became signed lead_error<=12 while still requiring unilateral WEZ. All160previous wins remained wins.26previous losses and3draws became wins;one loss became a draw;31losses and3draws remained. The full paired table is in AUDIT.json. Mean received HP fell from0.63330357to0.56630357 across these same224games. This total-damage metric does not establish an improvement specifically in the initial exchange. Neither consumed-screen win rate is a universal guarantee.

## Remaining failure localized

The sole remaining loss against s47238 is red,IC3401500004,action4900. One unchanged replay reproduced reference terminal HP0/0.406 at77.95s exactly. It entered attack58.90s,maintained unilateral WEZ1.20s,earned HP0.4885/0.406 by60.10s,and then remained in egress_turn until death. It never reached egress_hold.

The hold gate jointly requires range>=1100m,own target behind(absolute bearing>=120deg),and body-heading opposition>=150deg. Those conditions did not coincide. At60.10s headings opposed177deg,but range925m and target ahead. At63.95s range1054m,target behind150deg,but heading opposition had fallen to97deg. At65.95s range1813m,target behind120deg,but heading opposition only53deg. Constant right turning then produced another vulnerable approach and lost the earned lead. The trace disproves a suggestion that terminal heading hold had already been entered or that a source/runtime error caused the loss.

Next bounded candidate should test a positional egress criterion based on enemy bearing-to-own and separation opening,rather than body-heading opposition alone. This is a hypothesis for one future candidate and unchanged224screen,not an implemented fix or a claim that the remaining31losses share one cause. Preserve feedback001 as current working reference;do not assign it a qualified model version.

Learning0;related checks54;source/package loader parity6755consumed states plus duplicate calls. Native1758 and diagnostic18499 were observed exit0. This candidate224+1native executions,screen50.121004s/trace0.656983s. Entire goal turn added450native executions(two unchanged224screens and two exact diagnostic replays),zero neural training. Completed direct total3571. All prior artifacts and original7 are preserved.
