# Manual-derived feedback000: fixed224screen

Automated public-observation controller, not interactive piloting or neural training. The unchanged224screen completed in 47.991280s: **160 wins,6 draws,58 losses;0/7 opponent passes**. Every opponent had a loss. No confirmation, qualified v1, admission or publication occurred.

| Opponent | Wins | Draws | Losses | Opponent verdict |
|---|---:|---:|---:|---|
| s44701_step_002621440 | 24 | 0 | 8 | loss |
| s45800_step_002621440 | 22 | 0 | 10 | loss |
| s46201_step_005242880 | 21 | 1 | 10 | loss |
| s46401_step_005242880 | 25 | 1 | 6 | loss |
| s46500_step_005242880 | 24 | 0 | 8 | loss |
| s47236_step_002621440 | 22 | 1 | 9 | loss |
| s47238_step_010485760 | 22 | 3 | 7 | loss |

Candidate source/standalone parameters are preserved in candidate/. It uses original39public channels and original9actions, event-driven maneuvers, original simulator/weapons/initial-condition generation/verdict and preserved original7. Elapsed clock detects reset only. Learning amount0;related checks51;source/package parity6755consumed states plus duplicate calls. These offline checks are not closed-loop performance evidence.

## Localized failure

One exact consumed replay against s47236,red,IC3401500002,action4901 reproduced terminal HP0/0.6535 at106.5s exactly. Initial hypothesis that the close-merge threshold had not fired was rejected: the controller entered right_brake36.60s,right_hold39.85s,right_intercept43.95s. It then stayed in right_intercept until death.

The actual missed transition is the absolute-value gun-lead gate. At53.75s it first entered unilateral WEZ at range1498.33m, but signed lead error was already-17.304degrees. The absolute<=12degree requirement rejected that valid late left-reversal opportunity. Twelve unilateral ticks passed without entering attack. At52.80s lead was near zero but range was still too large; absolute-lead alignment and weapon eligibility occurred at different times. This diagnostic added one native execution, not another screen or unseen evidence. Next candidate should change only this left-reversal gate to signed lead<=12degrees while retaining unilateral WEZ, then use unchanged224screen.

Native sessions18093 and98589 were observed exit0. New executions225,learning0,total direct diagnostics3346. Screen wall47.991280s;diagnostic wall0.859186s. All raw cells and diagnostic trace are preserved;frozen hashes verified. Finite consumed-condition success or individual victories are not universal guarantees.
