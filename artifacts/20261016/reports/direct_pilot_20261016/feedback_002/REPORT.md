# Positional-egress feedback002: fixed224screen

The single positional-egress revision completed unchanged224screen in52.165296s: **203wins,4draws,17losses;1/7 strict opponent passes**. The passed opponent is s47238(31W1D0L). All189previous wins were preserved;14losses became wins,17losses and4draws remained. No unused448confirmation,v1,admission or publication occurred.

| Opponent | W | D | L | Strict verdict |
|---|---:|---:|---:|---|
| s44701_step_002621440 | 30 | 0 | 2 | loss |
| s45800_step_002621440 | 28 | 0 | 4 | loss |
| s46201_step_005242880 | 26 | 1 | 5 | loss |
| s46401_step_005242880 | 28 | 1 | 3 | loss |
| s46500_step_005242880 | 29 | 1 | 2 | loss |
| s47236_step_002621440 | 31 | 0 | 1 | loss |
| s47238_step_010485760 | 31 | 1 | 0 | win |

The source only changes egress-hold entry:enemy bearing-to-own>=90deg,horizontal opening separation,and recorded foeWEZ0,while retaining earned rounded4healthlead,range>=1100m,and own target behind>=120deg. Old unused body-heading parameter was removed. Hold is terminal once entered;these conditions are not continuously rechecked. The full candidate,source,parameters,old models and raw results remain preserved. Automated controller evaluation is distinct from interactive manual piloting and neural learning.

Verification:62relatedchecks;8314source/package action+phase observations and8314duplicatequeries exact;all frozen inputs and224cell digest/strict score verified. Learning0. Screen session88589 actualexit0. Exact consumed s46201redseed3401500000/action4900 replay2369exit0 reproduced a mutual-zero draw at57.20s. It stayed in right_intercept and never found unilateral eligibility;the second mutual lock depleted both HP0.472. This remaining mutual-attack problem differs from the earlier egress failure.

A separate manual006 continuation replayed the first828exact transitions(41.4s),then tested extra braking and turn reversal;it failed0/0.472 at74.3s. That diagnostic is fully preserved and is not adopted. See ../manual_006/REPORT.md.

This goal turn currently added226native executions:224screen+one exact diagnostic+one manual continuation. New neural learning0,total direct3797. A future candidate will test only less post-merge braking (brake_speed482to520) to retain a speed target near the nominal500kt turn regime. This is an untested controller-setting hypothesis,not a physics change or established fix. Original fixed224and conditional disjoint-unused448gates remain;finite consumed-screen success never implies victory in every situation.
