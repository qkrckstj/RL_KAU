# Less-braking feedback003: regression, parent preserved

This one-parameter trial completed unchanged224screen in51.666549s: **202wins,1draw,21losses;0/7 strict opponent passes**. Parent feedback002 remains the working reference with203wins,4draws,17losses;1/7 passes. The new candidate is preserved but not adopted. No qualified v1, unused448confirmation, admission or publication.

| Opponent | W | D | L | Strict verdict |
|---|---:|---:|---:|---|
| s44701_step_002621440 | 29 | 0 | 3 | loss |
| s45800_step_002621440 | 30 | 0 | 2 | loss |
| s46201_step_005242880 | 26 | 0 | 6 | loss |
| s46401_step_005242880 | 28 | 0 | 4 | loss |
| s46500_step_005242880 | 29 | 1 | 2 | loss |
| s47236_step_002621440 | 30 | 0 | 2 | loss |
| s47238_step_010485760 | 30 | 0 | 2 | loss |

The policy.py bytes are identical to the parent. Only brake_speed482to520 differs in the frozen controller parameters. Model metadata links the exact parent model identity. Both candidates used the exact same224opponent/IC/seat/RNG cells,original39public fields,9actions,and unchanged physics/weapons/verdict. The nominal envelope near500kt motivated the hypothesis;the closed-loop results do not support adopting this setting for the strict objective. This does not establish that the nominal envelope is wrong or that all higher-speed strategies fail.

Paired outcomes versus feedback002:
- draw to draw: 1
- draw to loss: 3
- loss to loss: 15
- loss to win: 2
- win to loss: 3
- win to win: 200

Source and tests were unchanged after62relatedchecks this goal turn. Official loader qualification on9458consumed observations plus duplicate queries matched direct source actions/phases with the actual520archive. These are loading/regression checks,not evidence of improved combat. Native session61507 observedexit0. Raw224cells,completion,digest,and frozen inputs independently recomputed and verified. Learning amount0;no neural imitation or curriculum.

## Goal-turn accounting and next branch

This goal turn added450native executions:two unchanged224screens(feedback00252.165296s,feedback00351.666549s),one exact s46201mutual replay(0.507849s),one failed interactive manual continuation(221.224634s wall including reasoning pauses). Total completed direct diagnostics4021. All raw records, original7 and prior models are preserved. See feedback002/REPORT.md and manual006/REPORT.md.

Next bounded hypothesis should return to feedback002 parameters(brake_speed482 retained) and test an earlier initial near-merge right turn,changing only merge_range240to600m. The remaining failures often lack sustained unilateral fire;an earlier turn may break the second-approach symmetry. This is not implemented or proven,not a claim that all remaining failures have one cause,and not a new evaluation protocol. Use one new output/candidate,unchanged224screen,and conditional own disjoint-unused448 only after all7pass. Select by the strict goal;do not call a high consumed-screen win rate a guarantee or assign a qualified model version prematurely.
