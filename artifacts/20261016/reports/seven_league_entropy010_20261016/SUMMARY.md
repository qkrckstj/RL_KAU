# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3727|550.54|224|37.13|0/7|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|29|2|loss|
|s45800_step_002621440|2|28|2|loss|
|s46201_step_005242880|4|22|6|loss|
|s46401_step_005242880|2|27|3|loss|
|s46500_step_005242880|4|25|3|loss|
|s47236_step_002621440|5|24|3|loss|
|s47238_step_010485760|5|22|5|loss|

Decision: Full roster preserved; half-uniform plus observed draw+4*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3745|547.78|224|36.93|1/6|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|3|28|1|loss|
|s45800_step_002621440|0|29|3|loss|
|s46201_step_005242880|1|31|0|win|
|s46401_step_005242880|1|29|2|loss|
|s46500_step_005242880|3|27|2|loss|
|s47236_step_002621440|4|24|4|loss|
|s47238_step_010485760|4|27|1|loss|

Decision: Full roster preserved; half-uniform plus observed draw+4*loss+1 allocation. Candidate retained as continuation parent.

