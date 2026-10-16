# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3791|598.07|224|37.72|0/7|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|27|3|loss|
|s45800_step_002621440|0|28|4|loss|
|s46201_step_005242880|3|24|5|loss|
|s46401_step_005242880|2|27|3|loss|
|s46500_step_005242880|3|25|4|loss|
|s47236_step_002621440|3|25|4|loss|
|s47238_step_010485760|3|24|5|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3774|598.12|224|37.63|1/6|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|0|30|2|loss|
|s45800_step_002621440|1|31|0|win|
|s46201_step_005242880|1|27|4|loss|
|s46401_step_005242880|1|28|3|loss|
|s46500_step_005242880|2|28|2|loss|
|s47236_step_002621440|2|28|2|loss|
|s47238_step_010485760|3|27|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Candidate retained as continuation parent.

