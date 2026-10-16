# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3811|601.31|224|36.57|2/5|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|30|1|loss|
|s45800_step_002621440|1|30|1|loss|
|s46201_step_005242880|2|27|3|loss|
|s46401_step_005242880|0|29|3|loss|
|s46500_step_005242880|4|27|1|loss|
|s47236_step_002621440|1|31|0|win|
|s47238_step_010485760|1|31|0|win|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Candidate retained as continuation parent.

|1|5242880|3782|583.56|224|36.46|2/5|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|30|1|loss|
|s45800_step_002621440|1|31|0|win|
|s46201_step_005242880|3|29|0|win|
|s46401_step_005242880|1|27|4|loss|
|s46500_step_005242880|4|26|2|loss|
|s47236_step_002621440|3|26|3|loss|
|s47238_step_010485760|1|28|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

