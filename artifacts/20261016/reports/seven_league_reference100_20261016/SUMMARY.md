# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3860|602.23|224|36.69|3/4|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|28|3|loss|
|s45800_step_002621440|2|30|0|win|
|s46201_step_005242880|2|30|0|win|
|s46401_step_005242880|2|27|3|loss|
|s46500_step_005242880|2|29|1|loss|
|s47236_step_002621440|3|29|0|win|
|s47238_step_010485760|2|29|1|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Candidate retained as continuation parent.

|1|5242880|3861|608.50|224|36.09|0/7|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|28|3|loss|
|s45800_step_002621440|1|29|2|loss|
|s46201_step_005242880|2|29|1|loss|
|s46401_step_005242880|3|23|6|loss|
|s46500_step_005242880|1|29|2|loss|
|s47236_step_002621440|1|29|2|loss|
|s47238_step_010485760|1|29|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

