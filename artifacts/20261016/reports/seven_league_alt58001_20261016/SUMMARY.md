# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3817|624.11|224|36.75|0/7|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|29|1|loss|
|s45800_step_002621440|2|29|1|loss|
|s46201_step_005242880|5|25|2|loss|
|s46401_step_005242880|2|25|5|loss|
|s46500_step_005242880|2|27|3|loss|
|s47236_step_002621440|2|27|3|loss|
|s47238_step_010485760|1|28|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3814|627.04|224|36.13|1/6|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|3|29|0|win|
|s45800_step_002621440|2|29|1|loss|
|s46201_step_005242880|4|27|1|loss|
|s46401_step_005242880|2|28|2|loss|
|s46500_step_005242880|3|27|2|loss|
|s47236_step_002621440|1|28|3|loss|
|s47238_step_010485760|0|30|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Candidate retained as continuation parent.

