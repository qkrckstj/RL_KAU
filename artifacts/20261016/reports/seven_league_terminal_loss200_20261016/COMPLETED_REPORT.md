# Terminal loss-aversion completed batch

Controller session72334 exited0;no live v13 controller/trainer. All old and new models remain preserved.

|Candidate|Training decisions|Episodes|Learning seconds|Screen W/D/L|Opponents passed|Screen seconds|
|---|---:|---:|---:|---|---:|---:|
|s75000_step_005242880|5242880|3686|577.751307|28/174/22|0/7|37.533781|
|s75001_step_005242880|5242880|3703|578.978920|21/181/22|0/7|38.485737|

Unchanged source74500:18W189D17L,0/7. Both candidates have22losses and rank below the source. Keep actual74500;global61500(3/7) remains separate. Neither candidate shows rear damage,received-then-rear or a rear streak. Increased terminal loss penalty did not reduce evaluation losses in these two runs;this does not establish that all risk-sensitive objectives fail.

Total learning10485760decisions/7389episodes/1156.730227s. Separate evaluation672games includes224exact baseline and448candidate screens. No confirmation or admission,pool7. Native smoke196608decisions/124episodes remains separate.

Per-opponent results:

Candidate75000:

|Opponent|Wins|Draws|Losses|Pass|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|27|3|No|
|s45800_step_002621440|3|26|3|No|
|s46201_step_005242880|4|24|4|No|
|s46401_step_005242880|4|23|5|No|
|s46500_step_005242880|4|27|1|No|
|s47236_step_002621440|5|24|3|No|
|s47238_step_010485760|6|23|3|No|

Candidate75001:

|Opponent|Wins|Draws|Losses|Pass|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|29|2|No|
|s45800_step_002621440|2|25|5|No|
|s46201_step_005242880|5|24|3|No|
|s46401_step_005242880|4|23|5|No|
|s46500_step_005242880|3|28|1|No|
|s47236_step_002621440|3|26|3|No|
|s47238_step_010485760|3|26|3|No|

Every actual production training loss receives additional-2;357and355losses,total-1424. Every Monitor episode return matches the independently accumulated pre-loss return plus adjustment. Training outcomes are distinct from screen performance. Each model applied40960Adamupdates per actor/critic parameter and320epoch increments;optimization movement is not tactical improvement.

Closedcampaign90candidates/448266240decisions/325579episodes. Productionevaluation30016 plus3360separateassessments and224interruptedbaseline yields33600standard evaluations. Counted exactly once. No live native jobs,confirmation,admission orpublication. Consumed adaptive screens are not unseen evidence or universal victory guarantees.
