# First early-range candidate: s75500

Learning: 5242880 decisions, 3683 completed episodes, 595.981451s. Independent evaluation:224 games,42.658219s.

Each opponent has8 shared initial conditions,both seats,and2 action RNG replicas. Any loss or all draws fails. These consumed panels are adaptive development evidence.

|Opponent|Wins|Draws|Losses|Passed|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|29|2|No|
|s45800_step_002621440|2|23|7|No|
|s46201_step_005242880|4|23|5|No|
|s46401_step_005242880|2|27|3|No|
|s46500_step_005242880|6|25|1|No|
|s47236_step_002621440|4|23|5|No|
|s47238_step_010485760|4|24|4|No|

Total23W174D27L,0/7;rear damage0,received-then-later-rear0. Changed zero-step control25W175D24L,0/7;unchanged source18W189D17L,0/7. Candidate is worse than both and the separately preserved global comparator61500(3/7). No confirmation/admission. Branch control retained;candidate archived.

Training outcomes402W2937D344L are not independent evaluation results. All3683 episodes first dealt/received damage share an observed decision tick;this does not prove within-step physical simultaneity. Early condition active75.196% of completed training ticks and74.0715% of evaluation ticks;activation is not useful positioning. Actor and critic each receive40960Adam updates and320 attempted epoch increments;optimization movement is not performance.

Controller session64149 continues second seed75501 under observed trainer1063366. Closed campaign remains90 candidates;first active completed candidate accounted separately. No GitHub upload.
