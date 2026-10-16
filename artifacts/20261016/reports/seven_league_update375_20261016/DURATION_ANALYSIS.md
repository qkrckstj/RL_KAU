# Completed-screen duration diagnostics

Completed consumed evaluation episodes,not training rollout boundaries or phase. Episode duration and damage-to-terminal delay do not prove rewards lacked credit;critic bootstrap propagates across rollout boundaries. Longer rollout performance is untested;no causal or unseen evidence. Active study unchanged.

|Panel|Games|Median duration(s)|Median first received to terminal(s)|Duration >32s|Post-hit delay >32s|
|---|---:|---:|---:|---:|---:|
|baseline|224|61.500|37.925|224|224|
|iteration_000|224|61.450|37.900|224|224|
