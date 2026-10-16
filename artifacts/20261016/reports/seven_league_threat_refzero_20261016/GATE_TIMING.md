# Consumed trajectory gate timing

This is an offline analysis of224 already consumed original69000 native replay tapes. No new games or learning. All source panel/checkpoint hashes,unique match joins,raw trace hashes and first-health-decline times are authenticated.

The public opponent-WEZ condition was first observed0.95seconds before the first received-health decline in all224games. Health at that first condition was1.0 in every game. No game had one full second of pre-hit gate activity. This condition changes the additive teacher bias only;the residual policy can still choose other actions outside it.

|Decision group|Decisions|Gate active|Fraction|
|---|---:|---:|---:|
|All|287068|22273|0.077588|
|Before first hit|104540|4256|0.040712|
|At/after first hit|182528|18017|0.098708|

Terminal states are excluded because no next action follows. Lead time uses the public pre-action condition and first observed health decline;it is not an internal physical substep timing measurement. These are69000 trajectories,not current74000 trajectories or the training distribution. They do not establish why learning fails or which intervention will work. No next intervention is selected on this evidence alone.

# Early zero-reference policy probe

At655360new learning decisions,on the same7306fixed consumed states(511gated),reference-to-policy mean KL in gated states changes0.065429to0.070828;teacher action probability0.389616to0.382248. Three gated greedy choices change;82over allstates. These descriptive changes are not tactical or performance evidence. Different source parents/training randomness prevent treating earlier positive-reference studies as a paired causal control.
