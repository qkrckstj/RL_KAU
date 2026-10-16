> **Latest seven-model Goal:** Draw-only comparison complete:70500=16W/196D/12L(2/7),70501=9W/200D/15L(2/7),bothzero rear damage/streak. Branch69000retained2/7,global61500retained3/7. Win-only bonus experiment active(controller1033097/trainer1033099;verify live),winbonus0->1,drawpenalty.25/gamma.9995/LR9.375e-7unchanged. Native24equivalence executions,163840smoke/106episodes(including11wins),224baseline replay and independent review passed. Completed353,894,400decisions/257,696episodes/27,776evaluation games including current/interrupted baselines;272diagnostic executions separate. No confirmation/admission/upload.

# Local seven-model league — 2026-10-15

The user explicitly requested autonomous local training after compact publication. The122historical plans are experimental inspiration only, not a script to replay. No GitHub upload is authorized.

## Live entry

- Run: `aircombat-rl/runs/seven_league_main_20261015`
- Controller at launch: PID952692; unified execution session56842. Verify process identity and files before any restart.
- Entry: `tools.seven_model_league_v2`; configuration frozen in `config.json`.
- No other learner was live before launch. CPU8processes/32environments; BLAS/OpenMP/MKL threads1. Approximately22GiB host memory available before preparation.
- Initial stage:224-game baseline. Then up to4iterations of1,310,720decisions,655,360checkpoint interval,224-game screens while the pool contains7models. The bounded batch returns to the agent for outcome review; it is not the end of the Goal.

## Protocol

Original seven models remain byte-preserved. Both seats and8shared initial conditions; paired action replicas4900/4901 use the same replica index for learner and opponent. Any loss fails an opponent; all draws also fails; at least one win with no losses passes. Complete all-opponent screen success triggers16fresh disjoint initial conditions (448games initially). Only complete authenticated confirmation success admits the candidate append-only. Confirmation panels are consumed once; old failures are never unseen evidence. Finite-panel passage does not guarantee victory elsewhere.

## Initial experiment and adaptation

Start from s45800 43-input PPO with actor/critic/initializedAdam/counters verified exact. Preserve existing native simulation/controller/terminal-health rule. Disable the historical quarter-position training curriculum so new training uses the original initial-condition sampler. Preserve source rewards and PPO settings initially. All14sampled opponent variants remain in periodic coverage. Weighted draws use half uniform plus half previous measured nonwin frequency. Two consecutive non-improving iterations trigger a bounded half-learning-rate test; all other settings remain fixed. Every proposal, parent, seed, allocation and result is recorded. Future decisions are based on outcomes rather than mechanically copying the122plans.

Training and evaluation are separate phases/processes. The selected continuation parent is chosen on consumed screen results; confirmation is used only for pool admission. Raw rows, episode logs, checkpoints, decision counts and time accounting remain local. Do not use historical same-model comparisons with different conditions as causal evidence.

## Validation and review

v1 smoke completed20,480decisions/56games, then independent review detected RNG reset during parent audit and missing direct-train duplicate guard. v1source/results remain preserved. v2 preserves Python/NumPy/Torch RNG during audit; per-iteration lock and single-use marker reject duplicate training; startup failures are recorded.11tests pass, including real-checkpoint RNG preservation. Independent reviewer approved v2. v2smoke completed20,480decisions/56games; actor and optimizer changed; restored parent state matched. Smoke conditions are disjoint from the main run and cannot qualify candidates.

## Inspect and continue

Read `baseline/completion.json`, `iteration_*/result.json`, `SUMMARY.md`, `PROGRESS.json`, `pool.json`, and actual PID/command. During training read `iteration_*/training_progress.json`. Errors produce `failure.json` or `training_failure.json` and are never blindly retried. A `STOP` marker is honored before the next iteration and during training. Evaluation currently finishes its active panel before stopping.

Frozen inputs must not be edited or bypassed. A changed implementation requires a new version and new output/reservations. The controller intentionally refuses overwriting existing iterations or automatic resume after admission; inspect preserved evidence and prepare an explicit verified continuation. No push, publication or merging is part of this Goal.

## Initial baseline diagnosis

The complete224-game baseline scored33wins/170draws/21losses; strict opponent passage0/7. All170draws had both terminal health values rounding to0; none was a no-damage draw. This is a consumed-panel diagnostic, not a causal explanation. Initial learner metrics inspected through33rollouts had no nonfinite numeric values. See `BASELINE_DRAW_DIAGNOSIS.json` and raw baseline cells.

## First completed iteration and identity audit

Iteration000:1,310,720decisions,932complete training episodes,142.744s learning;224screen games in38.757s,32wins/161draws/31losses,strict0/7. Parent baseline33/170/21remains selected; every candidate retained. Raw episode count932matches accounting;64logged rollouts contain no nonfinite numeric metrics;19checkpoint file hashes pass. Iteration001started with evidence-updated allocation.

Post-evaluation audit found v2candidate identity accidentally included generated Python bytecode. This changes no game outcomes but can falsely reject admission. New v3uses verified READY-listed artifacts plus READY hash, excluding runtime caches; v2frozen source remains preserved and live batch continues. No candidate has yet passed the screen. Before any future admission, use corrected identity verification against original immutable screen and confirmation artifact hashes. No confirmation is reclassified as unseen. New regression test requires cache immunity and rejects changed weights.

## Second iteration completed

Iteration001:1,310,720decisions,921training episodes,137.539s learning;224screen games,42wins/162draws/20losses,strict1/7. s46401passes with8wins/24draws/0losses. Candidate s51501_step_001310720becomes continuation parent on consumed development evidence. No full screen pass, no confirmation and no pool addition. Iteration002is live from this parent; stagnation count reset, so learning rate remains3e-5. This adaptive, different-seed comparison does not isolate the effect of allocation changes.

Audited672complete evaluation cells across baseline and two candidates,2,621,440training decisions and1,853completed training episodes. All metrics finite and checkpoint manifest hashes valid. v3identity fix passed12tests and independent review; use v3for the next freshly prepared run. Current frozen v2source remains intact.

## First batch audit and lower-rate continuation

Four iterations completed5,242,880decisions /3,703training episodes /569.967s learning. All1,120evaluation cells (baseline+4screens) authenticated, no confirmation games. Strict passes by iteration:0,1,0,0; bests51501retained. The original seven remain unchanged and no candidate is admitted. Full raw checks and time ledger are in `BATCH_AUDIT.json`.

Next bounded batch:2iterations froms51501, LR3e-5to1.5e-5aftertwo consecutive regressions. The original source settings/rewards/controller/initial sampler remain unchanged; evidence-guided allocation continues. New RNG/condition streams mean this is an adaptive continuation, not a matched causal estimate of learning rate. Source panels were authenticated and bound to their exact candidate/pool input hashes before parent selection. Source controller terminality/locks were checked. The next baseline replays the consumed screen and is counted; confirmation conditions are newly reserved.

The v3cache-independent identity and continuation guards passed15focused tests plus independent review. Future changes must use a new run/version if frozen sources would change.

Lower-rate baseline completed224games and exactly replayed all non-wall fields of the selected parent screen. v3model identity stayed stable after inference. Total evaluation executions now1,344including this consumed-condition replay; confirmed improvement remains only1/7development passes. Firstlower-rate training is live and parent actor/critic/Adam restoration is exact.


## Lower-rate batch completion and longer-duration hypothesis

Both lower-rate candidates completed: iteration0 32W/162D/30L and0/7 strict passes; iteration1 37W/164D/23L and1/7 (heading-memory opponent). Neither outranked retaineds51501:1/7,42W/162D/20L. All672 baseline/screen cells and exact candidate/pool bindings authenticated; original7unchanged. Batch:2,621,440decisions,1,846training episodes,287.665s learning,114.653s evaluation. Main+lower cumulative:7,864,320decisions,5,549episodes,1,792evaluation games;0confirmation/admission.

Next hypothesis: retain LR1.5e-5 and extend each candidate from1,310,720to5,242,880decisions, two bounded rounds. Keep655,360checkpoint interval and existing rewards/settings. Override automatic additional LRhalving before launch, recorded inEXPERIMENT_DECISION.json. Fresh training band3621500000and confirmation3501520000reserved. Reused screening is consumed adaptive evidence; new baseline replay is counted. Historical draw-utility0.25/0.5studies did not show consistent improvement on their different panels, so reward changes are deferred. This tests a duration hypothesis, not an isolated causal estimate. No native rule or physics changes. Available memory at preparation23.8GB; no other league run live.


Longer-run baseline audit:224/224game records match retained-parent screening exactly except measured wall time; READY-based model identity stable. Baseline red17W/80D/15L; blue25W/82D/5L. These consumed-panel diagnostics suggest tracking seat asymmetry, not claiming a general seat effect. Total completed evaluation executions now2,016. Iteration0trainer PID954567 restored exact policy/optimizer/counters before intervention; current learning is separate from completed baseline evaluation.


## Longer-duration first candidate completed

s52500completed5,242,880decisions /3,724training episodes /553.335s learning, followed by224evaluation games /38.211s. Result38W/170D/16L, strict0/7: every opponent had at least one loss. Fewer game losses than retaineds51501(20)does not override strict opponent-pass priority. Preserve candidate and retains51501(1/7)as continuation parent. All rawcells, input bindings, checkpoint manifest and frozen source hashes verified initeration_000/COMPLETED_AUDIT.json. No confirmation or admission. Second planned long candidate now training, PID955316; same LR1.5e-5, adaptive distribution follows latest consumed screen, original7retained. Cumulative completed production training13,107,200decisions /9,273episodes;2,240evaluation games. Current in-flight steps excluded.

Six earlier candidate screens were aggregated descriptively inPREVIOUS_CANDIDATE_DIAGNOSIS.json: encounter/speed opponents accounted for35/32losses across192games each, versus13..20for the other opponents. Those panels are correlated and consumed. If the duration batch fails, consider a fresh loss-prioritized allocation experiment rather than endless LRhalving; keep uniform coverage and immutable simulation rules.


## Longer batch complete; loss-prioritized continuation

Second long candidate s52501:5,242,880decisions,3,720episodes,559.613s learning;224screen games,39W/164D/21L,0/7strict passes. Full long batch10,485,760decisions,7,444episodes,1,112.948s learning;672evaluation games including baseline,115.195s evaluation. All completed rawcells, candidate/pool bindings and frozen sources authenticated inBATCH_AUDIT.json. No confirmation/admission; retain s51501(1/7). Cumulative completed production18,350,080decisions /12,993episodes /2,464evaluation games. Smoke statistics excluded.

v4loss-weighting smoke completed20,480decisions and56games after the long controller exited. Exact actual plan probabilities, parent state restoration, checkpoint hashes and frozen v4source verified; smoke cannot admit models.22tests passed; independent reviewer found no important calculation/subprocess defect and required explicitv4source freezing/weight4at preparation; both addressed. SeeINTEGRATION_AUDIT.jsonunderseven_league_lossweight_smoke_20261015.

New production batch:2rounds of5,242,880decisions, LR1.5e-5retained, allocation50%uniform+50%normalized(draws+4*losses+1). v4source frozen inconfig.jsonand explicitweight4recorded. Existing rewards, policy interface, native rules and strict evaluation unchanged. Training band3641500000, fresh confirmation3501540000reserved; consumed screen replay counted separately. Different RNGstreams and adaptive sampling prevent isolated causal attribution. No GitHub upload.

Loss-prioritized production baseline224games exactly reproduced all nonwall records; actual training probabilities verified againstweight4formula. Parent restoration exact. Total completed production evaluation2,688games including this replay; candidate0training started.

Allocation audit clarification: current PeriodicCoverageSampler(period2)forces half of episode selections through a uniform coverage permutation; the other half uses the configured50/50uniform+difficultyprobabilities. Therefore effective asymptotic selection is75%uniform+25%difficulty, not50/50across all episodes. Both prior and new runs preserve period2; only loss weighting changed. Snapshot at412completed training episodes contains all14sampled opponent variants in both seats. Seeiteration_000/ALLOCATION_SNAPSHOT_000412.json. Finite completed-episode frequencies have in-flight duration censoring.


## First loss-prioritized candidate completed

s53500:5,242,880decisions,3,688training episodes,548.215s learning;224screen games,38.781s evaluation;39W/159D/26L,strict0/7. All opponents had losses. Retains51501as best continuation parent; candidate archived. Completed rawcells/checkpoint/input bindings/source hashes authenticated and all training episode counts reconciled initeration_000/COMPLETED_AUDIT.json. The first candidate does not show improvement; second planned candidate now training with its own random stream and latest consumed-screen allocation. Cumulative completed production23,592,960decisions /16,681episodes /2,912evaluation games;0fresh confirmations/admissions.


## Loss-prioritized batch complete; received-cost hypothesis

Second candidate s53501:5,242,880decisions /3,784episodes /548.071s learning;224screen games,34W/167D/23L,1/7strict passes (heading-memory). Retaineds51501still ranks higher with1/7and20losses. Whole batch10,485,760decisions /7,472episodes /1,096.286s learning;672evaluation games /114.324s evaluation. All rawcells, bindings, checkpoints and frozen hashes checked inBATCH_AUDIT.json. No confirmation/admission. Cumulative production28,835,840decisions /20,465episodes /3,136evaluation games.

New hypothesis changes only received-damage learning cost0.5to0.75; LR1.5e-5,duration5,242,880,allocationweight4andperiod2retain prior values. Mutual-zero draws produced near-neutral shaping under symmetric0.5/0.5weights. Historical received075results were mixed on different panels and received0875did not beat its parent; this is an evidence-motivated trial, not historical replay or expected success. Avoidance may increase all-draw failures; original strict gate applies.

Validation:4current-wrapper fixture cases passed. First native diagnostic incorrectly assumed nonterminaloutcomeexists and stopped; failure record preserved with fresh conditions used for corrected replay. Corrected native replay4episode executions (2conditions/both seats in pairs) matched observations, transitions and verdict exactly; received cost produced-.25reward difference with nonzero received damage. See seven_league_received075_diagnostic_v2_20261015/completion.json. Short integration20,480decisions /56evaluation games verified config propagation, exact parent restoration and manifests;0completed training episodes in this short smoke, so native replay supplies terminal evidence. Independent review found no blocking issue; production config explicitly checked LRoverride.

Production uses fresh training3661500000and confirmation3501560000ranges. Config freezes v4source, reward test and immutable diagnostic/smoke evidence. Native rules and original7unchanged. No GitHub upload.

Received-cost baseline224games exactly reproduced prior nonwallrecords; exact parent restoration and actual trainingplanreceived=.75verified. Cumulative completed production evaluation3,360games; first reward-cost candidate training PID957840.

Completed lossweight checkpoint parameter audit confirms all actor-feature/action-head/critic/value parameters trainable and changed from retained parent. Actor-feature relativeL2changes2.62%/2.43%; action-head4.83%/5.21%; optimizer update counters8576to9600. This rules out unchanged actor weights in those candidates, not other optimization limitations. Evidence:seven_league_lossweight_20261015/PARAMETER_UPDATE_AUDIT.json.


## First received-cost candidate completed (2026-10-16)

s54500:5,242,880decisions,3,736training episodes,549.070s learning;224screen games,37.241s evaluation;23W/181D/20L,strict0/7. Every opponent had a loss. Compared with retaineds51501(42W/162D/20L,1/7), this candidate reduced wins and increased draws without reducing losses. The adaptive different-seed experiment is not a causal estimate. All rawcells, input bindings, manifests and frozen source hashes authenticated initeration_000/COMPLETED_AUDIT.json. Retain best parent; candidate archived. Second candidate training. Cumulative completed production34,078,720decisions /24,201episodes /3,584evaluation games;0confirmation/admission.


## Received-cost batch complete; entropy experiment

Second received-cost candidate s54501:5,242,880decisions /3,703episodes /550.530s learning;224screen games,27W/174D/23L,0/7strict passes. Whole batch10,485,760decisions /7,439episodes /1,099.601s learning;672evaluation games /113.587s evaluation. Frozen files, checkpoints, candidate/pool bindings and rawcells authenticated inBATCH_AUDIT.json. Both candidates failed to improve retaineds51501. No confirmation/admission.

Completed campaign12candidates:39,321,600decisions /27,904episodes /4,166.467s learning;3,808evaluation games /649.273s evaluation, including1,120baseline assessments. Excludes active entropy batch, smokes and diagnostics. Per-candidate settings/times/scores and opponent details consolidated inruns/SEVEN_LEAGUE_CAMPAIGN_20261016.mdand adjacentJSONwith source result digests.

v5changes only PPOentropy-loss coefficient after exact parent restoration.32tests passed; independent review confirmed ordering, serialization and subprocess correctness. Smoke20,480decisions/56evaluation games verified recorded.003to.01intervention, exact parent state, checkpoint hashes and reloaded modelent_coef=.01/priorbias6. Seeentropy010_smoke/INTEGRATION_AUDIT.json. Production config explicitly freezesv5source and entropy tests, holdsLR1.5e-5,receivedcost.75,allocation4,duration5,242,880. Fresh training3711500000and confirmation3501580000reserved. Low observed entropy motivates a hypothesis, not a cause established by evidence. No rule/physics changes or GitHub upload.

Entropy production baseline224games exactly reproduced prior nonwallrecords; parent restoration and recorded.003to.01intervention verified. Cumulative completed evaluation4,032games including new baseline. Candidate0training PID959368.


## First entropy candidate completed

s55500:5,242,880decisions /3,727training episodes /550.537s learning;224screen games /37.131s evaluation;23W/177D/24L,strict0/7. Mean rollout entropy.23831versus prior received-cost candidate.16899on different training streams: descriptively greater diversity, no performance gain or isolated causal inference. Every opponent had losses. Rawcells, exact candidate/pool bindings, checkpoint manifests and frozen sources verified initeration_000/COMPLETED_AUDIT.json. Retains51501best parent(1/7), preserves all candidates. Second entropy candidate training. Cumulative completed production44,564,480decisions /31,631episodes /4,256evaluation games. No confirmation/admission/upload.


## Second entropy candidate improves strict ranking

s55501:5,242,880decisions /3,745training episodes /547.782s learning;224screen games;16W/195D/13L,1/7strict passes (state-prior opponent1W/31D/0L). New rank(1,-13,16)exceeds prior s51501(1,-20,42). Wins decreased and draws increased; only the predefined strict-pass/loss ordering establishes this parent choice, not universal superiority. Other6opponents still fail, so no confirmation or admission. Full rawcells, source hashes, checkpoints and input bindings authenticated inBATCH_AUDIT.json.

Whole entropy batch10,485,760decisions /7,472episodes /1,098.319s learning;672evaluation games /111.898s evaluation. Completed campaign14candidates totals49,807,360decisions /35,376episodes /4,480evaluation games. Consolidated campaign report updated.

Continue new s55501parent at unchanged settings for two bounded5,242,880decision rounds. LR1.5e-5,entropy.01,receivedcost.75,allocation4andperiod2retained. Fresh training3731500000and confirmation3501590000reserved. Previous best and original7remain preserved. Original-parent family reassessment was conditional on no improvement; deferred now that strict ranking improved. Static compatibility audit remains recorded (different observation/history/export contracts must not be silently coerced). New baseline reuses consumed screen and will be counted. No GitHub upload.

New-best baseline224game replay matched every nonwallfield froms55501screen; exactpolicy/optimizer/counterrestoration verified. Entropy.01remains unchanged. Cumulative evaluation4,704games; first continuation candidate running PID960568.

Paired consumed-screen comparison of oldbest s51501andnewbest s55501: old20losses became19draws+1win, but10old draws and3old wins became13newlosses. Also36winsbecamedraws. Thus reduced aggregate loss count is not pointwise dominance. Newloss end-health margins range.01to.076, well above round4resolution. Full paired transition matrix inentropy_continue/BEST_PARENT_PAIRED_COMPARISON.json; no unseen inference.


## First new-parent continuation completed

s56000:5,242,880decisions /3,759episodes /550.737s learning;224screen games /36.384s evaluation;17W/184D/23L,strict0/7. Every opponent had a loss; retain s55501(1/7,13losses)and archive this candidate. Rawcells, input bindings, checkpoint manifests and frozen sources authenticated initeration_000/COMPLETED_AUDIT.json. Second planned continuation candidate training from retaineds55501. Cumulative completed production55,050,240decisions /39,135episodes /4,928evaluation games;0confirmation/admission/upload.


## New best: three strict opponent passes

s56001 completed5,242,880decisions /3,780episodes /547.951s learning;224screen games /37.439s evaluation;16W/198D/10L,3/7strict passes. Per opponent W/D/L: s44701=1/26/5; s45800=3/29/0; s46201=3/29/0; s46401=3/26/3; s46500=1/31/0; s47236=2/29/1; s47238=3/28/1. Passes are s45800,s46201,s46500. Others fail, so no confirmation/admission. Rawcells, exact input bindings, checkpoint manifests, frozen sources and original pool authenticated in entropy_continue/BATCH_AUDIT.json. Controller session19905 exited0.

Batch totals10,485,760decisions /7,539episodes /1,098.688s learning;672evaluation games /110.559s evaluation. Cumulative16completed candidates:60,293,120decisions /42,915episodes /5,152evaluation games. Reused development conditions are not unseen evidence.

Continue new best s56001 with unchanged settings for two bounded rounds in seven_league_threepass_continue_20261016, PID961780 at launch/session10793. Baseline replay started, then serial training/evaluation. Original-family assessment remains unlaunched because improvement occurred; its review and live-lock guard verification are preserved. No GitHub upload.

New-best224game baseline replay exactly matches all nonwall cell fields; parent restoration exact and frozen hashes verified. Cumulative completed evaluations5,376games including this replay. Iteration000training PID961825.


Paired comparison s55501 to s56001 on the same224consumed cells: previous13losses became11draws+2wins; new10losses came8previousdraws+2previouswins. Fourteenpreviouswins became draws and14previousdraws became wins. Newlosses are7blue/3red, health margins.0100to.0265. Improved strict ranking is not pointwise dominance; seat counts are descriptive small-panel evidence, not sufficient to change IC/seat weighting. Stored in threepass_continue/BEST_PARENT_PAIRED_COMPARISON.json. No new simulations; training remains separate from evaluation.


## First three-pass-parent continuation completed

s56500:5,242,880decisions /3,706episodes /556.047s learning;224screen games /40.945s evaluation;30W/154D/40L,strict0/7. Per-opponent W/D/L:44701=6/17/9;45800=4/23/5;46201=4/25/3;46401=5/21/6;46500=8/22/2;47236=1/22/9;47238=2/24/6. Every opponent had losses, so retain s56001(3/7,16W/198D/10L). Rawcells/input binding/checkpoint manifests/frozen hashes verified in iteration_000/COMPLETED_AUDIT.json. Second planned repeat training PID962482. Cumulative65,536,000completed decisions /46,621episodes /5,600evaluation games including baseline replays; no confirmation/admission/upload.


## Three-pass-parent batch completed; original-family assessment launched

s56501:5,242,880decisions /3,696episodes /560.285s learning;224screen games /43.355s evaluation;40W/136D/48L,0/7strict passes. Per-opponent W/D/L:44701=6/20/6;45800=10/13/9;46201=4/25/3;46401=6/18/8;46500=4/22/6;47236=4/20/8;47238=6/18/8. No improvement; retain s56001. Both candidate rawresults, checkpoint manifests, exact panel bindings, frozen sources and original pool authenticated in BATCH_AUDIT.json. Session10793exit0. Batch10,485,760decisions /7,402episodes /1,116.332s learning;672evaluation games /121.589s evaluation. Cumulative18candidates:70,778,880decisions /50,317episodes /5,824evaluation games before assessment.

Both continuations stalled, so launched previously reviewed source-adjusted original assessment, PID963321/session6251. Source lock, terminal batch, frozen hashes and no-live-trainer guards passed. Compare each original against7originals including self,224games each,1,568new executions on consumed development conditions. This is model-family direction finding, not unseen confirmation or admission. No training overlaps. No upload.


## Original-seven comparison completed

Session6251exit0;7models/1,568games/266.763s evaluation. All rawcells and exact model/pool bindings, frozen inputs and original artifacts authenticated in COMPLETED_AUDIT.json. Ranking summary:heading-memory1pass/14losses;residual1/21;predictive-aim0/21;state-prior0/30;encounter-speed0/36;rate-input0/37;encounter-isolated0/40. Current s56001remains3/7with10losses; no confirmation/admission. These are consumed-panel measurements including original self matches, not fresh tests. Full results in original assessment SUMMARY.md and completion.json. Cumulative7,392evaluation executions;70,778,880training decisions/50,317episodes unchanged.

Launched two-round lower-LR continuation froms56001,7.5e-6instead of1.5e-5; all other settings retained. New training and confirmation bands reserved. Controller PID963737/session15052; baseline replay then serial learning/evaluation. No GitHub upload.

Lower-LR baseline224game replay exact in all nonwallfields, parent restoration exact, frozen hashes verified. Cumulative7,616evaluation games. Trainer963812confirmed live.


Remaining-opponent comparison from authenticated original assessment: none of7originals strictly passes any of s56001remaining4failed opponents. Against44701,best original45800has2losses vs current5;against46401,heading-memory has2vs current3. Against47236/47238,current has1loss each versus minimum3among originals. Thus no original directly supplies a demonstrated no-loss solution to the remaining opponents; aggregate/current-parent retention is supported on this consumed panel, without ruling out future family training. Full table in original assessment REMAINING_OPPONENT_COMPARISON.json. No new simulations.


## First lower-LR continuation completed

s57000:5,242,880decisions /3,799episodes /563.561s learning;224screen games /43.437s evaluation;40W/137D/47L,0/7strict passes. Per-opponent W/D/L:44701=9/17/6;45800=4/20/8;46201=3/19/10;46401=2/23/7;46500=8/23/1;47236=7/17/8;47238=7/18/7. Lower learning rate did not prevent regression in this candidate; await second planned replicate before next change. Retain s56001(3/7). Rawcells, exact bindings, frozen hashes and checkpoint verified in iteration_000/COMPLETED_AUDIT.json. Second trainer PID964583. Cumulative76,021,760decisions /54,116episodes /7,840evaluation games including original assessment;no confirmation/admission/upload.


Completed-rollout diagnostic: s55501mean action entropy.2312, s56001.3154; failed s56500/s56501.6455/.6240; failed lower-LR s57000.5162. Mean training rewards respectively-.2353/-.2586/-.3085/-.3170/-.3047. Different on-policy streams, so no causal estimate; metrics CSV hashes and full statistics retained in lowerlr/COMPLETED_METRICS_COMPARISON.json. If second lower-LR candidate also fails to improve, a bounded entropy-coefficient reduction from retaineds56001is a justified next hypothesis, instead of repeated LR halving. Current live run unchanged.


## Lower-LR batch terminal; reduced-entropy continuation

s57001:5,242,880decisions /3,726episodes /556.656s learning;224screen games /43.182s evaluation;41W/136D/47L,0/7strict passes. Opponent W/D/L:44701=8/16/8;45800=5/17/10;46201=9/19/4;46401=7/21/4;46500=3/22/7;47236=6/19/7;47238=3/22/7. Both candidates regress, retain s56001. Rawcells/input bindings/checkpoint/frozen hashes/pool authenticated in BATCH_AUDIT.json. Session15052exit0. Batch10,485,760decisions /7,525episodes /1,120.216s learning;672evaluation games /123.971s evaluation. Cumulative81,264,640decisions /57,842episodes /8,064evaluation games including original assessment;0confirmation/admission.

Conditional entropy experiment launched: coefficient.01to.003only, keep LR7.5e-6 (explicitly override helper automaticfurtherhalving), preserve reward/allocation/physics/IC/verdict. Two rounds5,242,880decisions;training3791500000andconfirmation3501620000reserved. Tested v5entropy override after exact parentrestore; no source edits. No upload.

Reduced-entropy baseline224game replay exact in all nonwallfields; parent restoration exact and entropy intervention recorded. Frozen hashes verified. Cumulative8,288evaluation games; trainer965448confirmed live.


## First reduced-entropy candidate completed

s57500:5,242,880decisions /3,851episodes /558.808s learning;224screen games /37.361s evaluation;26W/173D/25L,strict0/7. Opponent W/D/L:44701=1/27/4;45800=3/24/5;46201=3/27/2;46401=2/25/5;46500=5/23/4;47236=5/25/2;47238=7/22/3. Fewer losses than the two prior47-loss lower-LR candidates, but different training streams and still inferior to retaineds56001(3/7,10losses); no causal claim or admission. All rawcells/input bindings/checkpoint/frozen hashes authenticated in iteration_000/COMPLETED_AUDIT.json. Second planned replicate training PID966060. Cumulative86,507,520decisions /61,693episodes /8,512evaluation games;0confirmation/admission/upload.

Completed entropy diagnostic: prior lower-LR candidates mean rollout entropy.5162/.5400, final-quarter.7068/.6944; first reduced-entropy candidate mean.3751, final-quarter.3948. Its meanapproxKL6.795e-5versus prior1.018e-4/1.121e-4. Consistent with reduced stochasticity/update size in these observed rollouts, but distinct training streams prevent causal attribution; still0/7strict passes. CSV hashes and full statistics in entropy003/COMPLETED_ENTROPY_DIAGNOSTIC.json. Current second replicate unchanged.


## Reduced-entropy batch terminal

s57501:5,242,880decisions /3,861episodes /561.911s learning;224screen games /35.965s evaluation;12W/197D/15L,0/7strict passes. Opponent W/D/L:44701=1/30/1;45800=1/28/3;46201=2/27/3;46401=1/26/5;46500=1/30/1;47236=2/29/1;47238=4/27/1. Every opponent still has a loss, retain s56001. Full rawcell/model/pool binding/checkpoint/frozen-source checks passed in BATCH_AUDIT.json. Session80062exit0. Batch10,485,760decisions /7,712episodes /1,120.720s learning;672evaluation games /110.524s evaluation. Cumulative91,750,400decisions /65,554episodes /8,736evaluation games including original assessment;0confirmation/admission.

Next two-repeat test sets entropy bonus0only, retains LR7.5e-6and all other settings from current experiment, starts from preserved bests56001. Evaluation still uses paired action sampling. Fresh training3811500000andconfirmation3501630000reserved; no code changes or upload.

Zero-entropy baseline224game replay exact in all nonwallfields, parentrestore exact, entropy.01to0applied afterrestore; frozen hashes verified. Cumulative8,960evaluation games; trainer966929confirmed live.


## First zero-entropy-bonus candidate completed

s58000:5,242,880decisions /3,886episodes /558.587s learning;224screen games /35.816s evaluation;14W/195D/15L,1/7strict passes (heading-memory2W/30D/0L). Other opponent W/D/L:44701=1/29/2;45800=3/26/3;46201=0/31/1;46401=4/27/1;47236=2/26/4;47238=2/26/4. Retain s56001(3/7,10losses). Rawcells/model/pool bindings/checkpoint/frozen hashes verified in iteration_000/COMPLETED_AUDIT.json. Second planned replicate training PID967560. Cumulative96,993,280decisions /69,440episodes /9,184evaluation games;0confirmation/admission/upload.

Training allocation diagnostic: period2combined with half-uniform configured mixture yields75%uniform+25%normalizeddifficulty in long-run expectation. At retainedbest, lossweight4 gives44701/46401effective15.51%/14.69%; tested maximumweight16would give18.04%/15.85%, with every opponent still>=12.77%. Analytical probabilities, not actual measured episode frequencies. Full table in entropy000/ALLOCATION_DIAGNOSTIC.json. Current run unchanged.


## Zero-entropy batch completed; stronger loss allocation

s58001:5,242,880decisions /3,823episodes /560.240s learning;224screen games /36.466s evaluation;14W/192D/18L,1/7strict passes (44701=4W/28D/0L). Other W/D/L:45800=2/29/1;46201=4/25/3;46401=1/27/4;46500=2/27/3;47236=0/29/3;47238=1/27/4. Preserve candidate but retain s56001(3/7). Full rawcells/bindings/checkpoint/frozen inputs authenticated in BATCH_AUDIT.json. Session86919exit0. Batch10,485,760decisions /7,709episodes /1,118.827s learning;672evaluation games /109.957s evaluation. Cumulative102,236,160decisions /73,263episodes /9,408evaluation games including original assessment;0confirmation/admission.

Next two-repeat experiment increases allocationlossweight4to16only, keeps entropy0/LR7.5e-6and all other settings. Preserve full7opponent roster,14sampled variants,period2coverage andbothtrainingseats. Training3831500000andconfirmation3501640000reserved. No upload.

Weight16baseline224game replay exact, parentrestore exact, actual14foeprobabilities match tested weight16allocation helper; frozen hashes verified. Cumulative9,632evaluation games; trainer968465confirmed live.


## First stronger-loss-allocation candidate completed

s58500:5,242,880decisions /3,813episodes /557.814s learning;224screen games /36.218s evaluation;17W/196D/11L,1/7strict passes (47236=4W/28D/0L). Other W/D/L:44701=2/28/2;45800=3/27/2;46201=0/30/2;46401=3/28/1;46500=0/30/2;47238=5/25/2. This candidate passes a previously failed opponent but loses elsewhere; retain s56001(3/7,10losses). Rawcells/model/pool bindings/checkpoint/frozen hashes authenticated in iteration_000/COMPLETED_AUDIT.json. Second planned replicate training PID969091. Cumulative107,479,040decisions /77,076episodes /9,856evaluation games;0confirmation/admission/upload.

Paired same-cell comparison s56001tos58500:10parentlosses become9draws+1loss, but9parentdraws+1parentwin become10newlosses. Fourteenwinsbecamedraws,16drawsbecamewins,1winretained. Thus11aggregate losses do not indicate the same failure set or pointwise improvement. Loss identities/margins retained in lossweight16/FIRST_CANDIDATE_PAIRED_COMPARISON.json. No new simulations.


## Weight16 batch completed; coverage frequency experiment

s58501:5,242,880decisions /3,863episodes /558.536s learning;224screen games /36.627s evaluation;13W/200D/11L,1/7strict passes (47236=5W/27D/0L). Other W/D/L:44701=0/30/2;45800=0/29/3;46201=3/26/3;46401=0/31/1;46500=1/30/1;47238=4/27/1. Bothrepeats pass47236but lose elsewhere;retain s56001(3/7). Rawcells/bindings/checkpoint/frozeninputs authenticated in BATCH_AUDIT.json. Session23000exit0. Batch10,485,760decisions /7,676episodes /1,116.350s learning;672evaluation games /110.106s evaluation. Cumulative112,721,920decisions /80,939episodes /10,080evaluation games including original assessment;0confirmation/admission.

Next changes only coverage_period2to4, allowing weighted selection3of4choices and retaining full14variant coverage. Entropy0/LR7.5e-6/weight16/rewards/nativeconditions/strictgate unchanged. Existing sampler8tests pass; freshtraining3851500000andconfirmation3501650000reserved. No upload.

Coverage4baseline224game replay exact, parentrestore exact, actual14foeprobabilities match weight16helper and actualplanperiod4verified; frozen hashes verified. Cumulative10,304evaluation games; trainer969996confirmed live.


## First coverage4 candidate completed

s59000:5,242,880decisions /3,803episodes /555.289s learning;224screen games /36.384s evaluation;16W/193D/15L,0/7strict passes. Opponent W/D/L:44701=1/29/2;45800=1/28/3;46201=4/26/2;46401=2/28/2;46500=3/27/2;47236=3/27/2;47238=2/28/2. Every opponent has a loss, retain s56001(3/7). Rawcells/model/pool bindings/checkpoint/frozen hashes authenticated in iteration_000/COMPLETED_AUDIT.json. Second planned replicate training PID970689. Cumulative117,964,800decisions /84,742episodes /10,528evaluation games;0confirmation/admission/upload.

Next-hypothesis preparation only: coverage4firstcandidate has193draws,ofwhich193end with bothhealthzero. Higher received-damage learning cost .75to1.0could discourage mutual damage but requires evaluation; all-draw opponent remains failure. Actual ControlGuidancewrapper on4syntheticfixture cases preserves exact observations/info/termination/native-call counts while rewarddelta=-.25*received. RECEIVED100_FIXTURE_DIAGNOSTIC.json records scope; no native simulations or performance claims, currentrun unchanged.


## Coverage4 batch completed; received cost1.0

s59001:5,242,880decisions /3,785episodes /565.212s learning;224screen games /36.635s evaluation;14W/198D/12L,0/7strict passes. Opponent W/D/L:44701=0/29/3;45800=1/30/1;46201=2/28/2;46401=1/29/2;46500=3/28/1;47236=3/28/1;47238=4/26/2. Retain s56001(3/7). Rawcells/bindings/checkpoints/frozeninputs authenticated in BATCH_AUDIT.json. Session95236exit0. Batch10,485,760decisions /7,588episodes /1,120.501s learning;672evaluation games /110.436s evaluation. Cumulative123,207,680decisions /88,527episodes /10,752evaluation games including original assessment;0confirmation/admission.

Native received-cost diagnostic completed4episodeexecutions in1.111s, fixedaction6, red589steps/blue627steps perpair. Received.75vs1.0produced exact observations/allinfo except Monitoraggregate/termination/verdict, onlyrewarddelta-.25*received. Eachseat received1.0, bothloss in bothsettings; this is reward-isolation evidence, not modelperformance. Source hashes verified; diagnostic band3871500000reserved.

Next two repeats change receivedcost.75to1.0only; freshtraining3881500000andconfirmation3501660000reserved. Initial one-off preparation ImportError incorrectly importedverify_hashesfromstorage; failed before reservations/output/training. Correct league-module import then validation/prepare succeeded; PREPARATION_CORRECTION.json preserves this. Newcontroller launched once; no frozen source changes or upload.

Received1.0baseline224game replay exact, parentrestore exact, actualworkerplanreceived1.0/full14foes/coverage4verified; frozen hashes verified. Cumulative10,976evaluation games; trainer971763confirmed live.


## First received1.0candidate completed

s59500:5,242,880decisions /3,826episodes /555.705s learning;224screen games /37.538s evaluation;26W/170D/28L,1/7strict passes (46201=5W/27D/0L). Other W/D/L:44701=3/25/4;45800=1/24/7;46401=3/25/4;46500=3/24/5;47236=5/23/4;47238=6/22/4. Fewer draws but greater losses than currentbest, so retain s56001(3/7,10losses). No causal inference from distinct training streams. Rawcells/model/pool bindings/checkpoint/frozen hashes authenticated in iteration_000/COMPLETED_AUDIT.json. Second planned replicate training PID972382. Cumulative128,450,560decisions /92,353episodes /11,200evaluation games;0confirmation/admission/upload.


## Received1.0 completed; intermediate assessment

Both candidates authenticated against raw results, checkpoint identities, original pool and frozen inputs; controller exited0. Batch 10485760 decisions /7608 episodes /1116.958s learning;672evaluation games /112.664s evaluation. Second candidate s59501:20W/183D/21L,0/7strict passes; per-opponent W/D/L44701=3/24/5;45800=2/26/4;46201=4/26/2;46401=4/23/5;46500=1/30/1;47236=3/27/2;47238=3/27/2. Retain s56001(3/7). Cumulative133693440 decisions /96135 episodes /11424 evaluation games including original assessment. No confirmations or admissions.

Predeclared6intermediate checkpoints from lossweight16 will be assessed on consumed development panel, without double-counting prior learning. Independent publication_review found no important defects; compilation and actual live-lock rejection verified. All-pass stops assessment for fresh confirmation; no automatic admission from screening. No GitHub upload.


## Intermediate assessment completed; smaller updates

Six checkpoints: passes3,2,1,0,0,1; W/D/L17/193/14,16/191/17,9/198/17,18/193/13,23/182/19,17/197/10.1,344new development games in218.826s; zero new training. All raw results and identities verified; exit0. Best remains s56001,3/7passes. Cumulative12,768evaluation games. Full opponent details in runs/seven_league_intermediate_20261016/SUMMARY.md. No confirmation/admission.

Next two full-budget repeats compare lossweight16 settings with only LR7.5e-6 to3.75e-6. Entropy0,receivedcost.75,coverage2,lossweight16; original7preserved. Fresh training3901500000 and confirmation3501670000 reserved. Existing tested runner unchanged.

New baseline224games authenticated and same retained-parent score; exact parent restoration and actual worker parameters verified. Trainer973600live. Cumulative12,992evaluation games including baseline, original assessment and intermediate assessment. No confirmation/admission.


## Completed-panel failure anatomy

Authenticated consumed-panel raw records for retained s56001, s58500 endpoint and s59501 endpoint: every draw ends with both health zero (198/196/183respectively). All10retained-parent losses have opponent health<=.05, median.01825; corresponding loss seat counts blue7/red3. s58500loss seat counts blue4/red7, s59501blue13/red8; action-replica concentration also changes. No single seat/replica explanation is established. All recorded loss received damage is classified head-on and none has first-shot advantage. Source completion hashes and details saved in current run COMPLETED_PANEL_FAILURE_ANALYSIS.json. Descriptive evidence only; no new games, no verdict tolerance changes and no active-plan modification.


## First weight16 lower-LR candidate completed

s60000:5,242,880decisions /3,868episodes /564.478s learning;224screen games /36.164s evaluation.13W/201D/10L,0/7strict passes; every opponent has at least one loss. Opponent W/D/L44701=1/30/1;45800=1/28/3;46201=2/29/1;46401=0/31/1;46500=5/26/1;47236=2/28/2;47238=2/29/1. Same total losses as retained best does not satisfy opponent-level pass rule. Retain s56001(3/7). Raw results, model/pool bindings and frozen hashes verified in iteration_000/COMPLETED_AUDIT.json. Second planned replicate PID974476started. Cumulative138,936,320decisions /100,003episodes /13,216evaluation games; zero confirmations/admissions/uploads.

Paired-cell audit of retained s56001 versus s60000: old10losses become9draws+1loss, but8old draws and1old win become new losses. Of16old wins,3remain wins,12become draws,1becomes loss;10old draws become wins. Equal10total losses therefore hides changed failure cells and lost opponent passes.224existing cells authenticated, no new games. PAIRED_TRANSITIONS.json preserves source hashes.


Completed update diagnostic: s60000mean approximate KL2.539e-5 versus6.290e-5/6.527e-5in prior weight16repeats; mean clip fraction.0002759versus.002355/.002355. Smaller measured updates still yielded0/7passes, so reducing per-update movement alone has not established retention of passed opponents. Entropy magnitudes.3405versus.3265/.3339. These are different on-policy state/random streams, not paired causal estimates. Full CSV hashes and statistics in COMPLETED_UPDATE_DIAGNOSTIC.json. No active settings changed.


## GAE runner preparation only

Historical long_credit0999 changed both lambda.99to.999and referenceKL.1; it is inspiration, not a recipe. New v6 only adds validated lambda intervention after exact parent audit, updating model and rollout buffer together and logging it. v5 remains unchanged. Initial test failed for missing module; actual PPO tests and regressions now39passed in1.75s. Independent publication_review found no important defects. Prepared conditional lambda.995experiment, gamma.999unchanged, no launch or native simulation yet. Current batch remains frozen. GAE_RUNNER_PREPARATION.json records review and source hashes.


## Weight16 lower-LR batch complete; GAE.995

s60001:5,242,880decisions /3,829episodes /567.359s learning;224screen games /36.535s evaluation.11W/200D/13L,0/7strict passes. Opponent W/D/L44701=2/28/2;45800=0/30/2;46201=1/28/3;46401=1/29/2;46500=1/30/1;47236=3/27/2;47238=3/28/1. Batch10,485,760decisions /7,697episodes /1,131.837s learning;672evaluation games /110.214s evaluation. Raw cells, bindings, checkpoint hashes verified; session6249exit0. Retain s56001. Cumulative144,179,200decisions /103,832episodes /13,440evaluation games.

New v6 GAE.995batch prepared: two5,242,880step repeats; lambda only changed, gamma.999,LR3.75e-6,entropy0,receivedcost.75,coverage2,lossweight16 unchanged. Automatic LRhalving explicitly overridden to isolate lambda. Actual retained-parent read-only restoration/buffer audit passed; synthetic tests39passed and independent review clean. v6/test/audit hashes frozen. Training3921500000 and freshconfirmation3501680000 reserved. No uploads.

GAE.995baseline224games authenticated and matches retainedparent score; exact restoration and actual.99to.995intervention verified. v6trainer975682confirmed live. Cumulative13,664evaluation games. Existing seven-model pool unchanged.


## First GAE.995 candidate completed

s60500:5,242,880decisions /3,802episodes /562.250s learning;224screen games /36.548s evaluation.14W/189D/21L,0/7strict passes. Opponent W/D/L44701=1/27/4;45800=2/29/1;46201=1/29/2;46401=1/29/2;46500=5/25/2;47236=2/25/5;47238=2/25/5. Retain s56001(3/7). Raw cells, bindings, checkpoint and frozen inputs verified in iteration_000/COMPLETED_AUDIT.json. Second planned repeat PID976426started. Cumulative149,422,080decisions /107,634episodes /13,888evaluation games. Zero confirmations/admissions/uploads.


## Conditional fixed-reference runner prepared

Existing ReferenceAnchoredPPO inspected and unchanged. New v7 adds archive freeze guard, exact-state-preserving reference attachment after parent audit, metadata logging, and v7 subprocess entry. New tests first failed for missing module, then complete regression suite51passed in1.80s. Actual retained s56001loaded read-only: initialized actor/Adam/counters match, attached frozen reference has zero initial KL on synthetic inputs. No native simulations or checkpoint mutation. Independent publication_review found no important defects. Conditional next comparison uses lower-LR baseline lambda.99plus referenceKL.1only, not historical mixed lambda/reference recipe; wait for current GAE batch outcome. Source hashes and review in REFERENCE_RUNNER_PREPARATION.json.


## GAE.995 completed; fixed reference.1

s60501:5,242,880decisions /3,861episodes /567.850s learning;224screen games /36.686s evaluation.13W/197D/14L,0/7passes. Opponent W/D/L44701=3/27/2;45800=0/30/2;46201=1/30/1;46401=1/28/3;46500=0/30/2;47236=4/26/2;47238=4/26/2. Batch10,485,760decisions /7,663episodes /1,130.100s learning;672games /110.746s evaluation. Raw cells/bindings/checkpoints/frozen inputs verified; session2185exit0. Cumulative154,664,960decisions /111,495episodes /14,112evaluation games. Retain s56001(3/7).

Next fixed-reference.1experiment compares lower-LR baseline directly, lambda.99,LR3.75e-6,entropy0,received.75,coverage2,weight16 unchanged. v7uses frozen s56001reference on training rollouts; unchanged public policy/export and simulator. Two full-budget repeats, newtraining3941500000andfreshconfirmation3501690000reserved.51tests/independentreview/actualparentaudit passed and frozen. No reference state selected from unseen confirmation; no upload.

Reference.1baseline224games authenticated and matches retained-parent score. Exact parent restoration, fixed archive hash, optimizer exclusion/state-preservation checks and actual finite referenceKL logging verified. Trainer977476confirmed live. Cumulative14,336evaluation games; no confirmation/admission/upload.


## First reference.1 candidate completed

s61000:5,242,880decisions /3,858episodes /603.309s learning;224screen games /36.198s evaluation.16W/196D/12L,0/7strict passes. Opponent W/D/L44701=3/27/2;45800=0/32/0;46201=2/29/1;46401=4/26/2;46500=3/26/3;47236=2/28/2;47238=2/28/2. All32draws against45800explicitly count as opponent failure. Retain s56001(3/7). Raw cells/bindings/checkpoints/frozen inputs authenticated in iteration_000/COMPLETED_AUDIT.json. Second planned repeat PID978346started. Cumulative159,907,840decisions /115,353episodes /14,560evaluation games. No confirmation/admission/upload.

First reference.1diagnostic: mean on-policy referenceKL.0004674, finalquarter.0006174, max.0012892. Despite small average divergence, retained parent16wins become3wins/11draws/2losses; old10losses become1win/8draws/1loss;9old draws become newlosses. Small average KL does not establish per-game stability. Existing224cells authenticated; no new simulations. Details and hashes in iteration_000/REFERENCE_DIAGNOSTIC.json. Current second repeat unchanged.


## Reference.1 completed; stronger reference1.0

s61001:5,242,880decisions /3,853episodes /606.806s learning;224screen games /36.601s evaluation.17W/187D/20L,0/7passes. Opponent W/D/L44701=3/23/6;45800=1/29/2;46201=2/27/3;46401=3/27/2;46500=4/27/1;47236=2/27/3;47238=2/27/3. Batch10,485,760decisions /7,711episodes /1,210.116s learning;672games /110.225s evaluation. Raw cells/bindings/checkpoints/frozeninputs verified; session87001exit0. Cumulative165,150,720decisions /119,206episodes /14,784evaluation games. Retain s56001(3/7).

Next two full-budget repeats change only referenceKL coefficient.1to1.0, same frozen s56001and all other settings. Actual plain parent compatibility verified, v7unchanged. Learningrate auto-halving overridden to retain3.75e-6. Freshtraining3961500000andconfirmation3501700000reserved. No upload.

Reference1.0baseline224games authenticated, retained score matches, parentrestore exact, actual coefficient1.0/fixedarchive and referenceKL logging verified. Trainer979434confirmed live. Cumulative15,008evaluation games. No confirmation/admission/upload.


## First reference1.0 candidate improved development rank

s61500:5,242,880decisions /3,860episodes /602.233s learning;224screen games /36.691s evaluation.14W/202D/8L,3/7strict passes. Opponent W/D/L44701=1/28/3;45800=2/30/0PASS;46201=2/30/0PASS;46401=2/27/3;46500=2/29/1;47236=3/29/0PASS;47238=2/29/1. Rank(3,-8,14) improves retained s56001(3,-10,16); pass set changes46500to47236, not per-cell dominance. Select s61500as next learning parent; preserve s56001and all originals. No all-pass, no fresh confirmation/admission. Raw cells, bindings/checkpoint/frozeninputs verified. Second repeat PID980418restores s61500exactly while keeping fixed s56001reference/coefficient1.0; actual plan and reference logs verified. Cumulative170,393,600decisions /123,066episodes /15,232evaluation games. No upload.

New best s61500paired-cell analysis vs s56001: old10losses become5draws+5losses; newlosses from1old draw+2old wins. Old16wins become7wins/7draws/2losses;7old draws become wins. Aggregate losses improve10to8without pointwise dominance. Mean training referenceKL4.301e-5(finalquarter4.825e-5), versus prior coefficient.1firstcandidate.0004674on different on-policy streams. These are descriptive consumed-panel/rollout statistics, not a causal or universal guarantee. Source hashes in iteration_000/REFERENCE_DIAGNOSTIC.json; no additional games.


## Reference1.0 batch complete; retain improved parent

s61501:5,242,880decisions /3,861episodes /608.496s learning;224screen games /36.091s evaluation.10W/196D/18L,0/7passes. Opponent W/D/L44701=1/28/3;45800=1/29/2;46201=2/29/1;46401=3/23/6;46500=1/29/2;47236=1/29/2;47238=1/29/2. Retain s61500(3passes,14W/202D/8L). Batch10,485,760decisions /7,721episodes /1,210.729s learning;672games /110.396s evaluation. Raw cells/bindings/checkpoints/frozeninputs verified; session59044exit0. Cumulative175,636,480decisions /126,927episodes /15,456evaluation games.

Next two full-budget repeats start retained s61500with unchanged settings/fixed s56001reference after one rejected descendant. Freshtraining3981500000andconfirmation3501710000reserved. Parent choice recomputed from raw authenticated evidence. Original7preserved, no admission/upload.

Continuation baseline224games authenticated and reproduces s61500score3passes/8losses. Exact parent restore and fixed s56001reference1.0verified; trainer981616confirmed live. Cumulative15,680evaluation games, no confirmation/admission/upload.


## First fixed-reference continuation completed

s62000:5,242,880decisions /3,869episodes /618.676s learning;224screen games /36.158s evaluation.20W/194D/10L,2/7passes(45800,46201). Opponent W/D/L44701=3/27/2;45800=4/28/0;46201=3/29/0;46401=4/27/1;46500=2/27/3;47236=1/29/2;47238=3/27/2. Higher game wins do not outrank fewer opponent passes; retain s61500(3/7,8losses). Raw cells/bindings/checkpoint/frozeninputs authenticated. Second planned repeat PID983242started. Cumulative180,879,360decisions /130,796episodes /15,904evaluation games. No confirmation/admission/upload.


Conditional reference-refresh preflight completed on a new archive: existing migration helper changed only reference_archive/reference_policy_hash metadata; all non-data payloads byte-identical, actor/Adam/counters exact, synthetic policy/value output exact, new self-reference KL0. Source s61500preserved.2existing reference-rebind tests passed in1.35s. No native executions or learning; not a production candidate. Evidence in runs/seven_league_reference_rebase_preflight_20261016. Current continuation unchanged; inspect its final result before any refresh experiment.


### Reference continuation: first-candidate paired audit

Authenticated all224consumed development cells against the retained s61500baseline. Of8baseline losses,2became wins,2draws and4remained losses;6new losses came from2previous wins and4previous draws. Of14baseline wins,4remained wins,8became draws and2losses. Candidate totals20W/194D/10L,2opponent passes; retained parent remains14W/202D/8L,3passes. Mean reference KL0.0000488262(last quarter0.0000460923)does not establish performance retention. Evidence: aircombat-rl/runs/seven_league_reference_continue_20261016/iteration_000/PAIRED_TRANSITION_AUDIT.json. No new training or evaluation executions were added by this analysis. Second trainer PID983242was verified live at elapsed05:08with2,611,808decisions and1,907training episodes; this is a timestamped observation, not a persistent live-state claim.


### Reference continuation batch completed

Both full-budget repeats exited0; frozen hashes, seven-original pool, checkpoint payloads and raw224-cell screens authenticated. s62000:20W/194D/10L,2passes; s62001:6W/201D/17L,0passes. Best s61500retained. 10485760decisions, 7699training episodes. Per-opponent counts and times: aircombat-rl/runs/seven_league_reference_continue_20261016/SUMMARY.md. Proceed with reviewed reference-refresh hypothesis, final-copy and raw-baseline hard gates before training.


### Verified reference-refresh launch

Fresh run seven_league_reference_refresh_20261016uses a metadata-only clone of s61500. All non-data learner ZIP payloads and export/RNG/worker files remain byte-identical; actor/critic/Adam/counters and synthetic inference exact. Two changed fields bind reference to final clone path and its policy hash. READYand1087frozen inputs verified by independent review. Separate224-game baseline authenticated and all raw fields except wall_seconds matched original s61500screen before controller launch; these are consumed development executions.

Learning rate3.75e-6,entropy0,received cost.75,loss allocation weight16,coverage2,GAE.99,reference coefficient1and5,242,880decisions per repeat remain fixed. Fresh training band4001500000and confirmation band3501720000reserved; two repeats planned. PID984543controller/PID984545trainer observed live at launch;22.277GiBavailable memory. Active reference_intervention.jsonconfirms new policy hash326340a9... and learner state preservation. Original s61500and s56001remain unchanged.


### Reference-refresh first candidate completed

s62500completed5,242,880decisions/3,853training episodes in604.674855seconds; separate224-game screen took36.461712seconds. Score20W/190D/14L,0/7opponent passes. Per-opponentW/D/L:44701=1/28/3,45800=1/29/2,46201=3/27/2,46401=8/23/1,46500=0/30/2,47236=3/27/2,47238=4/26/2. All fail under strict no-loss criterion. Raw screen, checkpoint and frozen inputs authenticated; seven-original pool unchanged. s61500retained. Second trainer PID985508started with same retained clone; inspect actual process for live state. No confirmations/admissions/uploads. Completed campaign plus current first candidate:191,365,120decisions,138,479episodes,16,576evaluation games including all counted baselines and separate development assessments.


### Reference-refresh paired transition audit

Authenticated224paired consumed cells: baseline8losses became2wins/3draws/3losses;11previous draws became new losses. Baseline14wins became4wins/10draws, with no win-to-loss changes. Candidate20W/190D/14L therefore does not dominate baseline14W/202D/8L. Mean new-referenceKL0.0000413810,last quarter0.0000460463; different reference anchors make cross-study KLvalues non-equivalent. Evidence: aircombat-rl/runs/seven_league_reference_refresh_20261016/iteration_000/PAIRED_TRANSITION_AUDIT.json. No extra simulations or training. Current second repeat unchanged.


### Reference refresh batch completed

Both repeats exited0and their raw screens, checkpoint bindings, seven-original pool and frozen files verified. s62500:20W/190D/14L,0passes. s62501:8W/204D/12L,1pass(s45800); s61500clone retained. 10485760decisions/7710episodes; per-opponent counts/timing in aircombat-rl/runs/seven_league_reference_refresh_20261016/SUMMARY.md. Refresh alone did not improve the retained screen rank in these repeats. Proceed with half-LRonly hypothesis.


### Refresh lower learning rate launched

Terminal authenticated continuation selected retained s61500clone and automatically halvedLR3.75e-6to1.875e-6after two non-improvements. PRELAUNCH_AUDITverified the only learning-setting change; all other top-level changes are fresh reserved train/confirmation/seed/provenance/frozen inputs. Training band4021500000,confirmation3501730000,seeds63000/63001,two full5,242,880-decision repeats. No other live league at launch;23,929,712,640available bytes. Baseline224raw non-wall cells reproduced exactly and authenticated. Controller986374/trainer986390observed live; runtime reference intervention preserves learner state and fixed new reference1.0. No new code or model mutation, no uploads.


### Refresh lower-LR first candidate completed

s63000completed5,242,880decisions/3,858training episodes in605.699390seconds. Separate224-game screen36.711848seconds:16W/195D/13L,3/7passes(s45800,s46201,s47236). Other opponents:44701=1W/25D/6L,46401=1/28/3,46500=1/29/2,47238=3/27/2; pass opponents45800=2/30/0,46201=3/29/0,47236=5/27/0. Same pass set as retained s61500but13losses versus8; retain s61500under rank(pass count,-losses,wins). Raw cells, checkpoint binding, frozen hashes and original7pool verified. Second trainer995672started; inspect live process. Completed totals201,850,880decisions/146,194episodes/17,248evaluation games; no confirmation/admission/upload.


### Refresh lower-LR first paired audit

Authenticated224paired cells: baseline14wins became9wins/5draws; baseline8losses became2draws/6losses;7previous draws became new losses and7became wins. Mean referenceKL0.0000174662,last quarter0.0000241022. Smaller observed policy movement and more retained winning cells than the previous refresh candidate are descriptive across different sampled training streams, not causal proof. Still13total losses and no rank improvement. Evidence: aircombat-rl/runs/seven_league_refresh_lowerlr_20261016/iteration_000/PAIRED_TRANSITION_AUDIT.json. No new training or simulations; second repeat unchanged.


### Refresh lower-LR batch completed

Both repeats exited0, frozen files and checkpoint bindings verified; seven originals retained. s63000:16W/195D/13L,3passes; s63001:15W/192D/17L,0passes. Retain s61500clone(3,-8,14). Per-opponent counts, full training and evaluation timing: aircombat-rl/runs/seven_league_refresh_lowerlr_20261016/SUMMARY.md. Neither full-budget repeat improved retained rank.


### Refreshed reference second lower-LR comparison launched

Authenticated terminal continuation retained s61500clone and reduced onlyLR1.875e-6to9.375e-7after two non-improvements. All frozen inputs/model payloads verified and no other live league at launch;23,862,108,160available bytes. Fresh training band4041500000,confirmation3501740000,seeds63500/63501; two full5,242,880-decision repeats. Baseline224raw non-wall cells reproduced exactly and authenticated. Controller996734/trainer996748observed live; actual plan and reference intervention confirm intended LRand unchanged reference1.0/state preservation. Best original remains3/7passes,14W/202D/8L; no confirmation/admission/upload.


### Refresh lower-LR2 first candidate completed

s63500completed5,242,880decisions/3,841training episodes in607.832146seconds. Separate224-game screen36.640196seconds:14W/196D/14L,2/7passes(s45800,s46201). Per-opponentW/D/L:44701=3/23/6,45800=1/31/0,46201=2/30/0,46401=2/28/2,46500=1/29/2,47236=3/28/1,47238=2/27/3. Retain s61500(3passes/8losses). Raw cells, checkpoint binding, frozen hashes and original7pool verified. Second trainer997691started; inspect live process. Completed totals212,336,640decisions/153,889episodes/17,920evaluation games; no confirmation/admission/upload.


### Lower-LR2 first paired audit and conditional direction

Authenticated224cells: baseline14wins became9wins/5draws;8losses became7losses/1draw;7draws became new losses and5became wins. Mean referenceKL0.0000102318is smaller than previous first repeat, yet retained loss count/rank did not improve. Descriptive comparison only across distinct training streams. Evidence: aircombat-rl/runs/seven_league_refresh_lowerlr2_20261016/iteration_000/PAIRED_TRANSITION_AUDIT.json; no extra training/games.


### Refresh lower-LR2 batch completed

Both repeats exited0; raw cells, model bindings and frozen inputs verified, original7pool preserved. s63500:14W/196D/14L,2passes; s63501:16W/192D/16L,2passes. Both pass onlys45800/s46201. Best s61500clone retained. Per-opponent results and timings:aircombat-rl/runs/seven_league_refresh_lowerlr2_20261016/SUMMARY.md. Proceed with preflight-tested GAE.95only, holdingLR9.375e-7and all other settings.


### Shorter GAE credit experiment launched

Authenticated terminal continuation retained s61500clone. Automatic LRhalving explicitly overridden to hold9.375e-7and isolate GAE.99to.95. All frozen inputs/model payloads verified, no other live league at launch;23,907,954,688available bytes. Fresh training band4061500000,confirmation3501750000,seeds64000/64001,two full5,242,880-decision repeats. Synthetic.95preflight preserves model/Adam/counters and passes analytic advantages/save-load/subsequent learning; evidence frozen. Baseline224raw non-wall cells reproduced exactly. Controller998893/trainer998914observed live; parent_restore.exact=trueand gae_intervention records.99to.95after restoration, LRunmodified. Original rules/pool/export untouched, no confirmation/admission/upload.


### GAE.95first candidate completed

s64000completed5,242,880decisions/3,864training episodes in611.336836seconds. Separate224-game screen36.287004seconds:14W/198D/12L,1/7pass(s47236). Per-opponentW/D/L:44701=2/24/6,45800=1/30/1,46201=0/32/0,46401=3/26/3,46500=1/30/1,47236=4/28/0,47238=3/28/1. s46201all32draws explicitly fails. Retain s61500(3passes/8losses). Raw cells, checkpoint binding, frozen hashes and original7pool verified. Second trainer999913started; inspect live process. Completed totals222,822,400decisions/161,608episodes/18,592evaluation games; no confirmation/admission/upload.


### GAE.95first paired audit

Authenticated224cells: baseline14wins became8wins/6draws;8losses became6losses/2draws;6draws became new losses and6became wins. Mean referenceKL0.0000119175and entropy0.348575nats. Shorter trace did not preserve the full original pass set in this repeat; no causal attribution across distinct trajectories. Evidence:aircombat-rl/runs/seven_league_gae095_20261016/iteration_000/PAIRED_TRANSITION_AUDIT.json. Zero extra native games/training.


### GAE.95batch completed

Both repeats exited0; raw cells, model bindings and frozen inputs verified, original7pool retained. s64000:14W/198D/12L,1pass; s64001:16W/198D/10L,2passes(s46401,s47236). Best s61500retained. s46401was also passed by early s51501, so this is not a first-ever campaign pass. Current candidate loses formerly passeds45800/s46201; retain this alternative pass-set artifact for future exploration. No production candidate has yet passeds47238. Full per-opponent counts and timings:aircombat-rl/runs/seven_league_gae095_20261016/SUMMARY.md. Next comparison uses lower-LR2baseline(lambda.99)and adds entropy.001only, per prior conditional plan.


### Small entropy experiment launched

Comparison baseline is completed lower-LR2(GAE.99), not immediately precedingGAE.95batch. Authenticated continuation retains s61500clone; latest.95results independently verified non-improving and frozen as decision evidence. HoldLR9.375e-7against automatic halving; only entropy0to.001relative to comparison. All source/model hashes verified, no other live league at launch;26,830,942,208available bytes. Training band4081500000,confirmation3501760000,seeds64500/64501,two full5,242,880-decision repeats. Baseline224raw non-wall cells reproduced exactly. Controller1001088/trainer1001103observed live; runtime entropy_intervention0to.001,GAE.99to.99and intendedLRconfirmed. Original rules/pool/export untouched, no confirmation/admission/upload.


### Entropy.001first candidate completed

s64500completed5,242,880decisions/3,853training episodes in592.818098seconds. Separate224-game screen36.752544seconds:19W/197D/8L,1/7pass(s45800). Per-opponentW/D/L:44701=1/28/3,45800=2/30/0,46201=3/28/1,46401=4/27/1,46500=3/28/1,47236=4/27/1,47238=2/29/1. Total losses equal incumbent8but distributed across6foes; strict pass count takes priority, so retain s61500(3passes/8losses). Raw cells, checkpoint binding, frozen hashes and original7pool verified. Second trainer1001888started; inspect live process. Completed totals233,308,160decisions/169,330episodes/19,264evaluation games; no confirmation/admission/upload.


### Entropy.001batch completed

Both repeats exited0; raw screens, checkpoint bindings, frozen hashes and original7pool verified. s64500:19W/197D/8L,1pass; s64501:17W/195D/12L,1pass. Retain s61500as global incumbent. Counts/timing:aircombat-rl/runs/seven_league_entropy001_20261016/SUMMARY.md. Repeated parameter changes from same retained parent have not improved3-pass incumbent. Next explore preserved s64001alternative pass-set lineage with unchanged settings from its GAE.95study; global incumbent remains separate.


## Alternative s64001 lineage launched

The entropy .001 batch completed: s64500 scored 1/7 (19W/197D/8L), s64501 1/7 (17W/195D/12L); neither displaced the global incumbent. The next two independent seeded continuations start from preserved s64001, which passes s46401 and s47236 on the consumed screen. This is an exploratory lineage, not a stronger overall parent. Full 224-game raw-row replay matched its source before launch. Reference, entropy, GAE, learning rate and native simulation rules match the source settings. Training seeds65000/65001 and fresh training band4101500000 are reserved; confirmation band3501770000 remains unused. Branch-local retention and global-incumbent comparison are separate. The original seven opponents remain unchanged.


### Alternative lineage iteration 000 completed

s65000 completed5,242,880 decisions /3,853 episodes in596.805412s learning time;224screen games took36.825102s. Authenticated result:1/7opponent passes,15W/196D/13L. It improves neither its branch parent(2/7,16W/198D/10L) nor global incumbent(3/7,14W/202D/8L). s64001 remains the branch parent and the planned second seed65001 is training. No confirmation, admission or upload. Per-opponent records and global comparison are in the run SUMMARY.md and iteration_000/GLOBAL_COMPARISON_AUDIT.json.


## Completed alternative s64001 lineage

Both candidates failed to improve branch or global incumbent. Controller exited0. No confirmation or admission.

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3853|596.81|224|36.83|1/6|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|0|28|4|loss|
|s45800_step_002621440|1|31|0|win|
|s46201_step_005242880|2|29|1|loss|
|s46401_step_005242880|5|23|4|loss|
|s46500_step_005242880|1|29|2|loss|
|s47236_step_002621440|4|27|1|loss|
|s47238_step_010485760|2|29|1|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3870|596.87|224|36.94|2/5|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|0|26|6|loss|
|s45800_step_002621440|2|30|0|win|
|s46201_step_005242880|2|28|2|loss|
|s46401_step_005242880|3|25|4|loss|
|s46500_step_005242880|0|29|3|loss|
|s47236_step_002621440|4|28|0|win|
|s47238_step_010485760|2|29|1|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.




## Reference coefficient0.3 follow-up

The alternate s64001 lineage completed two candidates without branch/global improvement (1/7 and2/7). Return to the global s61500 incumbent and compare against refresh_lowerlr2 settings. Change only frozen-reference KL coefficient1.0 to0.3; hold LR9.375e-7, GAE.99, entropy0, reward coefficients, roster coverage and loss weighting. This tests whether a weaker reference penalty helps remaining failures; it is not a claim that the previous penalty caused regression.

Use a fresh checkpoint copy whose learner ZIP changes only reference_coefficient metadata. Keep all learned ZIP payloads, optimizer, RNG, worker state and inference exports exact; preserve original archives. Existing attach guards remain enabled. Synthetic state/inference/attachment checks must pass and224raw native baseline cells must match the consumed incumbent screen before training. New training band4121500000 and confirmation3501780000, two seeds65500/65501,5,242,880decisions each. Screen224games per candidate; only all7passes trigger fresh448confirmation. No source changes, simulation-rule changes, original-model replacements or uploads.


## User-requested original-seven round robin

Reaggregated672unique authenticated existing games (21unordered pairs ×32cells) from the original-seven assessment using the identical current model hashes, eight initial conditions, both seats and two action RNGs. This adds zero new games to campaign totals. Shared-game standings exclude self-play and duplicate opposite-direction panels. s44701 and s46500 each pass1/6opponents; all others pass0/6. No model passes all six. Details: aircombat-rl/runs/seven_original_round_robin_20261016/SUMMARY.md. Training remains independent and unchanged.


## Reference0.3 iteration000 completed

s65500 trained5,242,880decisions/3,839episodes in593.188585s, then224screen games in36.947487s. Result2/7,12W/200D/12L; passes s47236 and s47238, each3W/29D/0L. It does not replace the global incumbent but is preserved as an alternative lineage. The52previously closed production candidates had no s47238pass. This is consumed-panel evidence only. Compared to the incumbent,6old losses remain,1becomes draw,1becomes win;6old draws become losses, and7of14old wins become draws. Second seed65501 is training from the unchanged incumbent copy.


## Completed reference0.3 study

Controller exited0. Both candidates failed to displace the global incumbent. s65500 passes s47236/s47238 but s65501 passes none, so no seed-robust improvement is established. No confirmation or admission.

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3839|593.19|224|36.95|2/5|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|27|4|loss|
|s45800_step_002621440|0|31|1|loss|
|s46201_step_005242880|2|28|2|loss|
|s46401_step_005242880|2|28|2|loss|
|s46500_step_005242880|1|28|3|loss|
|s47236_step_002621440|3|29|0|win|
|s47238_step_010485760|3|29|0|win|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3847|593.00|224|37.20|0/7|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|26|4|loss|
|s45800_step_002621440|3|28|1|loss|
|s46201_step_005242880|3|28|1|loss|
|s46401_step_005242880|5|26|1|loss|
|s46500_step_005242880|2|29|1|loss|
|s47236_step_002621440|2|29|1|loss|
|s47238_step_010485760|2|28|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.




## Alternate s65500 lineage

Reference0.3 produced s65500(2/7,12W/200D/12L; passes s47236 and s47238) and s65501(0/7,19W/194D/11L). Neither beats global s61500(3/7,14W/202D/8L). Explore the preserved first production s47238-passing candidate as a separate branch. Hold its settings:LR9.375e-7,GAE.99,entropy0,reference s61500 coefficient0.3,received cost0.75,loss weighting16,coverage2. Two new5,242,880decision continuations, seeds66000/66001, training band4141500000 and unused confirmation3501790000. No policy/archive modification is needed. Authenticate checkpoint, bindings, frozen hashes and reproduce224baseline raw rows before launch. Branch-local improvement does not replace the global incumbent unless its global rank improves. Original seven and all previous candidates remain preserved. Consumed screens are development evidence; no upload.


## Alternate s65500 iteration000 completed

s66000 trained5,242,880decisions/3,863episodes in596.141717s;224screen games took36.554735s. Result2/7,17W/193D/14L; passes s46201(1/31/0) and s47236(4/28/0). s47238(3/28/1) no longer passes. Candidate fails both branch/global improvement comparison, remains preserved, and no confirmation/admission occurs. Second seed66001 continues from original branch parent s65500. Full opponent records are in run SUMMARY.md and iteration_000/GLOBAL_COMPARISON_AUDIT.json.


## Completed alternate s65500 lineage

Controller exited0. Both candidates failed branch/global improvement; neither retains the s47238pass. No confirmation or admission.

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3863|596.14|224|36.55|2/5|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|3|23|6|loss|
|s45800_step_002621440|2|28|2|loss|
|s46201_step_005242880|1|31|0|win|
|s46401_step_005242880|2|27|3|loss|
|s46500_step_005242880|2|28|2|loss|
|s47236_step_002621440|4|28|0|win|
|s47238_step_010485760|3|28|1|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3862|591.28|224|36.99|0/7|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|0|27|5|loss|
|s45800_step_002621440|1|28|3|loss|
|s46201_step_005242880|4|25|3|loss|
|s46401_step_005242880|3|27|2|loss|
|s46500_step_005242880|3|28|1|loss|
|s47236_step_002621440|3|28|1|loss|
|s47238_step_010485760|2|28|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.




## Planned damage-reward0.25 study

The alternate s65500 continuations scored2/7 and0/7 and lost the parent's s47238pass. Return to global s61500 and compare against refresh_lowerlr2 settings. Change only the learning-only dense dealt-damage coefficient0.5 to0.25, keeping received-damage cost0.75, terminal utility, reference coefficient1.0, LR9.375e-7, GAE.99, entropy0, coverage2 and loss weighting16. This tests whether less reward for damage exchange helps reduce loss regressions; it may also reduce wins and is not assumed beneficial. Existing AimTrainingEnv supports this coefficient, so no frozen source edits are needed.

Before launch: verify exact restoration; synthetic aim_bonus checks must show the reward delta is exactly -0.25*dealt with all other inputs equal, including terminal and no-damage cases; baseline224raw native games must match the incumbent. Use two fresh training seed/IC ranges and a reserved disjoint confirmation range,5,242,880decisions per candidate. Separate224screen and conditional448confirmation; retain every candidate and the original seven. The consumed screen is adaptive development evidence. No GitHub upload.


## Damage0.25 study launched

Verified162synthetic reward cases, parent restoration, independent preparation review and exact224raw baseline replay. Fresh training band4161500000, confirmation3501800000, seeds66500/66501. Only dealt-damage learning coefficient changes versus refresh_lowerlr2; native rules and terminal utility remain fixed. Source settings are frozen.


## Damage0.25 iteration000 completed

s66500 trained5,242,880decisions/3,877episodes in586.827624s;224screen games took37.271382s. Result1/7,15W/197D/12L; only s46401passes(4W/28D/0L). The global incumbent remains unchanged. Candidate archived, no confirmation or admission. Second seed66501 is training from original global parent. Full opponent records and authenticated comparison are stored in the run.


## Completed damage0.25 study

Controller exited0. Both candidates failed to improve the global incumbent(1/7 and0/7). All candidates preserved; no confirmation/admission.

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3877|586.83|224|37.27|1/6|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|26|4|loss|
|s45800_step_002621440|1|30|1|loss|
|s46201_step_005242880|3|28|1|loss|
|s46401_step_005242880|4|28|0|win|
|s46500_step_005242880|2|29|1|loss|
|s47236_step_002621440|2|28|2|loss|
|s47238_step_010485760|1|28|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3861|605.22|224|37.27|0/7|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|26|4|loss|
|s45800_step_002621440|1|30|1|loss|
|s46201_step_005242880|4|27|1|loss|
|s46401_step_005242880|2|26|4|loss|
|s46500_step_005242880|1|29|2|loss|
|s47236_step_002621440|2|28|2|loss|
|s47238_step_010485760|2|27|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.




## Completed greedy deployment diagnostic

Both fixed archived candidates failed all7opponents.448new native games,72.098968s evaluation,zero training. Raw results and mode binding authenticated; original pool unchanged. Canonical sampled incumbent retained. This diagnostic is separate from production-training campaign totals. Cumulative separate assessments now3360games(1568original+1344intermediate+448greedy).

# Greedy deployment-mode diagnostic

No new training. Candidate RNG unused (deterministic); original opponents retain action replicas4900/4901. Consumed8-condition screen, both seats. Finite results do not guarantee universal wins.

|Candidate|Screen pass/foes|W/D/L|New games|Evaluation seconds|Admitted|
|---|---:|---|---:|---:|---|
|/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/seven_league_reference_refresh_20261016/parents/s61500_step_005242880|0/7|34/141/49|224|35.819|False|

|Opponent|W|D|L|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|3|21|8|loss|
|s45800_step_002621440|3|22|7|loss|
|s46201_step_005242880|4|21|7|loss|
|s46401_step_005242880|4|21|7|loss|
|s46500_step_005242880|5|25|2|loss|
|s47236_step_002621440|8|16|8|loss|
|s47238_step_010485760|7|15|10|loss|

|/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/seven_league_reference030_20261016/iteration_000/checkpoints/s65500_step_005242880|0/7|34/134/56|224|36.280|False|

|Opponent|W|D|L|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|21|9|loss|
|s45800_step_002621440|2|21|9|loss|
|s46201_step_005242880|4|21|7|loss|
|s46401_step_005242880|1|22|9|loss|
|s46500_step_005242880|7|22|3|loss|
|s47236_step_002621440|8|15|9|loss|
|s47238_step_010485760|10|12|10|loss|




## Relative potential0.5 study

Damage reward0.25 failed in both seeds; greedy mode increased losses substantially. Keep sampled deployment and return to the global s61500/reference-refresh lower-LR2 baseline. Historical122-plan summary records relative_potential_coefficient1.0; use that as context, not a schedule to replay. Test this coefficient at0.5, with all other baseline settings fixed:damage0.5,received0.75,reference1,LR9.375e-7,GAE.99,entropy0,coverage2,loss weighting16.

This is potential-based learning shaping, so a smaller coefficient is a finite-training optimization hypothesis rather than a new verdict or claim of different optimal behavior. The restored critic was trained with the old shaping scale; any transient/value adaptation must be observed rather than assumed safe or beneficial. Use fresh training seeds67000/67001 and reserved ICs, full5,242,880decisions each. Before launch, verify exact reward delta for synthetic terminal/nonterminal cases, unchanged other settings, model restoration and224raw baseline replay. Screens remain consumed adaptive-development evidence; fresh448confirmation only after all7pass, then append-only admission. No source/physics/verdict changes or upload.

## Relative potential0.5 experiment launched

Run `aircombat-rl/runs/seven_league_relative050_20261016`: only relative potential coefficient changes1.0→0.5 against refresh_lowerlr2 study settings. Damage0.5, received0.75, reference1.0, entropy0, GAE0.99, LR9.375e-7 remain fixed.32synthetic cases passed;224native baseline raw non-wall records match incumbent exactly. Seeds67000/67001, fresh training band4181500000 and reserved confirmation3501820000. Screen remains consumed adaptive development evidence. Two full-budget candidates with separate screens; no changes to original simulation or original seven models. Session18346 controller1011543 trainer1011545 observed live at launch.

## Relative potential0.5 completed

Both candidates trained5,242,880decisions. s67000:3,882episodes,626.330s learning,224games/36.599s evaluation,15W/198D/11L,passes46201/46500 only. s67001:3,846episodes,626.526s learning,224games/36.843s evaluation,15W/198D/11L,no opponent passed. Both retain incumbent61500. Raw screens authenticated; frozen hashes verified; controller session18346 exit0. No confirmation or admission. Consumed screen results do not imply universal performance. Reward reduction did not improve the incumbent in this bounded study; do not treat this as a paired causal conclusion.

## Guidance0 ablation completed

Each candidate trained5,242,880decisions.67500:3,873episodes/639.902s;224evaluation games/36.351s;16W/199D/9L,passes45800/46201.67501:3,863episodes/620.484s;224games/36.708s;18W/198D/8L,passes46201/46500. Both retain incumbent61500. Raw screens authenticated and frozen hashes verified;session35185 exit0. No confirmation/admission. These consumed-panel results do not establish universal performance or a paired causal reward effect.

## Alternate67501 completed

Each candidate trained5,242,880decisions.68000:3,861episodes/628.385s;224games/36.627s;14W/195D/15L.68001:3,839episodes/605.463s;224games/36.946s;14W/198D/12L. Both pass only46201 and retain branch67501(2/7);global61500 remains3/7. Raw screens authenticated;frozen hashes verified;session90781 exit0. No confirmation/admission. Consumed development evidence only;no universal performance claim.

## Alternate58001 completed

Each candidate trained5,242,880decisions.68500:3,817episodes/624.108s;224games/36.747s;16W/190D/18L,0passes.68501:3,814episodes/627.038s;224games/36.128s;15W/198D/11L,passes44701(3W/29D/0L). Branch improves from1pass/18losses to1pass/11losses;global61500 remains3passes/8losses. Raw screens authenticated,frozen hashes verified;session79935 exit0. No confirmation/admission. Consumed adaptive development evidence only.

## Continue68501 first candidate completed

s69000:5,242,880decisions,3,805episodes,623.538s learning;224games/36.642s evaluation,16W/197D/11L. Passes44701 and45800;other five fail. Authenticated raw cells and frozen inputs;checkpoint verified. Retained as branch parent for s69001;global61500 remains3/7. See run SUMMARY.md and ITERATION_000_AUDIT.json. Closed-batch campaign totals remain unchanged until the second candidate completes. No confirmation/admission/upload.

## Continue68501 batch completed

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3805|623.54|224|36.64|2/5|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|3|29|0|win|
|s45800_step_002621440|2|30|0|win|
|s46201_step_005242880|5|25|2|loss|
|s46401_step_005242880|1|29|2|loss|
|s46500_step_005242880|4|26|2|loss|
|s47236_step_002621440|0|29|3|loss|
|s47238_step_010485760|1|29|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Candidate retained as continuation parent.

|1|5242880|3824|625.94|224|36.11|1/6|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|30|0|win|
|s45800_step_002621440|3|28|1|loss|
|s46201_step_005242880|2|29|1|loss|
|s46401_step_005242880|2|26|4|loss|
|s46500_step_005242880|4|25|3|loss|
|s47236_step_002621440|2|27|3|loss|
|s47238_step_010485760|2|27|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.


Both raw screens and frozen inputs authenticated;controller session38752 exit0. Branch69000 retained2/7;global61500 remains3/7. No confirmation/admission. Next user-authorized sequential mutual-zero investigation;no new training launched.

## Sequential diagnosis and nonwin sampling launch

See aircombat-rl/runs/seven_mutual_zero_diagnosis_20261016/SUMMARY.md for authenticated termination/reward evidence. No native bug established and no simulation rule changed. Launch seven_league_nonwin_sampling_20261016 from69000 withloss allocation16->1only;two5,242,880decision candidates69500/69501,train4281500000,confirmation3501870000. Gate224raw replay exact;22tests and independent review passed. Exact actor/critic/optimizer restore verified. Session34231/controller1028104/trainer1028106 live at launch. Temporal and risk-objective comparisons remain pending completion review of this first intervention.

## Higher gamma user-directed experiment launched

User requested tolerating early damage for sustained later unilateral advantage. Nonwin experiment stopped with preserved interrupted learner/checkpoints:1,733,056collected decisions,1,720,320optimized rollout decisions,1,245episodes,206.474s wall;224baseline games,no candidate screen. Do not count as a completed candidate or infer a sampling result.

Gamma-only run seven_league_gamma9995_20261016 uses retained69000,loss weight16,ref58001coef1,LR9.375e-7,GAE.99,entropy0;gamma.999->.9995 in model,rollout buffer,and potential rewards. Native20,480decision smoke2.365s/0complete episodes;complete recorded win/draw/loss tapes separately verify terminal reward and potential telescoping under new gamma.37tests passed,224baseline cells exact,independent review approved;session63793/PIDs1029072/1029074live and exact restore verified. Seeds70000/70001,5,242,880decisions each,train2601500000/confirm3501880000. Standard strict screen and conditional fresh confirmation unchanged. Report sustained-position/rear metrics alongside WDL;gamma alone does not guarantee the desired behavior. Draw/win utility change deferred to a separate comparison.

## Higher gamma first candidate completed

s70000 trained5,242,880decisions/3,790episodes in613.053s;224screen games36.430s,14W/198D/12L,only44701passes.198draws allmutual-zero. Zero rear_damage games,zero received-then-later-rear events,zero rear WEZ streak;no measured positional behavior improvement in this candidate. Raw screen and actual model/pool bindings,frozen inputs and checkpoint authenticated. Candidate archived;branch69000 retained2/7 and global61500 remains3/7. Second seed70001fromsame retained parent is live;no early cancellation or utility change based on one seed. Closed campaign totals remain unchanged until batch terminal;current state includes completed candidate. See run POSITION_REPORT.md and ITERATION_000_AUDIT.json.

## Higher gamma batch complete

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3790|613.05|224|36.45|1/6|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|3|29|0|win|
|s45800_step_002621440|2|29|1|loss|
|s46201_step_005242880|1|30|1|loss|
|s46401_step_005242880|1|29|2|loss|
|s46500_step_005242880|4|24|4|loss|
|s47236_step_002621440|2|28|2|loss|
|s47238_step_010485760|1|29|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3799|600.17|224|36.52|2/5|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|4|28|0|win|
|s45800_step_002621440|2|30|0|win|
|s46201_step_005242880|4|27|1|loss|
|s46401_step_005242880|0|28|4|loss|
|s46500_step_005242880|3|27|2|loss|
|s47236_step_002621440|0|29|3|loss|
|s47238_step_010485760|2|27|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.


Both candidates have zero rear_damage games,zero received-then-later-rear events and zero rear WEZ streak. No positional-behavior improvement measured. Branch69000 retained2/7;global61500 remains3/7. Session63793exit0;raw/model/pool/frozen evidence authenticated. No confirmation/admission. Next sequential learning-only draw-utility comparison after native equivalence validation.

## Terminal draw penalty prelaunch validation

Learning-only43-input terminal preference worker/v9 passed47tests and independent review. Fixed-action native24executions/12outcome-seat-RNG cases reproduced original transitions,observations,termination and verdicts exactly;only draw terminal reward changes-.25,max stepdeltaerror2.78e-17. Monitor episode-return differences match. Diagnostic script self/source locks and complete trace/source hash bindings verified;live source guard rejects before simulation.

Native smoke81,920decisions/52episodes/9.070s:43draws,4wins,5losses;all terminal_preference_total values match settings(winbonus0,drawpenalty.25),saved43-input learner gamma.9995,READY plan binding and exact parent restoration verified. Smoke is not performance evidence. Production draw025prepared from retained69000,seeds70500/70501,train2621500000/confirm3501900000;LR9.375e-7held with explicit override of automatic half-rate recommendation. Only draw penalty changes versus gamma-only study. Baseline replay gate pending;no production learner yet.

## Terminal draw penalty production launched

Baseline224native non-wall records exactly match retained69000. Smoke81,920decisions/52episodes and native24equivalence executions passed;47tests and independent config/code review approved. Session7808/controller1031234/trainer1031236 observed live;actor/critic/optimizer restore exact,training plan winbonus0/drawpenalty.25/gamma.9995/LR9.375e-7 verified. Only terminal draw learning utility changes relative to higher-gamma settings. Source simulation/health verdict/original7 unchanged. Two full-budget candidates70500/70501;conditional448fresh confirmation remains required after all7screen pass.

Current evaluation total27,104=23,296completed production+3,360separate assessments+224interrupted nonwin baseline+224active draw baseline. Native diagnostics224mutual-zero+24utility-equivalence are logged separately. Interrupted nonwin1,733,056collected steps and gamma/draw smokes20,480/81,920steps remain separately recorded,not completed-candidate production totals.

## Draw-utility intermediate actor probe

At immutable70500step655360,offline probe on112consumed source69000 traces/7,306states sampled every20ticks shows meanKL(parent||checkpoint)1.271e-5,mean action-distributionL1change.001460,greedyactionchanges3/7306(.041percent),mean absolute critic value change.165680. After first received damage,mean teacher-action probability.927318 versus parent.927610. Descriptive support-bound observations only;not closed-loop performance,no causal attribution to prior/reference,no new simulations or training decisions,and not a new model-selection panel. Current training remains unchanged. Full source hashes and trace identities retained in run s70500_step_000655360_ACTOR_PROBE.json.

## Draw penalty first candidate completed

s70500:5,242,880decisions,3,799episodes,592.472s learning;224screen games36.918s,16W/196D/12L,passes44701/45800.196draws allmutual-zero,zero rear_damage games,zero received-then-later-rear events and zero rear WEZ streak. Versus parent16W/197D/11L,the observed panel has one fewer draw and one extra loss;no measured improvement and no causal attribution across training randomness. Candidate preserved;branch69000retained2/7,global61500remains3/7. Second seed70501fromsame parent active with exact restore verified.

Final offline actor probe on the same7,306consumed states showsgreedy action changes24/7306(.3285percent),mean parent-to-checkpoint KL7.1004e-6,meanL1probabilitychange.001648,mean absolute value change.173448. After first received damage,teacher-action probability.927928 versus parent.927610. This indicates small changes on this fixed state support;it does not establish why the policy remains frontal or measure new closed-loop performance. Frozen input and checkpoint/READY-plan bindings verified before documenting. Closed campaign totals wait for terminal batch.

## Draw penalty0.25 batch completed

# Seven-model local league

Finite-panel results only; no universal victory guarantee. No GitHub publication.
Originals:7; current pool:7; completed iterations:2; smoke:False.

|Iteration|PPO decisions|Training episodes|Train seconds|Evaluation games|Evaluation seconds|Opponent pass/fail|Promoted|
|---|---:|---:|---:|---:|---:|---|---|
|0|5242880|3799|592.47|224|36.92|2/5|False|

## Iteration 0 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|4|28|0|win|
|s45800_step_002621440|2|30|0|win|
|s46201_step_005242880|5|25|2|loss|
|s46401_step_005242880|1|27|4|loss|
|s46500_step_005242880|3|27|2|loss|
|s47236_step_002621440|0|30|2|loss|
|s47238_step_010485760|1|29|2|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.

|1|5242880|3809|578.84|224|36.19|2/5|False|

## Iteration 1 opponent results

|Opponent|Wins|Draws|Losses|Verdict|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|31|0|win|
|s45800_step_002621440|0|31|1|loss|
|s46201_step_005242880|2|30|0|win|
|s46401_step_005242880|0|30|2|loss|
|s46500_step_005242880|3|24|5|loss|
|s47236_step_002621440|2|26|4|loss|
|s47238_step_010485760|1|28|3|loss|

Decision: Full roster preserved; half-uniform plus observed draw+16*loss+1 allocation. Original continuation parent retained; candidate archived.


Both candidates retain zero rear damage/WEZ streak. Branch69000(2/7,16W/197D/11L)retained;global61500remains3/7. Session7808exit0;raw/model/pool/frozen/READY-plan evidence authenticated. No confirmation/admission. Next sequential comparison winbonus0->1 whiledrawpenalty.25 stays fixed.

## Win bonus prelaunch validation

Native12outcome-seat-RNG cases/24fixed-action executions13.134s preserve transitions/features/verdicts exactly;win bonus0->1 changes only terminal win reward by+1,max stepdeltaerror1.11e-16. Draw penalty remains.25inboth. Monitor reward sums match;live guard and hash/self-source lock checks retained. Native smoke163,840decisions/106episodes17.987s:11wins,85draws,10losses,allterminal_preference_total values exactlymatch win+1/draw-.25/loss0. Saved plan/READY binding,43input,gamma.9995 and exact parent restore verified. These are correctness checks,not performance estimates.

Production71000/71001 uses retained69000,only winbonus0->1 relative to draw study;gamma.9995,drawpenalty.25,LR9.375e-7same. AutoLRhalfexplicitly overridden with provenance. Train2641500000/confirm3501920000reserved;smoke2665500000separate. Independent config review approved;224baseline replay pending.

## Win bonus production launched

Baseline224native non-wall records exactlymatch retained69000. Native24fixed-action equivalence executions and163,840decision/106episode smoke(11wins,85draws,10losses)verify win-only terminal bonus and unchanged draw cost;validation and smoke are correctness evidence only. Existing47qualified tests remain applicable with no worker/runner change. Independent actualconfig review approved. Session18984/controller1033097/trainer1033099 observed live;exactactor/critic/optimizer restore and training-plan bonus1/drawpenalty.25/gamma.9995/LR9.375e-7 verified. Seeds71000/71001,two5,242,880decision candidates;fresh train2641500000/confirmation3501920000. Standard strict screen and conditional fresh448confirmation remain unchanged.

Current evaluation27,776=23,968completed production+3,360separate assessments+224interrupted nonwin baseline+224active win baseline. Native diagnostics224mutual-zero+24draw-equivalence+24win-equivalence are separately logged. Win smoke163,840steps remains separate from completed-production learning totals. No publication.

### Synthetic credit-horizon diagnostic during win-bonus training

The reviewed offline probe in `aircombat-rl/runs/seven_league_winbonus100_20261016/CREDIT_HORIZON_PROBE.json` reconstructs a real SB3 rollout buffer from the saved learner configuration. With zero critic values and a unit reward delayed by 400 decision ticks (20 seconds), direct GAE weight is about 0.014696 at gamma 0.9995/lambda 0.99 and 0.548674 at hypothetical lambda 0.999. The actual SB3 recurrence agrees with the analytical result. No native games, policy updates, or active configuration changes were made. Bootstrapped value estimates also transmit credit, and higher lambda can increase variance; these figures do not establish a hard horizon or improved combat performance. The original executed script and result were preserved when review clarified reconstructed-buffer provenance. A conditional follow-up plan is recorded in `docs/superpowers/plans/2026-10-16-longer-credit-followup.md`; current win-bonus candidates must finish before any such experiment is selected.

### Win-bonus first candidate completed; second candidate active

Candidate `s71000_step_005242880` completed 5,242,880 PPO decisions, 3,793 training episodes and 593.585069 seconds learning. Its separate 224-game development screen took 36.798448 seconds and produced 12 wins / 192 draws / 20 losses; all seven opponent matchups failed. Baseline is 16/197/11 with two opponents passing. All 192 draws ended at both health values zero; rear damage, received-then-later-rear damage, and rear WEZ streak remained zero. Raw cells, actual model/pool identities, frozen inputs, parent restoration, and checkpoint READY/plan binding were verified in `aircombat-rl/runs/seven_league_winbonus100_20261016/iteration_000/COMPLETED_CANDIDATE_AUDIT.json`. This candidate is archived and not retained as continuation parent; no confirmation or admission. Second candidate is active under the same frozen study; do not launch a replacement. Closed-batch campaign totals remain unchanged until this two-candidate batch terminates. The active completed candidate is recorded separately to avoid double counting. Standard evaluation count is now 28,000 including this active screen, excluding separately accounted native diagnostic replays. No GitHub upload.

### Completed first win-bonus candidate: optimizer accounting

Offline `aircombat-rl/runs/seven_league_winbonus100_20261016/OPTIMIZATION_AUDIT_1.json` binds the completed candidate and parent archives, raw authenticated screen, plan/READY identity, and immutable metrics CSV. Each of the 12 Adam parameter states increased by 40,960 applied steps; the attempted-epoch counter increased by 1,024. The saved model used normalized advantages, 640 ticks and 32 environments per rollout, four epochs and batch size512. These are measured optimization counters, not extra learner decisions or evaluation games. Logged approximate KL ranged from 8.82e-8 to 1.50e-5; those aggregate metrics alone cannot exclude transient minibatch early-stop events. No learner archive, active configuration, or native simulation was changed.

### Win-bonus batch closed; lambda-only follow-up qualification

Both winbonus candidates and all224screen cells each were authenticated against actual model/pool inputs and checkpoint plan/READY hashes. Training total10,485,760decisions/7,613episodes/1,181.373seconds. Second candidate71001:14W196D14L,2/7passes;no rear indicators. Both candidates failed to improve69000parent16/197/11,2passes. No confirmation/admission. Campaign now74closed candidates,364,380,160decisions,265,309episodes;24,640production evaluation games including baselines,28,224standard games with separate assessments/interrupted baseline. Diagnostic replay counts remain separate.

Prepared lambda-only follow-up in `aircombat-rl/runs/seven_league_lambda999_20261016`:lambda.99to.999,gamma.9995,winbonus1,drawpenalty.25,LR9.375e-7 and original retained69000parent unchanged. Automatic continuation LRhalf was explicitly overridden to isolate lambda. Fresh reservations2681500000and3501940000. Full-episode native163840smoke prepared separately at2705500000,seed72499;production must wait for verified smoke,savedmodel/buffer consistency,224baseline replay and independent concrete-config review. No GitHub upload.

### Lambda .999 production launched after gates

The163,840decision native smoke completed110episodes(13wins/92draws/5losses) in17.713344learning seconds. All actual terminal utility adjustments, exact parent restoration,43input interface, and saved gamma.9995/model+bufferlambda.999 were checked. This smoke is not a performance estimate. The224game baseline reproduced all previous non-wall raw fields exactly;independent config/finalgate reviews passed. Production session91220/controller1035306/trainer1035319 started with original69000actor,critic,Adam,counters restored exactly. Two candidates71500/1 use5,242,880decisions each;onlylambda changes relative to prior utility study,with LRhalf override explicitly recorded. Preserve frozen configs;smoke hashes continue to match productionconfig. Existing strictevaluation/admission protocol unchanged. Standardevaluation total28,448 includes224activebaseline;closedcampaign74candidates remains separate. No GitHub upload.

### Early lambda .999 actor distribution probe

The655,360decision checkpoint was examined offline on the same112consumed parent69000traces/7,306sampled states used by the earlier draw study. The unchanged audited probe script reconstructs teacher history at every tick and verifies trace/model/frozen hashes. Greedy action changed on18/7,306states(0.2464%),including3/4,590post-received-damage states. Mean parent-to-checkpoint KL was6.06765e-6;mean L1 distribution change0.00167916;mean absolute critic value change0.109243. Post-hit teacher-action probability was0.927562 versus parent0.927610. These are descriptive fixed-state early-checkpoint measurements,not candidate closed-loop performance,causal attribution,unseen evidence or admission selection. No new games or learner decisions were executed;production settings remain frozen. Source: `aircombat-rl/runs/seven_league_lambda999_20261016/s71500_step_000655360_ACTOR_PROBE.json`.

### Lambda .999 first candidate completed; second active

First candidate71500 completed5,242,880decisions/3,807episodes/591.125085seconds learning. The separate224game screen took35.995932seconds and produced10W196D18L,1/7strict passes(45800only). Baseline remains16/197/11,2passes,so69000parent retained. All196draws were mutual-zero;rear damage,received-then-later-rear,and rear WEZ streak remained zero. Raw screen,model/pool,READY/plan,frozen inputs and exactparentrestore were authenticated in `aircombat-rl/runs/seven_league_lambda999_20261016/iteration_000/COMPLETED_CANDIDATE_AUDIT.json`. No confirmation/admission. Second candidate71501 active,trainer1036158,controller1035306/session91220.

Final fixed-state actor probe on7,306consumed states found9greedy action changes(1/4,590post-hit states),KL7.80423e-6,meanL1change0.00147083,criticabsolutechange0.155026. These are not closed-loop/causal or unseen results. Saved optimizer audit confirms40,960applied Adam updates per parameter and1,024attempted epochs,matching the previous study optimization count;counter equality does not imply equal learning effectiveness. Completed-batch campaign stays74candidates until this batch finishes;the active completed candidate is accounted separately. Standard evaluation total28,672includes this screen;diagnostic executions remain separate. No upload.

### Conditional update-size design review during second lambda candidate

The conditional plan `docs/superpowers/plans/2026-10-16-update-size-followup.md` was independently reviewed. If the current batch fails meaningful progress,consider LR9.375e-7to3.75e-6 with the new gamma/utility/lambda context held fixed,after authenticating terminal results. This is motivated by measured small distribution movement despite40,960Adam steps;it does not establish LR as the cause. Report sampled-probability/KL alongside greedy changes,actualoptimizer counters and policy displacement;LR4x need not mean updates4x. Adaptive opponent allocations preclude equal-exposure paired causal claims. No next parent/run selected,no conditions reserved,and current frozen settings unchanged. Review: `aircombat-rl/runs/seven_league_lambda999_20261016/UPDATE_SIZE_PLAN_REVIEW.json`.

### Lambda .999 closed; update-size qualification

Both lambda candidates were authenticated from raw224game screens,actual model/pool bindings,READY/plan and frozen inputs. Second result14W194D16L,1/7passes;allrearindicators0. Batch learning10,485,760decisions/7,612episodes/1,183.897555seconds. No confirmation/admission;69000parent and global61500 preserved. Closedcampaign76candidates,374,865,920decisions/272,921episodes;25,312productionevaluationgames including8,288baselines;28,896standardgames with separateassessments/interruptedbaseline. Diagnosticsremainseparate.

Selected reviewed conditional update-size experiment after this terminal failure to improve:LR9.375e-7to3.75e-6 only,with gamma.9995/lambda.999/winbonus1/drawpenalty.25/reference58001coef1/prior6/entropy0 heldfixed. Newoutput `aircombat-rl/runs/seven_league_update375_20261016`,training2721500000,confirmation3501960000,seeds72000/1. Automatic continuation LRhalf explicitlyoverridden and recorded. Native163840full-episode smoke separate at2745500000,seed72999,session18219. Production requires savedmodel and actualoptimizerLR checks,exactrestore,native224baseline replay and concrete review. No publication.

### LR3.75e-6 production launched after qualification

Native smoke completed163,840decisions/109episodes(9W92D8L)/18.267930learning seconds. Exactparent restoration,READY/plan,43input interface,actualterminalutility,savedmodel andoptimizerLR3.75e-6,gamma.9995/model+bufferlambda.999 all verified. Smoke is not performance evidence. Baseline224pairedgames reproduced all non-wallrawfields of69000control exactly;script unchanged fromreviewedlambda gate andproductionconfigremainedbyte-identical. Independentconfigreviewapproved. Session78345/controller1037195/trainer1037208started after allgates with exactoriginal69000actor/critic/Adam/counters restoredbeforeintervention. Two candidates72000/1,5,242,880decisions each;training2721500000,confirmation3501960000. Strictscreen/admissionprotocolunchanged. Standard evaluation29,120includesnew224activebaseline;closedcampaign76remains separate. No GitHub upload.

### Early higher-LR fixed-state distribution movement

At655,360collected decisions, the higher-LR candidate changed greedy action on27/7,306consumed states versus18for previousLR at the same decision count. Post-hit changes9/4,590versus3. Mean parent-to-checkpoint KL6.29439e-5versus6.06765e-6;meanL1change0.00523143versus0.00167916. The comparison verifies identical parenthash,inputtraces,selectedkeys,probe script and teacherconfig,and actual savedLR/gamma/lambda. Newreport `aircombat-rl/runs/seven_league_update375_20261016/EARLY_UPDATE_SIZE_COMPARISON.json`. This describes larger distribution movement on these states;differenttrainingrandomness precludes causalattribution, and larger movement does not imply improved positioning/wins. No new nativegames,learningdecisions or settingchanges. Current production continues.

### Higher-LR first candidate complete; second active

Candidate72000 completed5,242,880decisions/3,802episodes/584.781184learningseconds. Separate224screen36.408696seconds:11W199D14L,1/7passes(44701only),below69000control16/197/11,2passes;parentretained. All199draws endedbothHPzero;rear indicators0. Rawcells,actualmodel/pool,READY/plan,frozeninputs andexactparentrestore verified in `aircombat-rl/runs/seven_league_update375_20261016/iteration_000/COMPLETED_CANDIDATE_AUDIT.json`. No confirmation/admission. Second72001 ACTIVE,trainer1037944/controller1037195/session78345.

Final consumed-state actorprobe changed greedyaction on7/7306states(3/4590post-hit),KL3.07948e-5,meanL1change0.00305357;descriptive,notcausal/unseen evidence. SavedAdam counters show40960appliedsteps perparameter/1024attemptedepochs. Logged approximateKLmax0.000115973 andclipfractionmax0.00302734cover recordedstatistics only. Standard evaluation29,344includesactivebaseline+firstscreen;closedcampaign76excludesthisactivecandidateuntilbatchterminal. No upload.

### Completed-screen duration analysis and conditional rollout design

Authenticated baseline69000and72000screens(448existinggames,no newgames) show everyfirstreceived-to-terminal delay exceeds32seconds,minima37.0/37.1,medians37.925/37.900;episode maxima97.9/99.8seconds. Rawcell hashes andstatistics independentlyrecalculated andmatched. This is evaluation-duration evidence,not measured trainingbuffer phase/distribution or proof of missingcredit;criticbootstrap transmits credit acrossboundaries. Source `aircombat-rl/runs/seven_league_update375_20261016/DURATION_ANALYSIS.json`.

Conditional640to2048tick(32to102.4s)rolloutdesign reviewed in `docs/superpowers/plans/2026-10-16-rollout-length-followup.md`. Stillmay split episodes andchanges mixing/updatetiming;longerbuffer isnotguaranteed performanceimprovement. Total5,242,880decisions/checkpoint655,360divideexactly;80rollouts/320attemptedepochs/max40960Adamsteps perparameter iffullepochs. Implementationtests,memory,nativesmoke,baselineandfinalreview remainrequired. No implementation/newparent/runselection/reservation;currenthigherLRcandidate1 continuesunchanged. Review evidence `aircombat-rl/runs/seven_league_update375_20261016/ROLLOUT_LENGTH_PLAN_REVIEW.json`.

### Rollout implementation qualified offline while v9 remained active

New `tools/seven_model_rollout.py` and isolatedv10 support640/2048emptyrestoredstandard buffers,exactbatch/checkpoint/decisiondivisibility,and explicitrunner/helperfrozenhashguards. Helper runs afterexactactor/critic/Adam/counterrestore;v9andallactivefrozeninputs unchanged. REDmissinghelper followedbyGREEN71tests4.47seconds coversrealReferencePPOstate/save/reload,longdelayedreturns,doneboundaries,32environmentshape,invalid/partial/mismatchedbuffers,640noopandhashguards. Independentreviewapproved.

Actualcurrentretained69000source additionallypassed2048x32x43buffer,50secondrewardrecurrence,doneisolationandsave/reloadwithactor/critic/Adam/countersexactandsourceunchanged. No nativegamesorlearningexecuted. Theseareimplementationcompatibilitychecks,notnextparentselectionorperformanceevidence. Unittrainingusedreferencecoefficient0;future native smoke mustattachactualfrozenreferencewithpositivecoefficient1. CurrenthigherLRbatchmustfinishbeforeselection/reservations;native196608decisionfull-episodesmokeand224baselinegatesremainrequired. Reports `aircombat-rl/runs/seven_league_update375_20261016/ROLLOUT_IMPLEMENTATION_QUALIFICATION.json` and `ROLLOUT_SOURCE_QUALIFICATION.json`.

### Higher-LR batch closed; rollout2048 actual qualification

Both higherLRcandidates authenticated:11/199/14,1pass and11/196/17,0passes;allrear0;69000parentretained. Batch10,485,760decisions/7,606episodes/1,182.145389seconds. Closedcampaign78candidates/385,351,680decisions/280,527episodes;25,984productionevaluationgames including8,512baselines;29,568standardgames with separateassessments/interruptedbaseline. No confirmation/admission;pool7.

Selected2048rolloutstudy at `aircombat-rl/runs/seven_league_rollout2048_20261016`:nsteps640to2048only;gamma.9995/lambda.999/LR3.75e-6/utilities/reference/prior unchanged,autoLRhalf override explicit. Reviewedv10/helper/tests/evidence frozen. Productionseeds72500/1,training2761500000,confirmation3501980000. Native smoke196608decisions/132episodes(12W111D9L)/21.606818secondspassedactual2048x32x43buffer,savedmodel+optimizerLR,utility,exactrestore,positivefixedreferencecoef1training;1536Adamsteps perparameter/12epochs. Smoke isnotperformanceevidence.224baselineactive43240/PID1039182;productionnotstarted. No upload.

### Rollout2048 production launched after all gates

Code/unit/sourcequalification,independentimplementation/config/finalreviews,nativepositive-reference196608smoke(132episodes),saved2048x32x43buffer/utility/LR/gamma/lambda/exactrestore,and224baselineallnon-wallrawfields exactpassed. Productionconfig stayedbyte-identical aftersmoke. v10session28192/controller1039292/trainer1039305started with exact69000actor/critic/Adam/counters beforebufferintervention;actualrollout_intervention.json confirms640to2048and65536samples perrollout. Two72500/1candidates5,242,880decisions each,655360checkpoint,80fullrollouts each. Strict224screen/conditionalfresh448confirmation/append-onlyadmissionunchanged. Standard evaluation29,792includesnew224activebaseline;closedcampaign78stillseparate. Largerbuffer doesnotguaranteefull-episodecontainment orperformance;update timing/mixingalsochanges. No upload.

### First2048rollout checkpoint: immutable budget and actor diagnostics

The655,360decision72500checkpoint passedartifact/READY/planbinding,saved2048tick/32env/batch512/fourepoch/gamma/lambda/LR/referencechecks. SavedAdamcountersexactly5120updates perparameter across12states;epochcounter40and10fullrollouts. This verifiesactualbudget;epochcounterdiffersfrom640tickrunswithoutimplyingreducedminibatchupdates. Evidence `aircombat-rl/runs/seven_league_rollout2048_20261016/EARLY_CHECKPOINT_BUDGET_AUDIT.json`.

Fixedconsumed7306stateprobe:9greedychanges(4/4590post-hit),meanKL1.58727e-5,meanL1change0.00197911,meanabsolutecriticchange0.155767. Parentpost-hitteacheractionprobability0.927610versuscheckpoint0.927786. This isdescriptivefixed-state earlycheckpoint evidence,notclosed-loopperformance,causalattribution,unseen selection oradmission. No additionalnativegamesorlearnerdecisions;currentproductioncontinuesunchanged. Source `aircombat-rl/runs/seven_league_rollout2048_20261016/s72500_step_000655360_ACTOR_PROBE.json`.

### First2048candidate completed;second active

Candidate72500completed5,242,880decisions/3,776episodes/585.855720seconds learning. Separate224screen36.706686seconds:9W200D15L,0/7passes;69000control16/197/11,2passes retained. All200draws mutual-zero;rearindicators0. Rawcells,actualmodel/pool,READY/plan,frozenrunner/helper/inputhashes,exactrestoreandactualsaved2048settingverified in `aircombat-rl/runs/seven_league_rollout2048_20261016/iteration_000/COMPLETED_CANDIDATE_AUDIT.json`. No confirmation/admission. Second72501ACTIVE,trainer1040016/controller1039292/session28192.

Savedoptimizer40960Adamsteps perparameter/320attemptedepochs/80metricsrows. Finalfixed-stateprobe26greedychanges/7306states(7/4590post-hit),meanKL3.53926e-5,L1change0.00303384,criticabsolutechange0.0930633. Descriptiveconsumedstates only;notcausal/unseen/closed-loop evidence. Standard evaluation30,016includesactivebaseline+firstscreen;closedcampaign78excludesactivecandidateuntilbatchterminal. Allmodels/results retained;no upload.

### Historical prior context and isolated reference-scalar semantics

Historicalglobalprior3study endpoints470/672and483/672wins regressed against its498/672prior6comparator;differentparent/settings/protocol,not currentstrictseven-opponentevidence. Do notcopythatstudyblindly orconcludeallpriorreductionsfail. Source `aircombat-rl/runs/aim_cpu_prior3_20261013/TWO_REPEAT_REVIEW.json`.

Isolatedactualclass counterfactual in `aircombat-rl/runs/seven_league_rollout2048_20261016/REFERENCE_PRIOR_COUNTERFACTUAL.json`:changing copied69000learnerprior6to3thenattaching actual58001archivecopieslearner scalarprior3into referencebeforeloadingarchive tensors. Reference tensorstate equalstoarchive,but scalarprior3differsfromarchiveprior6andonesyntheticfeaturevectorproducesdifferentactiondistribution,withteacheroddsfactor exp(-3). Actor/critic/Adam/counters andsourcehashes preserved;zerogamesorlearning. Currentlearner/referenceprior6matcharchive6andareunaffected. This is semanticsverification,not performanceevidence,activebugclaim,nextstudyselection orauthorizationtoalterfrozeninputs. Futurepriorinterventionmustexplicitlydefine/freeze reference scalar semantics;currenttrainingcontinuesunchanged.

### Conditional reference freedom design reviewed

If current2048batchfailsprogress,consider referencecoefficient1to.3onlywith actualarchive58001andbothprior6fixed;gamma/lambda/rollout/LR/utility/allocation/native/exportsettingsunchanged. Motivationissmallmeasuredpolicymovement despitefullupdatebudgets,notcausalproofanchoristhecause. Useauditedmetadata-onlyseparatecopy,allnon-dataZIPpayloadsandnonlearnerartifacts exact,exactactor/critic/Adam/countersandsyntheticinference,sourceimmutableandREADYcloneprovenance. Sourcecomparisonpanelmustbetheexactpanelthatevaluatedselectedparent,notautomaticallyruninitialbaseline. Nativefull-episodesmoke+224baselineandconcretereviewremainrequired;strict224screen/conditional448confirmationunchanged. Plan `docs/superpowers/plans/2026-10-16-reference-freedom-followup.md`,review `aircombat-rl/runs/seven_league_rollout2048_20261016/REFERENCE_FREEDOM_PLAN_REVIEW.json`. No nextparent/runselected,noreservations,nosimulationorlearning,activefrozenstudyunchanged.

###2048batchclosed;long-rollout reference.3qualification

Both2048candidates authenticated:9/200/15,0passes and10/203/11,2passes;allrear0;69000control16/197/11,2passes retained. Batch10,485,760decisions/7,590episodes/1,181.962866seconds. Closedcampaign80candidates/395,837,440decisions/288,117episodes;26,656productionevaluationgames including8,736baselines;30,240standardgames withseparateassessments/interruptedbaseline. No confirmation/admission;pool7.

Selected reviewedreferencecoefficient1to.3onlyin2048/gamma.9995/lambda.999/LR3.75e-6/win1/draw.25context. New `aircombat-rl/runs/seven_league_longref030_20261016` preservesoriginal69000andusesseparatemetadata-onlycopy:onlyreference_coefficientJSONfieldchanged;allnondataZIPpayloadsandnonlearnerbytesexact,actor/critic/Adam/countersandbitwiseinferenceexact,reference58001actualfunctionandbothprior6verified. READYexplicitcloneprovenanceandunicode-safepathremap. Exactsourcepanel=previous2048baseline69000. Copyaudit/config/preparationcodefrozen. Production73000/1,training2801500000,confirmation3502000000;native196608smoke73999band2825500000/session6943active,planexplicitlyusescopyparent/.3. Baselinewillbindsourceandcopyseparatelythencomparerawbehavior. Productionnotstarted;no upload.

### Reference coefficient 0.3 production launched

The metadata-only parent copy and concrete settings passed independent review. Native smoke completed196608 decisions and131 episodes in21.841247 learning seconds, with actual reference coefficient0.3,2048x32 buffer, saved LR/gamma/lambda/utility and exact copied-parent restoration verified. All12 Adam parameter states increased by1536 updates; the epoch counter increased by12. This smoke does not estimate performance. The separately bound source and copied parent reproduced all224 baseline games exactly except wall time. Production config remained byte-identical after smoke.

Production session72237/controller1041495/trainer1041508 started from the audited metadata copy with exact original actor/critic/Adam/counters. The fixed reference archive58001 and both prior values6 are unchanged. Two seeds73000/1 use5242880 decisions each; training band2801500000, confirmation band3502000000. Strict screens, fresh confirmation and append-only admission remain unchanged. All original models and source69000 are preserved. Standard evaluation count30464 includes the new active224 baseline; the closed campaign remains80 candidates. No GitHub upload.

### Early reference coefficient comparison on consumed states

At655360 decisions, reference0.3 candidate changed the greedy action on58/7306 fixed consumed states versus9 for the reference1 candidate at the same collected decisions. Post-hit changes were13/4590 versus4. Mean parent-to-checkpoint KL was6.89721e-5 versus1.58727e-5; mean L1 change0.00595488 versus0.00197911.

The comparison rechecked actual source/copy ZIP payload identity except the reference coefficient metadata, identical teacher/trace inputs/keys/probe script, checkpoint hashes and saved coefficient/rollout/LR/gamma/lambda settings. This is descriptive policy movement under different training randomness, not paired causal evidence, closed-loop performance or unseen selection. Larger movement does not establish better positioning or wins. No extra games or learning decisions were executed. Evidence: `aircombat-rl/runs/seven_league_longref030_20261016/EARLY_REFERENCE_COMPARISON.json`. Production continues unchanged.

## Offline command and actual-turn persistence,2026-10-16

Authenticated224consumed69000tapes,zero new simulations/learning:median longest posthit actual same-direction turn31.4s;all224have >=5s. Heading-command run median31.8s. Action changes alone do not show inability to sustain turns;action repetition is not selected on this evidence. Report:`aircombat-rl/runs/seven_league_longref030_20261016/ACTION_PERSISTENCE.md`. Frozen reference0.3 study continues;no configuration or acceptance changes.

## Reference0.3 first candidate complete,2026-10-16

73000:5,242,880decisions,3,811episodes,601.313776538s learning;224screen games,36.572652176s. W/D/L10/205/9;strictpasses47236and47238. Previous69000parent16/197/11,passes44701and45800. Thus fewer losses but fewer wins and changedpassedfoes;no rear damage,received-then-rear,or rear streak. Rank selects73000 for secondcandidate73001;original69000/global61500 preserved.

|Opponent|Wins|Draws|Losses|Strict result|
|---|---:|---:|---:|---|
|s44701_step_002621440|1|30|1|loss|
|s45800_step_002621440|1|30|1|loss|
|s46201_step_005242880|2|27|3|loss|
|s46401_step_005242880|0|29|3|loss|
|s46500_step_005242880|4|27|1|loss|
|s47236_step_002621440|1|31|0|win|
|s47238_step_010485760|1|31|0|win|

Raw224cells,model/pool/READY/frozen bindings and next exact parent restore verified. SavedAdam delta40960perparameter,epochcounter320. `FIRST_CANDIDATE_AUDIT.json` and `OPTIMIZATION_AUDIT_1.json` in `aircombat-rl/runs/seven_league_longref030_20261016`. Closedcampaign80 excludes this activecompletedcandidate;activebatch separately5,242,880decisions/3,811episodes. Standard evaluation30,688includes224newscreen;no confirmation/admission. Live secondtrainer1042667,controller1041495/session72237;inspectactualprocess. No publication.

## Offline health timing and conditional threat-prior design

224authenticated consumed69000tapes,zero new simulations/learning:medianhealth~.505 fromfirsthit+2through+10seconds;opponent-onlyWEZ66ticks,both22426of287068. Existingconflictpriorgate requiresopponent-only,so66is onlyanupperbound,notexactactivation. All224metadata joins andfirst-hit/samplehealth reconstructed in `HEALTH_TIMING_BINDING_AUDIT.json`. Independentdesignreview foundno importantblockingissues.

Conditional `docs/superpowers/specs/2026-10-16-threat-prior-design.md` and plan proposepublicenemyWEZ bias6to3,independently fixed originalreferencefunction,explicitchangeduntrainedcontrol,gatetransitionlogging,andstrictfreshconfirmationevenforpassingcontrol. No implementation/source/conditions/run selected;current73001continuesunchanged. Completed73000offlineactorprobe changed11/4590posthitgreedyactions;not performance/causal proof. No publication.

## Reference0.3 batch closed and next study selected

Controller72237exit0 and terminalfile verified;no live sourceprocesses. Total10,485,760decisions/7,593episodes/1184.868844773slearning;672evaluationsincluding224baseline. Candidate73000:10/205/9,2passes47236/47238.73001:14/197/13,2passes45800/46201;3,782trainingepisodes/583.555068235slearning,36.459465616sevaluation. Both rearindicatorszero.73000retainedbyrank;global61500unchanged. `COMPLETED_AUDIT.json` bindsraw/model/pool/READY/frozen/ledger. Campaign82/406,323,200decisions/295,710episodes/30,912standard evaluations;diagnosticsseparate. No confirmation/admission.

Selectednextstudy is the reviewedpublicthreatprior design,source73000anditsactualiteration000screen;original58001reference AimPriorPolicy/prior6 independentlyloaded/verified. `THREAT_PRIOR_SOURCE_SELECTION.json`;implementationpending,noruncreated,nobandsreserved. Newpolicyfunctionchangesmustbeexplicitandqualifiedbeforeproduction. No publication.

## Threat-prior policy and independent reference implemented

Newfiles only:threat_prior_policy.py/threat_prior_numpy.py/fixed_function_reference_ppo.py. RED/GREEN recorded in `aircombat-rl/runs/seven_threat_prior_reference_qualification_20261016/UNIT_VALIDATION.json`. Actual73000source on7306consumed69000states:511gated;actor/critic/Adam/counters/RNG preserved,no-gate function/critic/roundtrip exact,NumPyparityverified. Actual58001full reference AimPriorPolicy/prior6function independently restored and bitwiseexact underbothgatevalues;positive-reference tiny synthetic training/save/loadtests passed. No nativeperformance evidence,newlearning/evaluation/reservations. Export/officialsampledloader/v11runner/gatemetrics andindependentreview/nativequalification stillrequired beforeproduction. Original7/source/referencefrozenfilesunchanged;no publication.

## Threat-prior implementation and standalone qualification complete

Tasks1–3complete;106relatedregressionchecks passed4.68s. Officialloader NumPyexports on7306consumedfeatures:maxTorchprobabilityerror4.78e-7. Sixsequentialcases/7464predictions(two consumedrawtapes xgreedy/4900/4901) matchedexpectedactionsandpublicclockreset. No nativegames/learning. Gateinstrumentationusespre-actionstates;originalphysics/observations/reward/verdictcallsunchanged. Newv11freezesallhelpers,usescomplete originalreference,dispatchesv11,retainsstrict224/448andappendonlypool.

Wholecodeandboundedguardreviewby/root/publication_reviewfoundnoCritical/Importantblockers. AddedRED/GREENguardpreventsallpasszero-stepcontrolsilentlystartingtraining;outerqualificationmustfreshconfirmexactmodelbeforeadmission. Exportqualificationpredatesonlythisguardaddition;exactsource-reconstruction/hashaddendumprovesallinferencehelpersunchanged. Evidence:`aircombat-rl/runs/seven_threat_prior_export_qualification_20261016/EXPORT_QUALIFICATION.json`,guardaddendumand`aircombat-rl/runs/seven_threat_prior_reference_qualification_20261016/WHOLE_IMPLEMENTATION_REVIEW.json`.

Task4native-readyconversion/frozenconfig,196608full-episodesmoke,224changedcontrol,conditional448freshconfirmationandconcreterunreviewremainmandatorybeforeproduction. No newrun/reservation/nativejobsstarted;allpriorstatsunchanged;original7andallmodels preserved;no publication.

## Public-threat prior native qualification and active production

NewREADY copy preserves73000actor/critic/Adam/counters,deliberatelychangespolicyonlyunderpublicenemyWEZ,base6/threat3. Original58001complete referenceclass/scalar/tensors remain fixed. Frozenproductionconfigunchangedafterqualification;source andoriginal7preserved.

Native smoke39462exit0:196608decisions/132episodes/22.147234442s;2048x32buffer,gamma.9995/lambda.999,LR3.75e-6,terminalutilitywin1/draw-.25,positivefixedreference.3verified;1536Adamstepsperparameter/12epochs. Completedepisodegateactive13311/176651steps,440transitions;unfinishedepisodesexcluded. Smokeisnotperformanceevidenceandnotproductionparent.

Changeduntrainedcontrol50768exit0:224games/37.311526779s,13W195D16L,0/7passes;original73000 10W205D9L,2/7. All224nonwallcells changed;zero rear damage/received-then-rear. Gate22271/287723decisions(7.7404%),709transitions. Thisisworseinstallationeffect,notlearnedresult. No confirmation/admission;reservedcandidateconfirmationconditionsunused.

|Opponent|Wins|Draws|Losses|Strict result|
|---|---:|---:|---:|---|
|s44701_step_002621440|0|31|1|loss|
|s45800_step_002621440|2|29|1|loss|
|s46201_step_005242880|3|26|3|loss|
|s46401_step_005242880|0|29|3|loss|
|s46500_step_005242880|4|25|3|loss|
|s47236_step_002621440|2|27|3|loss|
|s47238_step_010485760|2|28|2|loss|

Finalindependentreviewapprovedtwo5242880decisioncandidatesfromuntrainedgatedcopy,not smokelearner. Production67408 ACTIVE(controller1045643/trainer1045648);training2841500000,confirmation3502020000reserved;strict224screen+conditionalfresh448verificationandappendonlyadmissionunchanged. Comparetrainedresultsagainstbothcontrolandoriginal/globalmodels;do notclaimuniversalvictory. Records:`aircombat-rl/runs/seven_league_threat_prior_20261016/SMOKE_VALIDATION.json`,`CONTROL_QUALIFICATION.json`,`REVIEW.json`,`PRODUCTION_PREFLIGHT.json`. Standard eval31136includesactive224control;closedcampaign82/406323200decisions unchanged. No publication.

## Early gated reference-direction diagnostic and conditional follow-up

At655360additionaldecisions,fixed7306consumed69000states(511gated):meanKL(reference||policy)gated.324642to.201086;CEMteacherrecommendationprobability.261690to.306022versusactual58001reference.472844. InactivegroupmeanKLmovesfarther(.0000783to.0007935). Descriptivefixedstateevidenceonly,not referencecausality/trainingdistribution/tacticalperformance. `REFERENCE_DIRECTION_655360.json/md` in currentrun.

Independentreviewapprovedconditionalreferencecoefficient.3to0design onlyafteractualbatchterminalresults,withmetadata-onlycopy/exactinitialpolicy/fullreferencepreservation,zero-upstreamnativequalification,actualsourcebound224exactbaseline,freshstrictconfirmation. No nextparent/run/reservationsselected. RecordcriticupdatesandKLearlystop-dependentactualoptimizercounts,notjustactormovement. Currentproductionunchanged;no newnativegames/learningforanalysis;no publication.

## Threat-prior first candidate complete

73500:5242880decisions/3791episodes/598.066305949slearning. Separate224screen16W180D28L,0/7passes;evaltime37.72182262095157s. Worse bystrictbranchrank than untrainedgatedcontrol13/195/16(0/7) andunchangedsource73000 10/205/9(2/7),andpreservedglobal61500(3/7). No rear damage/received-then-rear/streak. Gateactivefraction0.07688896223106913. Allraw/model/pool/frozen/READYandplanbindingsverified;actualAdam40960perparameter/epochcounter320.

|Opponent|Wins|Draws|Losses|Strict result|
|---|---:|---:|---:|---|
|s44701_step_002621440|2|27|3|loss|
|s45800_step_002621440|0|28|4|loss|
|s46201_step_005242880|3|24|5|loss|
|s46401_step_005242880|2|27|3|loss|
|s46500_step_005242880|3|25|4|loss|
|s47236_step_002621440|3|25|4|loss|
|s47238_step_010485760|3|24|5|loss|

Candidatearchived;untrainedgatedcontrolparentretainedforsecondseed73501,trainer1046265/controller1045643/session67408active. Preserveoriginal73000/global61500andall7foes. No confirmation/admission. Closedcampaign82excludesactivecandidate;activecompleted5242880decisions/3791episodesseparate,totalcompletedproduction411566080decisions/299501episodesincludingactive. Standard eval31360includes224newscreen. `FIRST_CANDIDATE_AUDIT.json` and `OPTIMIZATION_AUDIT_1.json` in currentrun. Nextreference0designremainsconditionaluntilwholebatchterminal;no publication.


## Threat-prior terminal and selected zero-reference follow-up

See `aircombat-rl/runs/seven_league_threat_prior_20261016/COMPLETED_REPORT.md` and `COMPLETED_AUDIT.json`. Two5242880decision candidates:16/180/28(0/7) and10/199/15(1/7),rear0both;no confirmation/admission. Closedcampaign84candidates/416808960decisions/303275episodes,standard evaluation31584. Actual73501 is source of a separate coefficient0.3to0 metadata-only copy;initial policy/critic/Adam/counters/export/full reference exact. Qualification and independent review pending;no upload.


## Zero-reference production active

`aircombat-rl/runs/seven_league_threat_refzero_20261016`:session17092/controller1048135/trainer1048141;inspectactualprocess. Two74000/1candidates each5242880decisions;referencecoefficient.3to0 only. Metadata-onlyactual73501copy/fullreference qualified;native64191exit0,196608decisions/122episodes/20.887643661s;baseline33458exit0,224exactnonwallrecords/37.362313559s. Finalreview approved;config unchanged. Observer66341log-path failure corrected only in separately versioned v2observer29510exit0,original frozen source/config/model preserved. Standard eval31808,closedcampaign84;no admission/publication.


### Offline diagnostics during zero-reference learning

`aircombat-rl/runs/seven_league_threat_refzero_20261016/GATE_TIMING.md` and `OFFLINE_DIAGNOSTICS_REVIEW.json`:all224consumed69000traces firstpublic-threatgate.95seconds/19ticks beforefirsthit,HP1.0atentry. Not current74000trajectories or causality. Early655360fixed-state probe moves farther fromreference inmeanKL(gated.065429to.070828),not performance evidence. Existing terminalterms are win2+.05ownHP/draw-.25/loss-1;damage/shaping/discount remain separate. Currentnative training remains unchanged;no upload.


### Zero-reference first candidate complete

74000:5242880decisions,3748episodes,581.238643s;224screen14W189D21L,0/7,rear0. Parent73501copy10/199/15,1/7 retained;global61500separate. Actor/critic each40960Adamupdates,320attemptedepochs;fullfixed-state probe271greedychanges(notperformance).74001activePID1049363/controller1048135/session17092. Closedcampaign84,activecompletedcandidate1separate;standard eval32032. Details `aircombat-rl/runs/seven_league_threat_refzero_20261016/FIRST_CANDIDATE_AUDIT.json`. Posthit gate coverage hypothesis remains unselected pending terminal batch. No confirmation/admission/publication.


## Zero-reference batch terminal

Actual17092exit0;74000=14/189/21,74001=11/197/16,both0/7/rear0. Batch10485760decisions/7547episodes/1145.131572s;672evalincludingbaseline. Campaign86/427294720decisions/310822episodes;standard eval32256. Source remains73501refzero1/7;global615003/7 preserved. See `aircombat-rl/runs/seven_league_threat_refzero_20261016/COMPLETED_REPORT.md`. NextselectedpublicWEZ OR ownHP<1 gate requires newversionedimplementation/state/export/nativequalification;no newrun/reservations orupload.


2026-10-16 WEZ OR damaged native qualification and production launch

Native smoke196608 decisions/127 episodes/20.972263876s completed exit0. Actual combined condition exercised; actor/critic1536Adamsteps each. Observer-only schema correction is documented in SMOKE_OBSERVER_SCHEMA_ADDENDUM.json; original frozen observer/config/runtime unchanged. Changed-function control224games/38.902450974s:16W189D19L,0/7,zero rear indicators. Original source10W199D15L,1/7 and global615003/7 remain distinct and preserved. Installation regression is explicitly retained. Independent reviewer approved74500/74501 bounded5242880decisions each; REVIEW.json binds exact config/control/smoke hashes. Guarded startup verified all frozenfiles/source/pool/conditions/26.57GBavailable. Live controller1051602/trainer1051605/session69240 verified. Strict screens, conditional fresh confirmation, append-only admission unchanged. Closedcampaign86;standard eval32480 includes newactivecontrol224. No publication.


> **Latest first result,2026-10-16:** WEZ OR damaged74500 completed5242880decisions/3708episodes/580.987s;separate224screen18W189D17L,0/7,38.640s,rear0. Authenticated model/READY/pool/rawcells/frozeninputs; actor+critic40960Adamupdates each. Better than changedcontrol16/189/19 but worse than originalsource1/7;global615003/7 preserved.74501ACTIVE controller1051602/trainer1052529/session69240. Closedcampaign86;activecompletedcandidate1separate;standardeval32704. No confirmation/admission/publication.



> **Latest terminal,2026-10-16:** WEZ OR damaged batch COMPLETE session69240exit0:74500=18W189D17L,74501=24W181D19L,both0/7/rear0. Authenticated10485760decisions/7368episodes/1158.576s,672evalinclcontrol. Retain74500withinbranch;original73501source1/7 and global615003/7 preserved. Closedcampaign88/437780480decisions/318190episodes;standard eval32928. No live nativejobs,confirmation,admission orpublication. Nextdesign not yet selected.



> **Latest implementation, 2026-10-16:** Terminal loss-aversion wrapper/worker and v13 added in new files. RED21 missing features then GREEN70; return telemetry refinement RED3 then GREEN73 related checks(1.75s). Actual retained74500 actor/critic/Adam/counters/full58001reference0/save-load exact; official loader7306consumedstates/7464sequential predictions+clockreset exact(maxprob5.03e-7). Additional loss2 only is planned; win/draw/native rules unchanged. Initial implementation/source review approved; telemetry addendum review pending before concrete preparation. No newrun/reservations/native games/learning. Closedcampaign88/437780480decisions;standard evaluation32928,pool7,noadmission/publication.



> **Latest prepared, 2026-10-16:** Loss-only utility v13 production/smoke configs frozen (2649/2650files), exact74500READY retained; no actor function change. Freshtrain2701500000/confirm3502120000/smoke2725500000reserved;seeds75000/1and74999. Only additionalterminal loss penalty0to2,total loss-1to-3.73relatedchecks/actualsource7306states/7464official predictions/implementationreview passed. Concrete pre-native review pending;native196608smoke/224exact74500baseline/finalreview mandatory beforeproduction. No live nativejobs/newlearning/games. Closedcampaign88/437780480decisions/318190episodes;standard eval32928,pool7,noadmission/publication.



> **Latest ACTIVE, 2026-10-16:** Terminal loss-aversion v13 started,controller1059034/trainer1059037/session72334;inspect actual process,neverduplicate. Two75000/75001×5242880decisions. Only additional terminal loss penalty0to2,totalterminal loss-1to-3;win/draw/gamma/lambda/rollout/LR/prior/reference0 held.73checks,actualsource/officialloaderqualification,native196608smoke124episodes/21.437s(actual20loss-2andMonitorreturnverified),224exact74500baseline and independentfinalreviewpassed. Exact74500READY parent,not smokechild. Strict224screens/conditionalfresh448confirmation/append-onlyadmission preserved. Closedcampaign88/437780480decisions/318190episodes;standard eval33152 includesactive224baseline. Original7/global61500preserved,pool7,noadmission/publication.



> **Latest first result,2026-10-16:** Terminal loss75000completed5242880decisions/3686episodes/577.751s;separate224screen28W174D22L,0/7,rear0. Worse than retained74500source18W189D17L;source/global61500preserved. Actual357traininglosses eachadditional-2,total-714andallMonitorreturnsverified;actor+critic40960Adamupdates each.75001ACTIVE controller1059034/trainer1059786/session72334. Closedcampaign88;activecompletedcandidate1separate;standard eval33376. No confirmation/admission/publication.



> **Latest terminal,2026-10-16:** Terminal loss batch COMPLETE session72334exit0:75000=28W174D22L,75001=21W181D22L,both0/7/rear0. Authenticated10485760decisions/7389episodes/1156.730s,672evalinclbaseline. Both worse than unchanged74500parent18W189D17L;74500/global61500preserved,pool7,noadmission. Actual712traininglosses eachadditional-2,total-1424andMonitorreturnsverified. Closedcampaign90/448266240decisions/325579episodes;standard eval33600. No live nativejobs/confirmation/publication. Nextdesign not yetselected.



> **Latest implementation,2026-10-16:** Early4kmpriorcondition core/export/metrics/worker/evaluator/v14 implemented in newfiles;old frozen sources unchanged. RED16core+16integrationmissingfeatures thenGREEN139relatedchecks5.68s. Actual74500source state/critic/Adam/counters/RNG/full58001ref0/roundtrip exact;7306consumedstates oldcombined4702/new5234/newly532,unaffectedfunctionexact/newlyteacherlogodds-3. Officialloader7306states/7464sequentialgreedy+sampled predictions/resetexact,maxprob4.86e-7. Policy/NumPy/rawmetrics sharefloat32boundary. Independentwholeimplementationreviewapproved31runtimefiles;concreteprep/nativequalification pending. Sourceconfigloss0,failedloss2notcarried. No newrun/reservations/nativegames/learning. Closedcampaign90/448266240decisions/325579episodes;standard eval33600,pool7,noadmission/publication.



> **Latest prepared,2026-10-16:** Early4km v14 changed-functionREADY/configs/smokeplan prepared,2653/2654frozenfiles. Actual74500originalweights/critic/Adam/full58001ref0 preserved;lowerprior6to3 newly532consumedstates only. Originalsource loss0config,failedloss2notcarried. Freshtrain2501500000/confirm3502160000/smoke2525500000,seeds75500/1+75499. Concretepre-native review pending;native196608smoke/224changedcontrol/conditionalownfresh448/finalreview required beforeproduction. No nativejobs,newlearning/games yet. Closedcampaign90/448266240decisions/325579episodes;standard eval33600,pool7,noadmission/publication.



Native early-range qualification completed: smoke session14477 exit0,196608 decisions,126 episodes,21.165767s,11564 newly active prehit decisions. Independent smoke review passed. Changed zero-step control session39324 exit0:224 games in42.571570s,25W175D24L,0/7 passes,rear damage0. Original unchanged source18W189D17L;installation alone increased both wins and losses. No confirmation or admission. Production awaits final independent review. Frozen win bonus1/draw penalty0.25 unchanged. No GitHub upload.


Early-range first candidate75500 completed5242880 decisions/3683episodes/595.981451s;separate224screen42.658219s,23W174D27L,0/7,rear0. Worse than changedcontrol25W175D24L and preservedsource18W189D17L. No confirmation/admission;candidate archived,changedcontrol retained. Second75501 trainer1063366 verifiedlive,controller1062183/session64149. Closed90;activecompleted1separate;standard34048. No GitHub publication. Evidence:aircombat-rl/runs/seven_league_early_range4000_20261016/FIRST_CANDIDATE_REPORT.md.


Early4km batch COMPLETE: actual session64149exit0,controller/trainers absent.75500=23W174D27L,75501=29W176D19L,both0/7,rear0. Total10485760decisions/7385episodes/1191.423493s;672independent evalgames inclchangedcontrol224.75501selected over changedcontrol25W175D24L,but remains below unchanged74500source18W189D17L and global61500(3/7). No confirmation/admission/upload. Closedcampaign92/458752000decisions/332964episodes;standard34272. Existing models/raws preserved. Further8km timing hypothesis has only consumed69000coverage evidence;no next native run selected or prepared. Evidence:aircombat-rl/runs/seven_league_early_range4000_20261016/COMPLETED_AUDIT.json.


Range8000 implementation/offline qualification COMPLETE:7newruntimefiles/5newtests;oldfrozen2653unchanged. RED53missingfeatures then192relatedPASS6.63s. Actual75501source7306states:4kmgate5234/8kmgate6130/newencodedstates896;4000functionexact,unchangedstatesexact/newteacherlogodds-3,critic/Adam/counters/RNG/full58001ref0/roundtrip exact. Officialloader7306states/7464sequentialgreedy+sample4900/4901/resetexact,maxprob4.81e-7. Wholeimplementationreviewapproved concrete preparation only;38runtimefiles explicit. Offline observer initial4000argument typo failedcorrectly and was preserved;v2corrected/provedexact. No config/reservations/nativegames/learning/admission/publication. Closed92/standard34272 unchanged.


## Direct piloting and evasion, 2026-10-16
Completed bounded manual/direct-controller development:1324native diagnostic games,zero PPO decisions,all actual sessions exited0. Final evasion study72variants/288developmentgames,then frozen224all-seven screen0W11D213L,0/7. Eleven no-hit games were all draws. Paired mean damage .9468to.5269 versus rear candidate,while wins26to0. Earlier isolated rear comeback required.7755HP damage;no light-damage maintained-rear success. Existing models/original7 preserved,no admission/freshconfirmation/publication. Full raw/hashed audit and per-opponent report:aircombat-rl/runs/direct_pilot_20261016/REPORT.md. Range8000smoke remains completed with observer pending;do not infer production running.


## Manual teacher 001 and fixed evaluation — 2026-10-16

The user corrected the workflow to actual agent-selected manual controls followed by a fixed224-game candidate evaluation. The pending784-game grid remains unlaunched. Manual session82614 exited0 after42commands and2401ticks, winning by health0.9835to0.9340 against44701. There was no rear attack. One public-clock replay of those commands completed exactly the original224-game screen (session7033exit0):7wins,30draws,187losses,0/7passes. The original manual terminal state reproduced exactly. New executions:225;neural training decisions:0. This rejects the fixed schedule as a qualified teacher. The next step is to pilot a failed condition and identify observation-based maneuver decisions. Goal remains active;no confirmation,admissionorpublication. Full verified report:aircombat-rl/runs/direct_pilot_20261016/manual_001_screen/REPORT.md.

## Manual side-attack continuation, 2026-10-16

manual_002 lost0/0.0925;manual_003 corrected the same consumed condition and won0.505/0 at54.50s. Exact replay prefix52.10s then three agent-selected continuation commands;2.50s unilateral WEZ,0.505HP dealt/0 received in that segment;initial received0.495 remains. Two new native games,noNN/no new224screen,totaldirect3119. Side attacks accepted;rear not mandatory. Strict224/fresh448 gates unchanged,all7goal unmet. Report:aircombat-rl/runs/direct_pilot_20261016/MANUAL_002_003_REPORT.md. No admission/publication.

## Opposite-seat manual win and future version records

manual_004 blue won0.505/0 at54.70s,2.60s unilateralWEZ,dealt0.538/received0 in that segment;initialreceived0.495. Same consumed IC3401500000/action4900 as red manual003. Session19906actualexit0;1nativegame,0NN,totaldirect3120. Strict224+unused448 gates unchanged and unmet. Userrequests first fully qualified modelv1,then preserved model-specific improvementsv2,v3 with lineage/artifact hashes/settings/results;no curriculum now and no version assigned yet. See docs/LEAGUE_STATE.json model_version_recording and aircombat-rl/runs/direct_pilot_20261016/manual_004/REPORT.md. No admission/publication.

## Next-opponent direct win, manual005

manual005 vs s45800 red won by final health0.604/0.043 at120.05s,not a kill. Sideattack3.05s unilateralWEZ,dealt0.693/received0;then egress preserved the earned lead. Session57049actualexit0;one diagnostic game,zeroNN/new224screen,totaldirect3121,6manualepisodes. Nextderive one public-observation controller and unchanged224evaluation;no variable grids/time-onlyreplay. All7goal unmet/no qualifiedv1/admission/publication. See manual005 report in aircombat-rl/runs/direct_pilot_20261016/manual_005/REPORT.md.

## Manual-derived observation feedback, fixed224screens

feedback000=160W6D58L,0/7,47.991s(session18093exit0). Exactreplay98589exit0 localized missed signed-lead transition. Signed-onlyfeedback001=189W4D31L,0/7,50.121s(session1758exit0);all160oldwins preserved,26losses+3draws becamewins. Exactremaining-loss replay18499exit0 localized egress_turn never reachinghold,not a terminal-hold failure. Two224screens+twoexactreplays=450newnativegames,zeroNN,totaldirect3571. All7qualification unmet;no unused448/qualifiedv1/admission/publication. Nextone positional-egress candidate with unchanged224screen. See aircombat-rl/runs/direct_pilot_20261016/feedback_001/REPORT.md.

## Positional egress improved;lessbraking rejected

feedback002=203W4D17L,1/7 strictpass(s47238=31W1D0L),52.165s,88589exit0. All189previouswins kept;14losses becamewins. Exact s46201mutual replay2369exit0;manual00657849exit0 extra braking+turnreversal failed0/0.472 after828exactprefixsteps. New520brake trialfeedback00361507exit0=202W1D21L,0/7,51.667s,so preserve203win002asworkingreference(brake482). This turn450native,zeroNN,totaldirect4021. Nextone merge_range240to600 earlierturn hypothesisfrom002,unchanged224screen. No qualifiedv1/fresh448/admission/publication. See aircombat-rl/runs/direct_pilot_20261016/feedback_003/REPORT.md.
