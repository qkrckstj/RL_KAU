# Sequential mutual-zero diagnosis — 2026-10-16

## Stage1: native termination

Replayed all224 authenticated s69000 screen games in45.158seconds; every original non-wall metric matched exactly. All197 draws reached zero health for both participants on the same final decision tick. Zero steps began with an already-dead participant. Combat.step computes both damage applications before the environment verdict. This is same-tick mutual depletion, not evidence of an extra post-death decision step; physical sub-tick simultaneity was not measured. Original rules are unchanged.

Native outcome labels were mutual212/kill6/died6; these differ from the user-authorized terminal-health-round4 outcomes16W/197D/11L. The native near-zero tolerance can label a surviving side mutual; our user verdict correctly retains its terminal-health advantage. No relabeling was performed.

## Stage2: offline reward decomposition

The224 compressed transition tapes were passed through the actual ControlGuidanceTrainingEnv without running additional simulations.20.191seconds; max per-step component reconstruction error2.23e-16, max discounted potential telescoping error7.46e-17. Independent review checked teacher state/call ordering and formulas; READY plan identity, full224keys and trace/source/script bindings subsequently authenticated in REWARD_BINDING_AUDIT.json.

|Outcome|Games|Mean undiscounted return|Mean discounted return|Mean terminal discount|
|---|---:|---:|---:|---:|
|win|16|0.743079|0.137205|0.238360|
|draw|197|-0.269133|-0.114059|0.288174|
|loss|11|-1.293287|-0.360311|0.232830|

Mutual-zero trajectories already have negative learning return; the damage/received undiscounted component is-0.25. Potential-shaped terms telescope to initial-state-dependent constants in these complete discounted trajectories. This does not imply no finite-training effect, no value-learning effect, or invariant effects of the other reward terms. These selected evaluation trajectories are not the actual training-rollout distribution, and do not establish why PPO retains this behavior.

## Stage3a: next comparison

Prepared nonwin sampling study from branch69000: allocation loss weight16->1 only, holding reward/architecture/reference/learning rate/budget fixed. Approximate effective foe exposure becomes13.97–14.55percent versus12.69–15.85percent for this baseline. Balanced exposure is an experimental hypothesis, not a solution claim. Fresh two-seed training and unchanged strict screen/conditional-confirmation protocol. No temporal representation or risk objective changes until this experiment is reviewed.

## Accounting and preservation

224new diagnostic native executions;0learning decisions;0additional simulations for reward analysis. These are separate from production screening/baseline totals. All original seven and previous candidates preserved. No unseen validation,admission or GitHub publication. Files:completion.json,reward_completion.json,REWARD_BINDING_AUDIT.json,inputs.json,reward_input_manifest.json,cells/,traces/,rewards/.
