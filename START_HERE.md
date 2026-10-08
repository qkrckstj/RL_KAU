# PAUSED by user — 2026-10-08

Do not start or resume any training, assessment, Goal, or recurring automation without a new explicit user request. No training/evaluation Python process remained when paused. Latest completed work is `league_compact_policy_drift_20261008`:32 consumed games exactly replayed,2,361 public states analyzed. The preceding113M rotated screen also completed960games/raw verified:rotated7801 win55.9896%,baseline7400=54.9479%,CEM6800=67.1875%;candidate gain+1.04%p,crossedCI[−5.21,+7.29]%p,no promotion.

Read [three-model recommendation and pause handoff](docs/MODEL_CHOICE_AND_PAUSE_20261008.md) first. Keep PPO7400 as fixed reference/fallback; preserve rotated7801 as the preferred research candidate; archive compact7800 for comparison. Do not continue all three learners. Original CEM and CEM6800 remain preserved comparators. No new teacher/reward/regularized learner was launched. The current user explicitly authorized this GitHub upload; future automatic uploads remain disabled.

On another computer, follow `REPRODUCE.md` and restore the archived evidence with `python scripts/restore_experiment_archive.py` from repository root before inspecting learner checkpoints. Then `python scripts/restore_experiment_archive.py --verify-only`. These commands never start training. Model and raw-result paths are listed in the recommendation. Archive runtime PIDs are historical, not live evidence. All earlier running/next-action labels below are historical.

# Previous: rotated candidate fresh inactive-opponent screen

Paired rotation completed2,621,440steps and960development games,exit0/raw and matched-roster verified. Identical initial tensor hashes and exact baseline episode records confirmed. Original compact7800/7801=59.375/54.375%;rotated=56.25/60.9375%;baseline58.125%. Rotated mean58.59375%,but neither branch passes mean-plus-tail extension guard. Retain rotated7801 as candidate,7800 retains source. New `aircombat-rl/runs/league_compact_rotated_assess_20261008`, entry `tools.league_compact_rotated_assess`, session56143:24current-inactive known archive foes,113M4ICs/bothseats/two action replicas,960games incl source7400/CEM6800;16workers12NumPy4ordinary. No automatic extension/promotion. This diagnostic follows observed development improvement despite tail regression;not a retroactive extension-gate pass. After raw_budget_screen verification inspect source differences/tails and decide next learning. Read COMPACT_LEAGUE report and LEAGUE_STATE. No duplicate/frozen edit/GitHub upload. Older live labels below historical.

# Previous: paired compact48 opponent rotation

First compact48 batch completed2,621,440steps,960development games,exit0/raw verified.111M baseline58.125%;7800=59.375%,7801=54.375% (retains baseline). Fresh112M inactive24 screen completed960games in82.16s:7800=58.8542%,baseline7400=63.2813%,CEM6800=61.4583%; paired uniform gain−4.4271%p,crossedCI[−8.33,−1.04]%p. No promotion or blind extension. Actual new run `aircombat-rl/runs/league_compact_rotated_train_20261008`, entry `tools.league_compact_rotated_train_v2`, session51250:keep40/replace8,still48active/27code groups/max4pergroup. Restore same7400 actor/critic/Adam,reuse7800/7801 and304/305M with separate paired-use reservations,111M development unchanged.1,310,720steps each,total2,621,440,8physical32virtual envs. Unique `rotated` export IDs.40,960-step v2 smoke passed. On completion run raw verifier already included,then compare both seeds to original compact controls at identical budgets and verify initial parameter hashes; assess111M active/inactive weakness before any more budget.112M now consumed for follow-up choice. Read `docs/COMPACT_LEAGUE_20261008.md` and `docs/LEAGUE_STATE.json`. Inspect actual processes,do not duplicate,modify frozen inputs,or upload GitHub. Earlier live labels below historical.

# Previous: compact48 training, full205 archive preserved

User authorized reducing active opponents. The205-foe refreshed run was stopped after preserving its1,310,720-step checkpoint; it is an interrupted experiment, not complete. New `aircombat-rl/runs/league_compact_ppo_train_20261008`, entry `tools.league_compact_ppo_train`, session62040, controller26476 at launch:48active opponents covering27code groups, maximum4/group, verified fixed-action-RNG duplicates collapsed. Source remains PPO7400 with actor/critic/Adam restored;8physical/32virtual environments. Two new7800/7801 streams,304/305M training,1,310,720steps each (2,621,440total), no automatic extension within this fixed run. Reuse111M development40foes; consumed development,not heldout.40,960-step actual smoke passed raw/settings/restore checks. Read `docs/COMPACT_LEAGUE_20261008.md` and `docs/LEAGUE_STATE.json`; inspect actual processes and immutable chunks/completion. Analyze both repeats before extra budget/roster rotation. Full archive and all old checkpoints preserved. No duplicate launch, frozen edits or GitHub upload. Older live labels below are historical.

# Previous: post-budget eight-condition diagnostic matches

Additional5,242,880PPOsteps completed exit0 with rate_ppo_raw_verified and1,440developmentgames.108Msource64.93%;7600first/final62.85/63.54 retains source7400;7601first/final65.97/65.28 retains1,310,720checkpoint. Equal-final mean64.41%,retained mean65.45%;not monotonic improvement. Actual current run aircombat-rl/runs/league_rate_budget_assess_20261008,PID17584/session95677,entry tools/league_rate_budget_assess.py.5,376games,110M8ICs/bothseats,48previously observed foes covering27code groups+8new checkpoint peers.16workers(14verifiedNumPy+2ordinary).Repeated-source role aliases reuse exact same games,not independent learners. Screen only,not full205-foe win estimate;all policies preserved. Read docs/RATE_PPO_20261008.md and LEAGUE_STATE. Do not duplicate live runs,modify frozen sources or upload GitHub. Older live labels below historical.

# Previous: additional budget with unchanged PPO settings

The15,760game109M assessment completed exit0,raw+paired comparisons verified.173archive win: source70.09%,control7401=70.27%,control7400=73.92%,low-rate7401=70.81%.7400 also improves observed group/tail scores; archive uniform pairedCI excludes0 but groupCI and novel-parameterCI cross0. No universal or promotion claim.109M and8parameter combinations are consumed after choosing7400 for continuation. Actual new run aircombat-rl/runs/league_rate_budget_train_20261008,session41385,entry tools/league_rate_budget_continue.py --role repeat. Exact existing173foes,sampling,PPO and reward settings retained; new7600/7601 streams,302/303Mtraining,reused108Mdevelopment,two1,310,720step chunks each,total5,242,880 additional steps.40,960step smoke exit0/raw/settings/Adam save-restore verified. Read docs/RATE_PPO_20261008.md and LEAGUE_STATE; inspect actual runtime/process before acting. Preserve all old policies/frozen code;no duplicate launch or GitHub upload. Older live labels below are historical.

# Previous: rate PPO fresh and untrained-parameter assessment

Learning completed10,485,760steps,exit0,raw+matched verified. Initialdev64.24%,controlfinalmean66.67%,low-ratefinalmean64.24%;low-rate method extension screen failed. Both controls and low-rate7401 retained. Actual next run aircombat-rl/runs/league_rate_ppo_assess_20261008,PID17440/session31787,16workers(14verifiedNumPy+2ordinary),15,760games,109M4ICs/bothseats.173trainingfoes+16newcheckpoint peers+8untrained parameter combinations from known tactical family;not unknown architectures/student entrants. Nominees fixed before outcomes. New panel2,048synthetic checks passed,existingloader signatures unchanged. Read docs/RATE_PPO_20261008.md and LEAGUE_STATE. No duplicate runs,frozen edits or GitHub upload. Older live labels below historical.

# Current: matched PPO learning-rate continuation

Assessment11072games completed,exit0,raw+resume hashes verified.157archive win:candidate75.12%,repeat73.69%,source73.85%,CEM75.88%;candidate-sourceCI crosses0,switchingunchanged,evader16draws,no promotion. Actual new learning aircombat-rl/runs/league_rate_ppo_train_20261008,PID22452/session18798,8physical/32virtual envs,173foes,learningrate1e-4vs2.5e-5,seeds7400/7401,two1,310,720step chunks each,total10,485,760,no third chunk. Sourcecontrol7300selected1.31M checkpoint;shared ancestor.300/301Mtraining,108Mdevelopment.81920-step smoke exit0/raw/rate/optimizer/savechecks passed. Read docs/RATE_PPO_20261008.md and LEAGUE_STATE. No duplicate runs,frozen edits or GitHub upload. Older live labels below historical.

# Resume update: current assessment

Original assessment controller8428 exited without terminal report; session84424 unavailable. Saved818batches(6544games) validated; remaining566batches resumed with unchanged frozen plan. Current PID32080/session27822 confirmed live. Entry tools/league_prioritized_assessment_resume.py; manifests preserve old runtime/worker identities/saved hashes. One old PID reused by conhost, verified unrelated and untouched. Read LEAGUE_STATE; no duplicate launch. Current completion timing covers resumed invocation only.

# Current: prioritized PPO fresh full-archive comparison

Training completed10,485,760steps,exit0,raw verified. Source dev67.71%;retained controls68.92%mean;both prioritized branches retain source. Final2.62M meanscontrol65.28%,prioritized66.32%,no extensions. Actual next run aircombat-rl/runs/league_prioritized_ppo_assess_20261008,PID8428/session84424,16workers(14verifiedNumPy+2ordinary),11,072games:157archive+16newcheckpoint peers,107M4ICs/bothseats. Compare development-selected controls7300/7301,sourcePPO,pureteacher,CEM6800. All original foes consumed;new peer models not unseen architectures. Read docs/PRIORITIZED_PPO_20261008.md and LEAGUE_STATE. Preserve all sources/checkpoints;no duplicate run or GitHub upload. Older live labels below historical.

# Current: prioritized-opponent PPO continuation

Risk-break CEM completed9,880games,exit0,raw verified;7200development56.25%vsbase55.99%,7201retainsbase.No method adoption;active0.29s candidate preserved as foe. Actual new run aircombat-rl/runs/league_prioritized_ppo_train_20261008,session63526,8physical/32virtual envs,157foes,control70/15/15vsprioritized32.5/15/52.5 effective uniform/group/weakness episode schedule.Linearvs squared smoothed failure weights;marked ancestor priors for unmeasured foes. Sourcecontrol7000 selected using consumed104Mscreen;actor/critic/Adam restored,shared ancestors not fresh initialization. Two seeds7300/7301,two1,310,720step chunks each,third only after nonregressing improvement.Base10,485,760/max15,728,640steps.298/299Mtraining,106Mdevelopment.81,920-step smoke exit0/raw/settings/sampling checks passed. Read docs/PRIORITIZED_PPO_20261008.md and LEAGUE_STATE. Never duplicate live run,modify frozen inputs or upload GitHub. Older live labels below historical.

# Current: defensive-break tactical CEM learning

Mixture screen completed3,840games,exit0,raw verified. Selected48foes:mixture53.52%,PPO7001/7000 55.21/56.38%,source54.17%;no mixture adoption. Mutual-kill/opening weaknesses remain. Actual new run aircombat-rl/runs/league_risk_break_train_20261008,session5168,8ordinaryworkers,two independent CEM searches7200/7201,4generations x10candidates,6defensivebreakparameters around preserved CEM6800. Full156archive retained,33training/48development foes. Training296/297M,development105M. Six tests and16actual disabled-gate baseline reproduction games passed. This is tactical parameter learning,not PPO. Raw analyzer tools/league_risk_break_analysis.py runs after completion. Read docs/RISK_BREAK_CEM_20261008.md and LEAGUE_STATE. Preserve all old policies/rules;no GitHub upload. Older live labels below historical.

# Current: episode mixture and PPO fresh screen

Discount pilots completed5,242,880steps,exit0,raw/matched verified. Control mean57.14%,gamma.9999mean54.11%,initial54.29%;long discount not adopted. Best control7001 retained,all other models preserved. Frozen60%pureteacher+40%CEM6800 episode mixture exported;9,600synthetic actions and80actual loader qualification games passed including24endpoint exact reproductions. Actual next run aircombat-rl/runs/league_mixture_screen_20261008,session27823,16workers(14verifiedNumPy+2ordinary),3840games across48stratified foes of154retained,new104M4ICs/bothseats. Compare actualmixture twoactionRNGs,both control PPOs,sourcePPO,teacher,CEM. Fullarchive not simulated here;switchingfoes alreadyconsumed. Component-weighted reference is separate from actualmixture outcomes. Read docs/MIXTURE_FEASIBILITY_20261008.md and LEAGUE_STATE. No GitHub upload/promotion. Older live labels below historical.

# Current: matched physical-step discount PPO learning

Expanded PPO comparison completed9,344games,exit0,raw verified. Existing134foes:6900/6901/sourcePPO win83.54/83.26/82.84%;8previously untrained switching rules30.47/39.84/20.31%,wide4IC uncertainty,no promotion. Evader16timeouts;extend-left/delayed-left each0W11mutual5loss for6900. Actual next run aircombat-rl/runs/league_discount_train_20261008,session14142,8physical/32virtual envs,146foes,two pairedRNG7000/7001,gamma.999vs.9999,5,242,880totalplannedsteps,onechunkperbranch/noautoextension. Source6900selected2,621,440 actor/critic/Adam;shared ancestor,not fresh independent initializations.103Mdevelopment,294/295Mtraining.81,920-step smoke passedactualreward/model/bufferdiscount and raw/save-restore checks. Read docs/DISCOUNT_HORIZON_AUDIT_20261008.md and LEAGUE_STATE. GitHub excluded. All older live labels below historical.

# Current: expanded PPO fresh comparison

Both continuations completed 7,864,320 additional steps, exit0 and expanded_ppo_raw_verified. Both selected step2,621,440; development win66.015625% /65.234375% versus initial61.71875%. Both third chunks regressed. Actual next run aircombat-rl/runs/league_expanded_ppo_assess_20261008, session74846,16workers(14 verified NumPy+2ordinary),9,344games on134archived+4newPPO peers+8untrained switching rules, fresh102M fourICs/both seats. Compare both PPOs, source PPO, pureteacher and CEM6800. Switching rules use known expert components and are consumed after this audit, not wholly unseen architectures. Read current LEAGUE_STATE and EXPANDED_PPO_CONTINUE_20261008 report. No GitHub upload or promotion. Below is historical.

# Current: expanded-league PPO continuation

INTERIM UPDATE: first6900 continuation completed3,932,160steps after automatic extension. Development win61.71875%initial ->65.234375%at1,310,720 ->66.015625%at2,621,440 ->62.890625%at3,932,160. Retain2,621,440checkpoint. Same PID23116/session54193 now runs6901; full batch raw verifier not yet run. New untrained switching-opponent panel frozen at `aircombat-rl/experiments/league/switching_audit_20261008/panel.json`,8rules over known frozen expert components,19,200synthetic checks,no match outcomes and not in training. Preserve for post-selection evaluation. Read current state before acting.

Broad CEM comparison completed10,720games, exit0, raw verified. CEM6800 archive win85.0379% vs PPO83.7121%, but PPO comparison CI crosses0, code-group win lower and opening-right2W1D13L; preserve both CEMs/PPO and do not promote. New actual PPO `aircombat-rl/runs/league_expanded_ppo_continue_20261008`, session54193,8physical/32virtual envs,134opponents including both new CEMs. Source entropy-zero PPO6700 step1310720 actor/critic/Adam; continuation RNG6900/6901 share this ancestor,not fresh independent initializations. Two1,310,720-step chunks each (base5,242,880 total), maxoneextra each after improvement/tail guard (max7,864,320). Fresh101M development,292M/293M training.40,960-step smoke and raw/optimizer/config checks passed. Also180game exact NumPy qualification passed for new PPO/CEM/reactive families; separate `execution_profile_extend_20261008.json` is for future comparisons, current training still8ordinary workers. Read `docs/EXPANDED_PPO_CONTINUE_20261008.md` and LEAGUE_STATE. No GitHub upload. Older live labels below are historical.

# Previous: broad tactical CEM vs PPO assessment

CEM learning completed7,120actual games,33unique parameter vectors per seed. Development win:6800=70.703125%,6801=66.796875%,baseline67.96875%; worst score improves both but mean/tail tradeoffs mixed. Training finished; original post-run verifier failed on float aggregation order (~2.22e-16), driver exit1. Original frozen sources untouched; separate `league_extend_parameter_analysis_v2.py` passed full raw/CEM/cache/selection verification with1e-12 aggregate-only tolerance, exit0. Actual next run `aircombat-rl/runs/league_extend_parameter_assess_20261007`, session20423,8workers(4pure+4ordinary),10,720games across132archived+2newCEM peers,8fresh100M ICs/both seats. Roles candidate=CEM6800,warm=CEM6801,early=PPO6700(two actions),teacher=pure extend. Deterministic policies evaluated once, generic paired metrics checked against4,800old games. Read `docs/EXTEND_TACTICAL_CEM_20261007.md` and LEAGUE_STATE. No final/unseen/promotion/GitHub claim. Prior live labels below are historical.

# Previous: tactical parameter CEM learning

Transferv2 completed7,392games, exit0, raw verified. Known120foes: PPO6700/6701 win86.71875/87.34375%, initial86.71875%, pure teacher85.520833%. New8reactive foes:53.125/48.4375%, initial47.65625%, teacher51.5625%; uncertainty wide and weak-tail improvement not consistent across learners. Preserve both PPOs; do not dismiss their code-group gains or claim proven universal improvement. Actual next run `aircombat-rl/runs/league_extend_parameter_train_20261007`, session50969,8ordinary workers: two independent CEM searches6800/6801 of7tactical parameters,4generations x10candidates, cached exact repeats.132archive foes retained;24training/32development,290M/291M training and99M development. This is CEM controller parameter learning, NOT more PPO.16real baseline-reproduction games passed after12,000synthetic-state equivalence checks; actual candidates evaluating. Read `docs/EXTEND_TACTICAL_CEM_20261007.md` and LEAGUE_STATE. Prior reactive audit is now consumed by learning. No GitHub upload. Older live labels are historical.

# Previous: fresh-condition transfer and reactive-opponent audit

Entropy-zero PPO completed2,621,440steps, exit0, raw and matched comparisons verified. Two-repeat mean78.7109375% versus matched entropy.005 controls76.5625%, but initial sampled policy79.296875%; no extension or promotion. Actual next run `aircombat-rl/runs/league_extend_transfer_v2_20261007`, session90822,8workers(4pure+4ordinary),7,392games:120archived foes+4learned peer variants+8new untrained reactive scripts,4fresh98M ICs/both seats. Roles candidate=entropy-zero final6700, warm=other final6701, early=untrained sampled prior6, teacher=pure extend-right. Last three named roles are comparators, NOT original PPO/CEM. Report archive/peer/reactive partitions separately. Initial v1 failed before any matches on a Torch import in NumPy worker; frozen failure preserved, correctedv2 imports validation only after pool exit and passed import/aggregation checks. Read `docs/EXTEND_TEACHER_PPO_20261007.md` and LEAGUE_STATE. Actual PID/closed matches must be checked. No GitHub upload.

# Previous: matched entropy-zero PPO follow-up

Fresh extend-teacher PPO completed 5,242,880 learning steps and 1,664 development games, exit0 and raw verified. prior6 final mean76.5625% versus initial79.296875%; prior8 mean77.5390625% versus initial76.953125%; pure teacher77.34375%. No branch met the mean-plus-tail extension guard, no global promotion. New actual run `aircombat-rl/runs/league_extend_entropy_train_20261007` uses prior6 with ent_coef0 instead of .005, two EXACTLY matched initial parameter tensors/seeds6700/6701, same120foes/sampler/training streams/development96M. Completed prior6 controls reused, not retrained. 20,480-step integration and16greedy-reproduction games passed; main base2,621,440steps/max5,242,880. Main session81581; inspect current runtime/PID before acting. Driver `tools/league_extend_entropy_train.py` runs raw analysis automatically after completion. Read `docs/EXTEND_TEACHER_PPO_20261007.md` and current LEAGUE_STATE. No GitHub upload. Older live labels below are historical.

# Previous: fresh PPO around the stronger extend teacher

UPDATE: teacher assessment completed4800games, exit0, raw verified. Across120foes: extend_right88.0208%, current PPO68.3333%, extend_left47.7083%, preserved CEM42.5%. Right extension also improves group-balanced and lower-tail scores. This is a scripted-controller result, not a neural learning gain. Actual next run `aircombat-rl/runs/league_extend_teacher_train_20261007`, PID29584/session42046,8physical/32virtual workers,120foes. Fresh independent neural seeds6700/6701, prior6vs8. Each gets1,310,720steps, automatic one-chunk extension only after beating both initial execution and pure teacher without lower-tail regression.32real greedy-reproduction games and40,960learning-step smoke_v2 passed, raw verified. Original smoke failed before learning on float32 rounding check and is preserved. Read `docs/EXTEND_TEACHER_PPO_20261007.md` and LEAGUE_STATE. Initial learners are `warm_starts[arm][seed]`, not the unused inherited single `warm_start`. No GitHub upload. Older live labels below are historical.

# Previous: stronger teacher candidate assessment

UPDATE: four duration PPO pilots completed5,238,673physical steps, exit0, raw verified. control6500/6501 win70.3125/61.328125%, hold5 variants58.984375/60.15625%, common baseline67.96875%. No method-level improvement/promotion. New15-role cross-play completed1160games, raw verified; archive expanded104to120 while retaining all old foes. Existing extend-right wins91.0714% equal-role average on this small panel, so assess a stronger starting teacher before further small PPO changes. Actual run `aircombat-rl/runs/league_teacher_candidate_assessment_20261007`, PID15344/session46783,8workers(4pure+4ordinary),4800games,95M conditions,120foes. IMPORTANT roles: candidate=current PPO, warm=extend_right, early=extend_left, teacher=original CEM; comparison signs are PPO minus comparator.100/600closed batches verified live. Read `docs/RECENT_LEAGUE_20261007.md` and LEAGUE_STATE. No GitHub upload. Older live labels below are historical.

# Previous: decision-duration PPO learning

UPDATE: portfolio transfer completed3328games, exit0, raw verified. Candidate65.6851% versus baseline65.5649%; group-balanced candidate worse, no promotion. Action-hold diagnostic208executions completed; immediate hold5/10 reduced sparse-panel wins. New actual PPO run `aircombat-rl/runs/league_hold_train_20261007`, PID20220/session9831,8physical/32virtual environments,104foes, two RNGs, control1-step versus hold5-step decisions. Actual40,960physical-step integration, five checks, raw smoke verification and eight exact real-flight comparisons passed. Max5,242,880physical steps across four pilots, report separately from learner decisions. Gamma/GAE scale with duration. Read `docs/ACTION_HOLD_PPO_20261007.md` and LEAGUE_STATE. Keep old checkpoints/frozen sources, no GitHub upload. Older live labels below are historical.

# Previous: portfolio transfer assessment

UPDATE: CEM portfolio learning completed, exit0, all5792 game executions raw verified. Development baseline63.671875%, search6400 candidate56.640625%, search6401 candidate68.75%; repeats mixed, no robust method-level claim. Selected6401 by development rank before new results. Actual next run `aircombat-rl/runs/league_portfolio_transfer_20261007`, PID28728/session8215,8ordinary workers,3328games across all104 archived foes on fresh92M ICs, both seats/two action RNGs.42/416batches verified closed while process live. Read `docs/PORTFOLIO_CEM_20261007.md` and LEAGUE_STATE. Action-hold diagnostic code prepared/tested but NOT launched. No GitHub upload or policy promotion. Older live labels below are historical.

# Previous: CEM portfolio gate learning

UPDATE:312 diagnostic games completed, exit0, `raw_geometry_analysis.json` verified. Baseline loses all16 extend-right games and draws all16 evader games without dealing damage; teacher wins6/8 against evader; both focal PPOs improve weave locally. New actual run `aircombat-rl/runs/league_portfolio_train_20261007`, PID24184/session71283,8ordinary workers. This trains a public-motion expert selector using CEM, NOT more PPO updates. Four experts and official physics/verdicts preserved. Two search RNGs,3generations x8candidates; separate90M development. Read `docs/PORTFOLIO_CEM_20261007.md`. Inspect actual process and completion/failure before resuming. Older live statements below are historical. No GitHub upload.

# Previous: focused weakness geometry diagnosis

UPDATE: focused-curriculum training completed4194304steps, exit0, raw verified. Archive mean65.234375%, focal61.1328125%, initial68.75%; all retain previous model. Focal weave gains do not remove extend-right losses. New diagnostic `aircombat-rl/runs/league_weakness_geometry_20261007`, PID12704/session63585,8workers,312games,89M conditions. Instrumentation matched all72 qualification game executions. No policy/physics/verdict modification; both focal seeds included. Read `docs/WEAKNESS_GEOMETRY_20261007.md`. Older live labels below are historical.

UPDATE: credit-horizon training completed6291456steps, exit0, raw analysis verified. All six pilots retained the prior model. Mean win: short56.25%, long_rollout67.1875%, long_credit62.5%, shared baseline71.09375%. Focused curriculum smoke completed65536steps, exit0, weights/Adam/config restore and raw checks passed. New main `aircombat-rl/runs/league_focal_curriculum_train_20261007`,8physical/32virtual envs,104foes, archive versus focal35, two RNGs,4194304steps planned. Read `docs/FOCAL_CURRICULUM_20261007.md` and LEAGUE_STATE. Preserve older runs. Older live labels below are historical. No GitHub upload.

UPDATE: 5600 fresh-condition matches completed, exit0, raw-transfer verification passed. Selected control67.3125%, history60.5%, common initial56.4375%; extend-right remains0/16losses. New training `aircombat-rl/runs/league_credit_horizon_train_20261007`, session29042,8physical/32virtual environments,104archived foes, six1M pilots. Actual three-arm98304step smoke passed settings/weights/Adam restore checks; raw analyzer passed. Treatments128/.99,512/.99,512/.997 are declared and actual model/buffer settings audited. Read `docs/CREDIT_HORIZON_20261007.md` and current LEAGUE_STATE before continuing. Older live labels below are historical. No final/heldout claim or GitHub upload.

UPDATE: extra-budget training completed4194304steps, exit0, raw_budget_analysis verified. Control mean64.84375%, history55.46875%; one control lineage improves, no history lineage updates retained best. Fresh-condition transfer assessment now runs at `aircombat-rl/runs/league_history_transfer_assessment_20261007`, PID10472/session77896,8workers,700batches/5600games,86M conditions. Nominees frozen by development rank. Roles candidate=current-state, warm=history, early=shared initial. Check actual process/completion first. No unseen/final/heldout claim or GitHub upload.

The configuration-matched comparison completed4194304steps, exit0, raw records verified. Mean win: control59.375%, history62.5%; opposite seed outcomes, no promotion. Actual continuation smoke completed65536steps, exit0. New main run: `aircombat-rl/runs/league_history_budget_extend_20261007`, driver `tools/league_history_budget_extend.py`, four extra1M branches,8physical/32virtual environments. Check actual PID/telemetry before resuming. Read `docs/HISTORY_BUDGET_20261007.md`. Older live PID statements below are historical.

`aircombat-rl/runs/league_history_config_matched_20261007` verified live PID32164/session82285. Control/RNG6000 completed1M; history/RNG6000 reached1048576steps. New RNG6000/6001, new training276M/277M, development85M. IMPORTANT: previous `league_history_matched_20261007` completed but its observation-only interpretation is INVALID: history prototype omitted PPO settings and used different gamma/GAE/epochs/entropy/clip/KL. Preserve that confounded experiment and all models; see its treatment_config_audit.json. Corrected prototype now preserves complete PPO configuration and runtime guard rejects the bad one. Read docs/LEAGUE_STATE.json and docs/HISTORY_PPO_20261007.md. Never duplicate live runs, modify frozen inputs/sources, or upload to GitHub without explicit request. Final/heldout remain unopened. Everything below is historical.

---

# Current: matched current-state versus history PPO learning

`aircombat-rl/runs/league_history_matched_20261007` verified live, PID32376/session54064. First control pilot completed1,048,576steps; history/RNG5900 reached656480actual steps. Four pilots total4,194,304new steps,8physical workers/32environments/100opponents. Read `docs/HISTORY_PPO_20261007.md` and `docs/LEAGUE_STATE.json`. History clone, causal feature tests, and65,536-step integration passed; initial flight records matched. Frozen old files remain unchanged; new controller uses declared adapters. Transfer assessment completed4,800games but improvement remained mixed and extend-right0/16. No final/heldout/promotion/GitHub upload. Inspect actual process before acting; never duplicate a live run or modify frozen sources/inputs. Everything below is historical.

---

# Current: fresh-condition control transfer assessment

`aircombat-rl/runs/league_control_transfer_assessment_20261007` verified live, PID30708/session48819, 99/600closed batches, 4,800scheduled games. This is EVALUATION, not training. Both extra-budget control paths completed with exit0 and raw verification; neither surpassed its retained development best. No further blind same-setting extension. Candidate fixed before fresh83M outcomes. Read `docs/LEAGUE_STATE.json` and `docs/MATCHED_PPO_CURRENT.md`. Preserve all models/frozen inputs; no GitHub upload; no final/heldout opened. Everything below is historical.

---

# Current: matched control extra-budget probe

`aircombat-rl/runs/league_sampled_control_extend_20261007` was verified live: PID11256, session78206, RNG5700, 139296actual newsteps. Two pilot control lineages each receive1,048,576extra steps; previous best policies remain protected. Read `docs/LEAGUE_STATE.json` and `docs/MATCHED_PPO_CURRENT.md`. The previous six pilots completed with exit0 and raw-game verification; their small reused development improvements were mixed. No promotion, final, or heldout test. Do not duplicate live processes, edit frozen sources, or publish to GitHub without explicit request. Everything below is historical.

---

# Current experiment: matched sampled PPO strategy pilots

Read [the current experiment report](docs/MATCHED_PPO_CURRENT.md) and `docs/LEAGUE_STATE.json` first. Controller PID9084/session98077 was verified live; first control/RNG5300 reached180,928steps. Inspect actual process and telemetry before acting. Do not duplicate the live run. Prior entries below are historical; some older text has encoding damage. GitHub publication remains excluded unless explicitly requested.

---

# ?? ?? ?? ? 2026-10-07

? ?? `aircombat-rl/runs/league_sampled_strategy_20261007` ??? ????. PID 9084, ?? ?? 98077. ? ?? control/RNG5300 ?? ??? 32 ? 123,168??? ?????, 77???? ??? ????. ?? ?? ??? ?? ?? launch_verified.json. ?? ??? telemetry? ?? ????? ??? ?. ? 3???2?? RNG?1,048,576?? = 6,291,456??. ??? 8?, ?? ?? 32?, ?? 100?. ?? ??/?? FairFight/CEM ??. GitHub ??? ??(??? ?? ??).

?? ?? `league_sampled_strategy_smoke_20261007`? ?? ?? 0, ?? 98,304??, 3?? ??/??? ??? ?? ??? ????. ? ?? ?? `league_sampled_fresh_assessment_20261007`? 9,408?? ????. ??? 8M ?? ???? ??? ?? 2M ?? ???? ?? ?? ???? ???. ?? ?? ??? 32? ???? ??? ???. ??? ?? ??? `docs/SAMPLED_STRATEGY_20261007.md` ??.

?? ??? ?? ??? ?? ????. ? PID? ?? ? ??? ?? ??? ???? ? ?. ? ??? ?? ??? ?? ???? ?? frozen ??/??? ???? ? ?. ?? ? progress.json? ???? ?? ? ?.

---

# 새 작업 세션은 여기서 시작

**2026-10-07 14:51 KST ? ?? ?? ???? ?? ?? ?? ?:** CPU ?? ????31308/??88518? ????0?? ???. ? ?? ??? ??2,097,152??? ??? ?????, ??? ??? ??216?? **128?32?56?**? ????. ?? ???? ?? ??1,048,576?? ?? **152?26?38?**? ????. ? ??4600?2,097,152?? ??152?26?38?? ????. ? ??? ?? ? ???? ??/?? ?? ??? ??? ?? ???? ? ??? ????. `runs/league_sampled_ppo_finish_cpu_20261007/completion_analysis.json` ??. ?? ?? ?? ??? ??? ???? ? ??? ?? ? RNG/??? ?? ?? ????.

?? ?? **PID15068/??58228**? `runs/league_sampled_fresh_assessment_20261007`? **???8?(NumPy6???2)**? ? ???? ??? ????. ??96??? ?? ??? ??? ??2?, ?8????/? ??, ? **9,408??(588??)**?. ?? ?? ???199?. ???? ??? ??2????? ??8M ??2???????2M ??? ???CEM ?? ??? ?? ???? ????. ??? ?? ??? ?? ??? ??/? ?? ??? ??? ???? ???? ?? ????/?????? ?? ??? ??.

?? ?? ??? `runs/league_sampled_numpy_qualification_20261007`?? ????0?112?? ?? ?? ? ??? ?? ?? ?? ??? ????. ?4???? NumPy ??? **?? ? ?????** ???? ?? execution_profile? ????. DQN? ?? ????. 15? ?? ?? ??? ??CPU55.48%??? ??35.80%, ?? ??3.64GiB?? ???8?/?? ??? Torch ???? ????. CPU ?? ??? ?? ??? ???? ???.

?? Python? `aircombat-rl/.venv-cpu/Scripts/python.exe`; ??? ????3??1? ????. ?? progress.json? ?? ?? ??58228??? matches/match_*.json??? ????? ????. ?? ?? ? ??? ??? ??? ?? ?? ??? ?? ????. ?? ????? ? ???GitHub ?? ??? ??. **?? ??/?? ??? ?? ????.**

**2026-10-07 14:28 KST ? Goal ???CPU ?? ? ?? ??:** ?? PID31308/??88518? `aircombat-rl/runs/league_sampled_ppo_finish_cpu_20261007`?? ? ?????? ???96??? ????. ??? ?? 86,024???57????. ???4601 ??? ?? ??524,288??? ???? ?? ?? ?? ??? ???? ?? ????. ???4600? ????? ???.

??? `aircombat-rl/.venv-cpu/Scripts/python.exe`, PyTorch2.14.1+cpu?JSBSim1.3.0?SB3 2.9.0??. ?? ??? ?128? ?? ?? ??? ?? ??? ??? ????. ??? ??? ?61MiB/????? ???. ?? ??32,768?? ??? ????0?learner/optimizer ?? ????? ?? ??1.960GiB? ????. `runs/cpu_environment_qualification_20261007.json` ??. ? ???? ?? ??? ??2.4GiB/??? ?2.0GiB? ??? ???? ?? ?1.8GiB/??1.5GiB??? ?? ??/?? ?? ??? ????. ?? ?? ??? ????.

?? ??266M???4701? ?? ?? ??????? ? ??? ????. ??? ?? ??/RNG ??? ???? ?? ??? ???. ?? ?? ?? ? ?? ??? ??? ? ? ?? ??? ????. ?? ?? ?????? ?? ???? ???. GitHub ?? ??? ??. ?? ??? ?? ????.

**2026-10-07 13:45 KST 현재 상태 정정: 학습·시뮬레이션·시작 대기 프로세스 모두 없음.** 13:08의 복구 학습 PID25168은 두 번째 실행 1,572,872 상호작용에서 메모리 가드로 종료코드1이었다. 첫 실행4600과 두 번째 실행의 비상 learner/optimizer는 보존됐다. 남은 기본 학습량은 **524,288단계**이며 첫 실행을 다시 학습하면 안 된다.

한 작업자·여덟 환경 및 메모리 회복 대기를 구현한 `tools/league_sampled_ppo_finish.py`는 실제32,768단계 통합 점검을 종료코드0으로 통과했다. 그러나 본 실행48360은 학습 시작 전 커밋 여유 부족으로 종료했고, 별도 시작 대기47932도 **13:31:59 KST에300초 시간 초과·종료코드1**로 끝났다. 현재 메모리 회복을 자동으로 기다리는 프로세스도 없다. 실패한264M/265M 훈련 조건은 예약됐지만 사용되지 않았으며, 기존 폴더와 고정 조건 예약이 있으므로 같은 명령을 그대로 재실행하지 말 것. 메모리 회복 후 새 시도 계획으로 검증된 실행기를 재사용한다. 사용자 앱·서비스·OS 설정은 변경하지 않았다.

현재 자원 측정과 종료 확인: `aircombat-rl/runs/resource_status_20261007_1345.json`. 과거4작업자 학습 표본은 전체CPU26.73%/학습15.38%,2작업자는15.54%/10.03%였으므로 CPU 전체 사용 상태가 아니었다. 묶음 실행 비교의 처리량1.4135배 개선은 이미 측정됐지만 전원 상태와 반복 편차 한계가 있다. 현재는 물리 RAM 여유와 별개인 시스템 커밋 여유가 시작 조건4.5GiB에 못 미쳐, 작업자 수를 올리는 단계가 아니다.

후속 NumPy 실행 검사는 **아직 미실행**이다. 아래 역사 기록에 있는 원래 복구 실행을 `--source`로 삼는 명령은 현재 무효다. 두 분기 결과를 합쳐 정상 완료한 **향후 새 remainder 실행 폴더**만 소스로 사용할 것. 기존 모델·공식 물리·최종시험 조건을 보존하며 GitHub 자동 업로드는 계속 제외한다. **아래 시간별 항목은 역사 기록이며 실행 중이라는 문구를 현재 상태로 해석하지 말 것.**

**2026-10-07 13:06 KST 두 번째 실행도 개발 개선:** 실제 PID25168/세션21894는 RNG4601 두 번째 구간을 학습한다. 첫 추가1,048,576단계·810훈련경기·96상대 경험 후 **152승26무38패/216(70.37%)**로 시작138승보다 개선됐다. 두 행동 난수의 승수는78/108·74/108이며 개발 순위는 `[.720775,.339286,0,-3]`이다. 첫 실행2M의152승과 총합은 같지만 상대별 결과가 다르고, 첫 실행2M의 그룹 균등·하위 상대 순위가 더 높다. 이번 시드에서는1M 후보를 보존하며,2M 결과로 추가 연장을 판단한다. `aircombat-rl/runs/league_sampled_ppo_recovery_20261007/s4601/checkpoint_audit_001.json` 참조. 두 실행의 개선은 공통 시작점/재사용 개발 조건에 한정되며 독립 최종 검증이 아니다. 우측 고속 이탈에는 여전히0승0무8패다.

후속 실행 검사 도구 **준비됨, 실제 대결 검사는 아직 미실행:** `aircombat-rl/tools/league_sampled_numpy_qualify.py`. 두 학습이 정상 종료되고 컨트롤러 PID가 사라진 뒤, 스레드 환경변수3개를1로 설정하고 `python -m tools.league_sampled_numpy_qualify --source runs/league_sampled_ppo_recovery_20261007 --out runs/league_sampled_numpy_qualification_20261007`로 실행한다. 같은56경기를 일반2작업자와 분리4작업자에서 각각 실행해112경기의 공식 기록·선택된 전체 피해 추적 일치 및 순수 작업자 Torch 미수입을 요구한다. 새 성능 시험/속도 벤치마크가 아니다. `runs/sampled_numpy_qualify_checks_20261007.json`은 문법·기존 실제 기록54개 검사·중복 좌석/추적 거부·실제 살아 있는 학습을 감지한 시작 차단을 확인했다. **새 소스 승인은 아직 없고 execution_profile은 변경하지 않았다.** 현재 도구 소스 해시7a214c4d4f2c28fc92a22860c4ee9e6f8dfa7e384e97b225541e38ca4272dc43.

**2026-10-07 12:56 KST 첫 복구 실행 완료·두 번째 실행 시작:** 실제 **PID25168 / 세션21894**는 계속 살아 있으며 **RNG4601**을 학습한다. 새 작업자는 **PID40800/49780**이다. RNG4600은 추가 **4,194,304단계·3,047훈련경기·96상대 전부 경험**을 완료했다. 네 체크포인트의 동일 개발216경기 승수는 **134 → 152 → 127 → 128**로, **2,097,152단계(152승26무38패,70.37%)**를 보존한다. 마지막420만 단계는128승38무50패다. `aircombat-rl/runs/league_sampled_ppo_recovery_20261007/s4600/result.json`과 `checkpoint_audit_004.json`에서 완료·선택을 확인했다. 첫 실행의 학습 가드 최소 커밋 여유2.307GiB였고 중단은 재발하지 않았다. **전체 컨트롤러는 아직 완료가 아니다.**

RNG4601은 첫 실행의 최선/마지막 가중치가 아니라, **동일한 원래 비상 learner/optimizer**에서 새 난수와 훈련 조건263백만으로 시작한다. 우선2,097,152단계, 기존 개선 기준을 만족할 때4,194,304까지 연장한다. 이는 공통 시작점 이후의 별도 실행이며 처음부터 독립 학습한 결과가 아니다. 결과가 나온 뒤 새 조건으로 기존 CEM·이전 PPO와 비교한다. 미사용 최종시험/홀드아웃은 계속 보존하며 GitHub에는 올리지 않는다.

후속 평가 가속 준비: `aircombat-rl/runs/sampled_numpy_source_review_20261007.json`에 새 NumPy 정책4소스군의 검토를 기록했다. **아직 순수 NumPy 작업자 사용을 승인한 것이 아니며 현재 실행/기존 execution_profile도 바꾸지 않았다.** 두 학습이 끝난 다음 동일 공식 대결·양 좌석·선택된 전체 피해 추적에서 기존 로더와 일치하고 순수 작업자가 Torch를 불러오지 않는지 확인해야 한다. DQN 두 모델은 일반 작업자로 유지한다. 학습 중 별도 시뮬레이션 풀을 띄우지 말 것.

**2026-10-07 12:48 KST 세 번째 구간 후퇴·최고 체크포인트 유지:** PID25168/세션21894가 RNG4600의 네 번째 구간을 계속 학습한다. 추가3,145,728단계 후보는 개발216경기에서 **127승38무51패(58.80%)**로 내려갔다. **152승을 기록한2,097,152단계 모델을 그대로 보존**한다. `aircombat-rl/runs/league_sampled_ppo_recovery_20261007/s4600/checkpoint_audit_003.json`은 완료된 세 구간의 원시 승패·조건 쌍·선택 순위를 재계산했다. 약속된4,194,304단계 후 첫 실행을 마치고, 같은 비상 시작 가중치에서 RNG4601을 실행한다. 추가 연장을 새로 승인하거나 최신 모델로 최고 모델을 덮어쓴 것이 아니다. 아직 새로운 최종 평가나 홀드아웃은 실행하지 않았다.

**2026-10-07 12:40 KST 복구 학습 개선·자동 연장:** 실제 PID25168/세션21894는 RNG4600 세 번째 구간을 학습한다. 추가 **2,097,152단계·1,506훈련경기·96상대 모두 경험** 후, 같은 개발 평가216경기에서 시작 모델 **138승25무53패(63.89%) → 후보152승26무38패(70.37%)**로 개선됐다. 두 행동 난수를 같은 비중으로 합친 개발 순위는 `[.644965,.267857,0,-5] → [.742911,.375,0,-3]`이다. 사전 연장 기준을 충족해 **4,194,304단계까지 자동 연장**했다. 첫105만 단계134승 후보는 기준을 넘지 못했으며, 현재210만 단계 후보를 보존한다. 완료된 두 구간의 평가는 학습+체크포인트 평가 시간의14.75%다. `aircombat-rl/runs/league_sampled_ppo_recovery_20261007/s4600/budget_analysis.json`에서 원시 승패·동일 조건 쌍·순위·연장 결정을 재계산했다.

**아직 남은 약점:** 우측 고속 이탈0승0무8패, 우측 직조 기동0승3무5패, evader0승8무0패다. 재사용한 개발2조건·양 좌석·행동난수2개의 결과이므로 최종 개선이나 독립 반복 성공으로 취급하지 않는다. RNG4601은 첫 실행이 끝난 후 동일한 비상 시작 가중치에서 별도 난수/훈련 조건으로 실행한다. 두 시작 모델과 모든 체크포인트를 보존하며 최종시험·홀드아웃은 열지 않았다. `optimizer_diagnostic_001.json`은 기존 로그에 수치 오류가 없고 업데이트가 진행됨을 확인했지만 승률 변화의 원인을 증명하지 않는다. GitHub 업로드 없음.

**2026-10-07 12:23 KST 메모리 제한 복구 실행:** 실제 **PID25168 / 실행 세션21894**가 `aircombat-rl/runs/league_sampled_ppo_recovery_20261007`에서 학습 중이다. **2작업자·8환경·96상대**, RNG4600/4601 순차 실행, 환경당 512단계로 전체 PPO 롤아웃 4,096개를 유지한다. 표준출력에서 60.1초에 147,464단계·104경기를 확인했다. 직전 4작업자 실행29468/3068은 **12:02 메모리 안전 기준으로 종료코드1**이었고 이전 자식들도 종료됐다. 기존 실행을 다시 켜거나 완료로 취급하지 말 것. `runs/league_sampled_ppo_continue_20261007/termination_analysis.json` 참조.

저장된 **1,196,048단계 비상 learner/optimizer**에서 새 난수·새 훈련 조건262/263백만으로 시작한다. 마지막16개 미완료 롤아웃 전이는 버리며 중간 JSBSim 상태·비상 RNG를 복원한 정확한 궤적 재개는 아니다. 과거 공통 경로까지 포함한 10,616,856 상호작용은 두 실행에 중복 합산하지 않는다. 이전의 더 나은8M 개발 기준 모델도 보존한다. 시드당 우선 2,097,152단계, 사전 개발 개선 규칙을 만족할 때 4,194,304까지 연장한다. 최종시험·홀드아웃은 아직 열지 않는다.

복구 통합 점검 `runs/league_sampled_ppo_recovery_smoke_20261007`은 **32,768 실제 학습 단계·가중치/옵티마이저 복원 확인 후 종료코드0**이었다. 96상대 캐시, 3개 점검 상대×2경기×2행동난수×전후=24평가경기; 학습13.89초, 학습 가드 측정 최소 커밋 여유3.219GiB. 짧은 실행 점검이며 본 학습의 안정성·품질을 보장하지 않는다. 실행 중 가드는 커밋1.8GiB/물리1.5GiB로 유지하고 정상 보고마다 부모·작업자 메모리를 추가 기록한다.

12:23:12 KST의 15초 표본에서 전체CPU15.54%, 학습 프로세스 합계10.03%(16논리코어 기준), 커밋 여유3.14GiB·물리 메모리8.96GiB, 전원 연결·배터리100%였다. `resource_observation_001.json` 참조. 현재 CPU 전체를 활용하는 상태가 아니며, 메모리와 실행 구조의 제약이 남는다. 과거 묶음 실행 비교의 처리량1.41배 개선과 이번2작업자 속도를 직접 통제 비교로 취급하지 않는다.

`progress.json` 대신 세션21894·`telemetry.jsonl`·`process_memory.jsonl`·완료된 `chunk_*.json`을 확인한다. **새 실행의 소스/입력은 동결됐으므로 수정하지 말 것. GitHub 업로드 제외.** 아래 이전 실행 상태는 역사 기록이며 이 항목이 최신이다.


**2026-10-07 11:55 KST 추가 PPO 학습 시작·첫 구간 완료:** 현재 실제 **PID29468 /실행세션3068**이 `aircombat-rl/runs/league_sampled_ppo_continue_20261007`에서 학습한다. **4작업자·16환경·96상대**, 학습난수4500→4501을 순서대로 실행한다. 첫 난수의 추가1,048,576단계·765경기가 끝났고96상대를 모두 만났다. 학습297.77초(**3,521단계/초**), 개발 평가216경기57.41초였다. 같은27상대·두 행동난수에서 시작 모델은 **135승·25무·56패(62.50%)**, 새 첫 후보는 **126승·32무·58패(58.33%)**라 시작 모델을 보존하고 두 번째 구간을 진행 중이다. `s4500/first_chunk_analysis.json`·`chunk_01.json` 참조. 개발 평가는 이번 학습+평가 벽시계의16.16%이며, 사용한 개발 조건의 중간 결과이지 최종 개선 판정이 아니다.

이번 시작 가중치는 직전 비교에서 고른 **RNG3300의8,388,608단계 learner/optimizer**다. 이전 공통 전반부까지 포함한 경로의 상호작용은9,420,808회다. 이미 만들어 둔 체크포인트를 확률적으로 재평가했더니 첫 경로의2M/4M/6M/8M 개발 승률은 **42.19%→50.52%→55.73%→64.58%**였다(두 행동난수4900/4901 평균, 각192경기). 두 번째 경로의8M은36.98%로 낮았다. 따라서 첫 경로는 탐욕적 평가와 달리 개선 추세가 있었지만, 반복 간 차이와 작은 재사용 개발 표본의 선택 편향은 남는다. `aircombat-rl/runs/league_residual_sampled_development_20261007/warm_start_choice.json` 참조. 해당 선별PID22844/세션72165는 종료코드0으로 닫혔다.

**학습 규칙:** 기존72상대와8개 탐욕적·16개 확률적PPO 스냅샷을 보존해96상대를 구성했다. 기존PPO 학습률/보상/공식 물리는 유지하고, 묶음 실행의환경당256단계·전체4,096개 롤아웃을 사용한다. 난수당 우선2,097,152단계, 첫→두 번째 구간에서 보존한 개발 순위가 충분히 좋아지고 시작점을 넘으면4,194,304단계까지 자동 연장한다. 개발 상대는기존24개+시작 모델 행동복제2개+보존한2M 탐욕 모델로27개다. 초기192경기를 재사용하고 새 상대24경기만 추가했다. 매 구간은 두 행동난수를 같은 비중으로 평가한다. 현재 실행기는 최종시험/홀드아웃을 열거나 모델을 승격하지 않는다. 완료 결과를 보고 보존 기준 및2M 탐욕 모델과의 별도 새 조건 비교를 결정한다. 이전 기준과 실패 기록을 덮어쓰지 않는다.

실제32,768단계 통합 점검은96상대 캐시·평가 중 환경 상태 보존·학습/옵티마이저 정확 복원을 확인하고 종료코드0이었다(`runs/league_sampled_ppo_smoke_20261007/completion.json`, 소프트웨어 검사2개). **현재 실행기와 입력은 동결 상태이므로 수정하지 말 것.** 활성`progress.json` 대신 세션3068의표준출력·추가 기록형`telemetry.jsonl`·닫힌`chunk_*.json`을 본다. 11:57:46 자원 측정에서는 충전기 연결(ACLineStatus1)·배터리100%, 커밋 여유약3.15GiB·물리메모리8.07GiB였다. 전체CPU26.73%·학습15.38%는15초 표본이며 여전히 전체CPU 포화라는 뜻은 아니다(`resource_observation_001.json`). GitHub 업로드는 제외한다.

**2026-10-07 11:35 KST 묶음 실행 비교 완료·재학습 시작점 선별 실행:** PID14004/세션4106은 종료코드0으로 끝났다. 같은4환경의 실제65,536단계 궤적과 최종 가중치가 기존 실행과 일치했고, 설정별 반복도 일치했다. 4작업자·4환경 중앙48.38초 대비4작업자·16환경34.23초로 이번 비교의 처리량은 **1.4135배**, 시간은약29.25% 감소했다. 묶음 두 반복 모두 대응 기본 실행보다 빨랐고 커밋 여유 최소3.148GiB·물리 메모리7.506GiB로 사전 실행 후보 기준을 통과했다. **반복 속도 편차가 크며, 끝 직후 Windows는ACLineStatus0(전원 분리)·배터리97%였다.** 시행별 전원 상태를 기록하지 않았으므로1.41배를 안정적인 고정 배속이나 전원 조건을 통제한 인과 효과로 주장하지 않는다. `aircombat-rl/runs/league_grouped_ppo_benchmark_20261007/analysis.json`·`runs/power_cpu_sets_20261007_1134.json` 참조. 모든 후보와 기존 실행을 보존하고 다음 학습의 실행 후보로만 사용한다.

지금은 **실행세션72165**에서 `aircombat-rl/runs/league_residual_sampled_development_20261007`이 기존8개 PPO 체크포인트를 확률적 행동으로 비교한다. **이미 사용한 개발66000000 조건·24상대·4경기·행동난수4900/4901**을 모든 후보에 동일하게 적용하며 총1,536경기다. 두 행동난수를 같은 비중으로 합쳐 다음 학습의 시작 가중치만 고른다. 새 최종시험/홀드아웃을 열거나 모델을 승격하지 않는다. 실행기가 끝나면 `warm_start_choice.json`을 확인하고 **그 가중치의 별도 후속 PPO 학습**으로 이어갈 것. 단일 행동난수 최고값을 골라서는 안 된다. 이미 확인한 행동 방식 문제를 반영해 후속 학습의 개발 평가도 확률적 실행과 맞추며, 보존한2M 탐욕 모델·CEM 기준과의 비교는 유지한다. 속도 측정을 다시 반복하거나 또 다른 광범위한 선별을 먼저 추가하지 말 것. GitHub 자동 업로드 금지는 유지한다.

**2026-10-07 11:23 KST 행동 방식 진단·PPO 계측 완료:** 실제30816·48620은 모두 종료코드0이며 관찰세션72634도 닫혔다. 행동 진단의 새24상대/16조건/양 좌석에서 2M 탐욕적 모델은 **493승·142무·133패/768경기(64.19%)**, 같은6M 가중치의 탐욕적 모델은 **83승·420무·265패(10.81%)**였다.6M 확률적 행동4난수는 합계 **1,698승/3,072경기(55.27%)**, 같은6M 탐욕 대비 **+44.47%p**, 조건/행동난수 교차 부트스트랩95% 구간 **[+39.55,+49.48]%p**다. 코드그룹 균등 차이도+48.99%p로 사전 진단 기준을 통과했다. 다만 확률적6M도 앞선2M 탐욕 모델보다 **8.92%p 낮고**, 우측 고속 회피는8승/128경기에 그친다. **행동 방식 불일치는 큰 원인이지만 장기 학습의 모든 퇴행을 설명하지 않는다.** `aircombat-rl/runs/league_residual_action_mode_20261007/analysis.json`·`completion.json` 참조.4행동난수는 독립 학습4회가 아니며 모델 승격은 없다.

PPO 계측4회에서 실제 학습 궤적/초기·최종 가중치 해시가 일치했다.65,536단계 중앙26.06초(**2,515단계/초**), 계측 추가 시간 약1.29%다. 벽시계 기준 환경 결과 대기·수신·쌓기 약61.2%, 행동/가치 계산20.4%, 전송5.5%, 옵티마이저 갱신4.4%였다. **대기에는 남은 시뮬레이션도 포함되어 순수 통신 비용이 아니다.** `aircombat-rl/runs/league_residual_profile_20261007/analysis.json` 참조. 모든 회차가 같은 궤적 해시 계산을 포함하며 untouched production 속도라는 뜻은 아니다.

**현재 속도 개선 실험:** 새 `experiments/league/grouped_vec_env.py`는 작업자마다 여러 독립 환경을 묶는다. 합성 검사2개 및 **실제4환경/65,536단계의 기존4작업자→묶음2작업자 전환에서 궤적·최종 가중치 완전 일치**를 확인했다. 실제 **PID14004 /실행세션4106**이 `aircombat-rl/runs/league_grouped_ppo_benchmark_20261007`에서 기존4작업자·4환경과 새4작업자·16환경을ABBA로 비교 중이다. 묶음당 PPO배치는4,096개로 같지만 환경당 수집 길이가1,024→256이라 **학습 데이터 구성과 GAE 범위가 달라지며 모델 품질 동등성을 주장하지 않는다.** 두 반복 모두 빨라짐·중앙 시간10% 이상 감소·커밋 여유2.8GiB 이상 등을 만족해야 다음 학습의 실행 후보로 권고한다. 기존/새 동결 실행기나 원본 정책은 수정하지 말고 세션4106을 계속 관찰한다. GitHub 업로드는 제외한다.

**2026-10-07 11:08 KST 4M 후보 후속 평가 완료:** 실제 PID50356은 종료코드0, 결과는 `selection_failed`다. 새 69010000 조건·74상대에서 기준은 **1,571승·879무·510패/2,960경기(53.07%)**, 4M 후보는 **2,088승·471무·401패(70.54%)**였다. 평균 +17.47%p, 코드그룹 균등 +5.29%p로 일반 프로필은 통과했지만 시간 상대 점수 +.021875(요구 .025), 우측 고속 회피 +.025(요구 .10)로 **약점 보완 기준에 미달**했다. Ace 승률 -67.5%p 등 퇴행도 있어 평균 상승만으로 승격하지 않는다. 최종70000000·홀드아웃71000000은 열지 않았다. `aircombat-rl/runs/league_residual_repair_followup_20261007/selection/selection_audit.json`은 원시 승패·점수·단계 수·양 좌석/조건·주요 판정을 재계산했다. 74상대가 두 모델의 저장기록42종으로 묶였고 최대26개 집단이 같았으며, 기록 집단별 균등 사후 점검은 -11.79%p다. 이 값으로 원래 판정을 바꾸거나 행동의 전역적 동일성을 주장하지 않는다.

**중복 원인 점검:** `aircombat-rl/runs/league_residual_continuation_20261007/selection/shared_interceptor_audit.json`에서 표준 로더로 26개 모델의 소스 경로와 실제 로드된 계수를 확인했다. 가중치 파일26종·소스4종이 제대로 로드되며 모두 같은 요격 계수와0.5초 분기 규칙을 공유한다. 합성 공개 입력96열/모델당576개에서 직진 상대 조건의 분기 후 행동은1종, 선회 조건은16종이었다. **로더 혼동 증거는 없고 특정 분기에서의 구조적 중복이 확인됐다.** 합성 입력은 실제 비행 궤적이나 완료 대결에서의 분기 기록을 입증하지 않는다. 다음 학습은 기존 모델을 모두 보존하되 학습 전용 조건에서 다양성을 반영하는 샘플링을 준비한다.

**현재 실행은 행동 선택 진단:** 실제 **PID30816**,4작업자가11:08:23부터 새72000000 조건의 **4,608경기**를 진행한다. 같은 6M 가중치의 탐욕적/확률적 행동을 비교하고 2M 모델도 함께 보고하며 새 모델을 승격하지 않는다. 그 뒤 대기PID48620이 PPO 속도 계측을 실행한다. 관찰 **세션72634/PID51476**을 그대로 사용하며 종료한50356·50212를 재시작하지 않는다. GitHub 업로드는 제외한다.

**2026-10-07 10:54 KST 주 PPO 실행 정상 종료·선별 실패:** 실제50212는 종료코드0으로 끝났고 관찰세션12352도 닫혔다. 새72상대/2,880경기씩 비교에서 기준은 **1,603승·784무·493패(55.66%)**,2M PPO 후보는 **1,850승·698무·332패(64.24%)**였다. 평균 승률+8.58%p·코드그룹 균등+5.35%p·하위 사분위 점수+.07639지만 열세 상대7→9, 시간 상대 평균 점수-.028125, 우측 고속 회피 개선0으로 사전 약점 기준에 실패했다. 최종67000000·홀드아웃68000000은 열지 않았다. `selection/selection_audit.json`에서 원시 승패/양 좌석·조건/프로필/선별 판정을 재계산했다. 기존 기준 및 모든PPO 후보를 보존하고 글로벌 모델을 승격하지 않았다.

**평균 개선의 적용 범위:** 이번 두 모델·20초기조건·양 좌석에서72상대는 **저장된 경기 기록40종**으로 묶였고, 가장 큰26상대 집단의 기록이 일치했다. 이 집단은 기준9승22무9패→후보22승17무1패로 바뀌어 평균 개선의 큰 부분을 차지한다. 같은 저장기록 집단마다 동일 비중을 주는 **사후 민감도 점검**에서는 승률 변화가-2.6875%p였다. 이는 반올림된 저장 지표의 일치이며 전역적 행동 동일성/독립 상대 표본을 뜻하지 않고, 원래 선별 기준을 바꾸는 근거로 쓰지 않았다. 새 열세 상대는 좌/우 펄스 정책이다(좌18승19무3패→5승29무6패, 우21승1무18패→14승5무21패). 현재 예약을 마친 뒤에는 모든 기존 상대를 보존하면서 **학습 전용 조건에서 반응 다양성을 측정하고 중복 상대의 샘플링 비중을 줄이는 방향**을 우선 검토한다.

**지금 실제 실행:** 후속 **PID50356**이 정상 종료를 확인한 뒤3300의4M 약점 후보와 기준을 **74상대·새69010000 조건에서 총5,920경기**로 선별 중이다. 이후30816 행동 방식 진단→48620 속도 계측 순서를 유지한다. 새 관찰세션은 **72634 /관찰PID51476**이며 이 세 프로세스의 생성시각·네이티브 핸들을 보유해45초 간격으로 기록한다. `runs/league_followup_chain_observer_20261007/runtime.json`·`events.jsonl` 참조. 관찰은 NumPy/Torch 없이 동작하며 추가 시뮬레이션은 없다. **세션12352를 다시 기다리거나 주PPO를 재시작하지 말 것.** `docs/LEAGUE_STATE.json`에서 현재 실행은 후속 평가로 바뀌었고, 완료PPO는 `completed_residual_ppo_continuation`에 있다. GitHub 업로드 금지는 유지한다.

**2026-10-07 두 PPO 기본 학습 종료·전체 상대 선별 진행:** RNG3300·3301은 각각추가8,388,608단계를 마쳤고 **둘 다 연장하지 않았다.** 추가학습 합계16,777,216단계·11,497완료경기, 공유 전반부1,032,200단계를 한 번 포함한 총17,809,416상호작용이다. 두 난수 모두72상대를 만났다. 같은 전반부 가중치/옵티마이저에서 시작한 별도 후속 학습이며 처음부터 독립 학습한 두 모델이라는 뜻은 아니다. `completed_training_prefix_analysis.json`이 완료된 두 결과/예산/원시 합계/입력 해시를 확인한다.

RNG3301의 마지막 개발 후보는 **13승·40무·43패/96경기**로 기준52승보다 낮아 기준을 유지했다. RNG3300은2M의64승 후보를 유지하고4M의58승 약점 후보도 보존했다. `s3300/s3301`의 `audited_chunks_004.json`·`development_tradeoffs_004.json`·`training_vs_greedy_diagnostic_004.json` 참조. 더 오래 학습한 후보를 자동으로 채택하지 않는다. 현재실제 **50212는 신경망 갱신을 마치고 전체72상대·새66010000 조건에서 기준과3300의2M 후보를 각2,880경기, 합계5,760경기로 선별 중**이다. 새로운4작업자 풀이 실행 중이며 관찰세션12352를 그대로 사용한다. **전체 실행 완료나 최종 개선 판정은 아직 아니다.** 선별/최종 조건부 분기가 끝난 뒤50356→30816→48620이 순서대로 이어진다. 원래 학습을 새로 시작하지 말고 실제 상태를 확인할 것. GitHub 업로드는 제외한다.

**2026-10-07 두 번째 PPO 난수의629만 단계에서 부분 회복:** RNG3301의 추가6,291,456단계 후보는 **29승·36무·31패/96경기**다. 직전1승·29무·66패보다 회복했지만 기준52승·24무·20패보다 낮아 기준 모델을 유지하며 약점 보완 자격 후보도 아직 없다. Ace2승2패·Pursuit2승2무·Lead2승1무1패, 우측 고속 회피1승3패·우측 펄스2승2무다. `s3301/audited_chunks_003.json`·`development_tradeoffs_003.json`에서 원시 기록/조건 짝/순위/해시를 확인했다. 실제50212는 마지막 기본 구간을 진행하고,839만 단계에서 기존 개선 기준으로 연장 여부를 정한다. 성능 하락이 단조롭다고 해석하거나 부분 회복만으로 연장이 유리하다고 결론내리지 않는다.

`s3301/training_vs_greedy_diagnostic_003.json`의 세 번째 확률적 학습 구간은392승/1,175경기(33.36%)다. 앞선38.41%→34.85%와 함께 기록하되 고정 탐욕적 평가와는 조건이 달라 원인 비교로 쓰지 않는다. 현재 학습과 후속50356→30816→48620의 순서를 유지하며 GitHub 업로드는 하지 않는다.

**2026-10-07 두 번째 PPO 난수의419만 단계 결과:** RNG3301의 추가4,194,304단계 후보는 **1승·29무·66패/96경기**다. 첫 후보4승·50무·42패보다 악화돼 기존52승 기준을 계속 유지한다. Ace/Pursuit/Lead와 우측 고속 회피·우측 펄스는 각각0승4패이며 Circler는4무다. `s3301/audited_chunks_002.json`·`development_tradeoffs_002.json`은 원시 결과/조건 짝/순위/해시를 확인했으며 이 난수의 약점 보완 자격 후보는 아직 없다. 실제50212는 사전 기본 예산의3번째 구간을 이어가지만, 현재까지 추가 연장을 정당화하는 개선은 없다.

`s3301/training_vs_greedy_diagnostic_002.json`에서 완료 구간별 확률적 학습 승률도38.41%(512/1,333)→34.85%(422/1,211)로 하락했다. 첫 난수에서 관찰한 “확률적 학습 승률 상승·탐욕적 평가 하락”이 두 번째 난수에서도 반복됐다고 말하지 않는다. 두 지표는 상대/조건/정책 고정 여부가 달라 여전히 직접 원인 비교가 아니다. 현재 학습과 이미 예약한50356→30816→48620 순서를 유지한다. 이 기록은 후속의 고정 후보/규칙을 바꾸거나 새 비행을 추가한 결과가 아니다.

**2026-10-07 10:02 KST 두 번째 PPO 난수의 첫 후보 악화:** RNG3301은 추가2,097,152단계·1,333학습경기 뒤 같은 개발96경기에서 **4승·50무·42패**를 기록했다. 같은 예산의RNG3300은64승·16무·16패였으므로 초기 개선이 반복됐다고 볼 수 없다. 기준52승·24무·20패 모델을 유지하며 실제50212는 다음 기본 구간을 진행한다. `s3301/audited_chunks_001.json`·`development_tradeoffs_001.json`에서 원시 경기·조건 짝·순위/해시를 재계산했고 약점 보완 기준은 통과하지 못했다. 이번 학습 구간의512승/1,333경기는 변하는 정책·가중 상대 분포의 값이므로 고정 탐욕적 후보의 원인을 단정하는 데 쓰지 않는다. 예약된 행동 방식 진단은RNG3300의 고정2M/6M 후보를 대상으로 하며 미래 결과를RNG3301에 자동 일반화하지 않는다. 현재 예산/실행 순서는 유지하고 GitHub에 올리지 않는다.

**2026-10-07 09:59 KST PPO 속도 계측 대기 연결:** `aircombat-rl/runs/league_residual_profile_20261007` 실제PID48620(런처53056)이 행동 방식 진단30816의 생성시각·네이티브 핸들을 확인하고 기다린다. **예약 순서는50212 학습 →50356 약점 후보 평가 →30816 행동 방식 진단 →48620 속도 계측**이다. 대기 중 시뮬레이션0개·NumPy/Torch 미수입·사유 메모리17.54MiB다. 합성 환경에서 실제PPO의 행동·업데이트가 계측 전후 동일하고 원래 메서드가 복원되는 등 검사3개가 통과했다. `runs/residual_profile_checks_20261007.json` 참조. 계측 소스/검사는 이제 동결돼 수정하지 않는다.

계측은 같은2M 체크포인트·난수·4환경/1024단계 설정으로 각65,536단계씩 **계측 없음→있음→있음→없음**, 합계262,144단계를 실행한다.72상대 캐시를 먼저 채우고 시작 시간은 따로 기록한다. 모든 실행에 동일한 궤적 해시 콜백을 넣어 추가 비용도 별도로 기록하며, 실제 비행에서는 궤적과 최종 가중치 해시가 모두 같아야 비교를 인정한다. 롤아웃 전체·행동/가치 계산·명령 전송·결과 대기·신경망 갱신·작업자 환경 계산 시간을 구분한다. 대기에는 시뮬레이션·통신·역직렬화가 섞이고 시간 항목은 중첩되므로 단순 합산하지 않는다. **실제 속도 결과나 가속 적용은 아직 없다.** 기존 예약이 정상 완료된 뒤 실행하며, 실패 시 중복 재시작하지 않고 원인을 확인한다. 이전 “계측 미예약” 이력은 이 내용으로 대체한다. GitHub 업로드 금지는 유지한다.

**2026-10-07 09:51 KST 첫 PPO 후속 학습 완료·두 번째 진행:** RNG3300은 추가8,388,608단계·6,401경기를 완료했고72상대를 모두 만났다. 마지막 탐욕적 개발 후보는 **2승·73무·21패/96경기**다. 보존된 최고 후보가2번째 체크포인트 이후 개선되지 않아 사전 규칙에 따라 연장하지 않았다. 기존64승 후보를 선택하고58승 약점 보완 후보도 보존했다. `s3300/audited_chunks_004.json`·`development_tradeoffs_004.json`은 완료 원시 기록·조건 짝·순위·해시를 재계산했다. **실제50212는 RNG3301의 학습을 계속 중**이며, 두 후속 학습이 공유하는1,032,200단계 전반부는 한 번만 계산한다. 첫 학습의 체크포인트 평가 비중은 측정된 학습+체크포인트 평가 시간 중2.419%이며 시작/저장 등은 제외한다.

**행동 방식 진단을 실제 대기 연결했다:** `aircombat-rl/runs/league_residual_action_mode_20261007` 실제PID30816(런처51120)은 후속 평가50356의 생성시각·네이티브 핸들을 확인하고 기다린다. **실행 순서는 학습50212 → 약점 보완 평가50356 → 행동 방식 진단30816**이다. 대기 중 추가 시뮬레이션0개, NumPy/Torch 미수입, 사유 메모리 약17.6MiB다. 검사5개가 통과했고 실제6M 가중치의 원래ZIP 항목 보존·다른 경로에서 합성786행동 재현·탐욕적/확률적 모드의786개 로짓 일치를 확인했다. 현재/대기 소스·입력 해시를 모두 검증했다. `runs/residual_action_mode_checks_20261007.json` 참조. 새 진단 소스/검사도 이제 동결돼 수정하지 않는다.

진단은 기존24상대·새72000000대역의16초기조건·양 좌석에서6M 후보의 최대 확률 행동과 온도1 확률적 행동을 비교한다. 행동 난수4100~4103을 모두 동일 가중치로 집계하고2M 탐욕적 후보도 참고한다. 총4,608경기이며 실행 직전 재사용 호환성 조건에서4경기의 리셋/재현 검사도 한다. **아직 진단 비행 결과는 없다.** 승률 향상5%p와 두 가중 방식의 탐색적 교차 부트스트랩 기준은 원인 가설의 다음 검토 방향만 정하며 모델 승격/공식 탐욕적 점수/새 상대 일반화를 뜻하지 않는다. 후속 프로세스가 실제 종료했다고 확인되기 전 다른 비행을 겹치지 않는다. 이전 “진단 미구현” 이력은 이 내용으로 대체한다.

**PPO 자원 확인:**09:42의15초 표본은 전체CPU18.8%·학습13.3%,4작업자, 커밋 여유 최소3.95GiB·물리 메모리8.58GiB다. 최근 완료3구간은 약2,614학습단계/초다. 예전16작업자 CEM의90%대 활용률을 현재PPO에 적용하지 않는다. 통신/동기화는 아직 원인 가설이며 현재 실행 설정은 변경하지 않았다. `runs/resource_assessment_ppo_20261007_0942.json`과 상태 파일 참조. 이미 예약된 실행 뒤 구간별 속도 프로파일링을 고려하되 **속도 비교는 아직 대기 연결되지 않았다.** GitHub 업로드 금지는 유지한다.

**2026-10-07 09:34 KST 세 번째 PPO 후보 악화:** RNG3300의 추가6,291,456단계 후보는 탐욕적 행동 평가에서 **3승·56무·37패/96경기**로 악화됐다. 기존64승 최고 후보와58승 약점 보완 후보는 보존돼 있으며, 동결 순위는64승 후보를 유지한다. `s3300/audited_chunks_003.json`·`development_tradeoffs_003.json`에서 원시 결과·양쪽 좌석/초기조건·순위·체크포인트 해시를 확인했다. 현재PID50212는4번째 기본 구간을 계속하며8,388,608단계의 연장 판단과 독립 후속 RNG3301은 변경하지 않는다. 대기 후속50356과 관찰 세션12352도 유지한다.

**원인 가설은 아직 미확정:** 각 구간의 확률적 학습 경기 승률은47.1%→49.6%→53.6%로 올랐지만, 고정 체크포인트의 탐욕적 개발 성능은64→58→3승으로 바뀌었다. `s3300/training_vs_greedy_diagnostic_003.json`에 완료된 구간별 승패와 학습 로그를 보존했다. 학습/개발은 상대 분포·초기조건·변하는 정책이 달라 이 차이만으로 원인을 단정하지 않는다. 이후 같은 가중치·상대·초기조건에서 확률적 행동과 최대 확률 행동을 직접 비교할 필요가 있다. 여러 행동 난수는 사전에 고정하고 가장 좋은 난수만 고르지 않는다. **그 진단은 아직 구현/대기되지 않았으며**, 이미 대기 중인50356의 실행 슬롯을 침범하지 않는다. 학습을 길게 했다는 이유만으로 신경망 전체가 무너졌다고 결론내리지 않는다.

**2026-10-07 09:30 KST 조건부 후속 연결 완료:** `aircombat-rl/runs/league_residual_repair_followup_20261007` 실제PID50356(런처4244)이 현재 학습50212의 생성시각과 네이티브 핸들을 확인하고 종료를 기다린다. 대기 중 **시뮬레이션 작업자0개·NumPy/Torch 미수입·사유 메모리 약18MiB**다. 관련 검사4개와 실제 두 체크포인트의 선별 예시를 확인했고 소스/입력 해시를 동결했다. `runs/residual_repair_followup_checks_20261007.json`·후속의 `queue_plan.json`/`queue_runtime.json` 참조.

현재 실행이 최종 기준을 통과하면 후속은 생략한다. 실패하면 두 RNG의 완료된 모든 체크포인트 중 **개발 약점 보완 기준을 통과했으나 주 후보로 선택되지 않은 모델을 각 최대1개** 고른다. 이미 주 비교에 포함된 모델은 제외하고, 후보 순위에는 개발 결과만 사용한다. 해당 모델이 없으면 추가 경기를 생략한다. 원래72상대에 주 후보/추가 후보를 더한 동일 아카이브에서 새 선택69010000·최종70000000·매개변수 홀드아웃71000000을 사용하며 기존 기준을 유지한다. 선택은 최대9,120경기, 통과할 때만 최종 최대12,160경기와 홀드아웃1,280경기다. 새로운 학습이나 현재 동결 실험의 후보 변경이 아니다.

**후속50356이 다음 시뮬레이션 실행 슬롯을 보유한다.** 학습 관찰 세션12352가 끝나면 실제 대기 프로세스를 먼저 확인하고 다른 학습/평가 풀을 겹치지 않는다. 비정상 종료에는 자동으로 재시도하지 않고 원인을 확인한다. 아래 “후속이 아직 없다”는 이전 이력은 이 내용으로 대체한다. 현재 PPO 학습과 GitHub 업로드 금지는 계속 유지한다.

**2026-10-07 09:19 KST 두 번째 PPO 체크포인트:** RNG3300의 추가4,194,304단계 후보는 **58승·18무·20패/96경기**, 첫 후보는64승·16무·16패, 기준은52승·24무·20패다. 동결된 평균 승률 중심 순위는 첫 후보를 유지하고, 실제PID50212·4작업자는3번째 구간 학습을 계속한다. `s3300/audited_chunks_002.json`에서 두 체크포인트의 원시 경기·동일 초기조건/양쪽 좌석·순위·가중치 파일을 재계산했다.

두 번째 후보는 우측 고속 회피0승4패→1승3패, 우측 펄스0승4패→1승3무로 보완했지만, 첫 후보 대비 Ace4승→1승3패·Pursuit4승→1승1무2패·Lead4승→3승1패로 악화됐다. 최악 상대 점수0→.25, 하위 사분위 .27083→.3125이며 **작은 개발 패널에서는 기존 약점 보완 기준을 통과**한다(시간 상대 평균 점수 +.078125·우측 고속 회피 +.25, 기준 모델 대비). 첫64승 후보는 그 개발 기준을 통과하지 않는다. 이는 최종/미사용 상대 성능 검증이 아니며 상대당4경기뿐이다.

`review_development_tradeoffs.py`·`s3300/development_tradeoffs_002.json`에 판단 근거가 있다. **다음 후속 준비:** 평균 점수 선두만 남기는 현재 선별 때문에 약점 보완 후보를 놓칠 수 있다. 현재 동결 실험은 바꾸지 않고, 이번 전체 실행이 최종 약점 기준에 실패하면 개발 기준을 통과했으나 제외된 체크포인트도 새 선택/최종 조건에서 별도로 비교하는 후속을 준비한다. **아직 그 후속 실행기나 대기 프로세스는 없다.** 후보 선택에 최종 결과를 사용하지 말고 새 기준/입력을 먼저 동결한다. 현재 학습/관찰 세션12352는 계속 유지하며 중복 실행하지 않는다.

**첫 PPO 후보 추론 묶음 보관:** `aircombat-rl/experiments/league/ppo_checkpoint_s3300_2097152_20261007.zip`(56,487바이트, SHA256 `579ba829d9eda353b3efa8e284765d7f6552ac08cafd9385f7bef06ceba8a774`)은 아래64승 개발 후보의 정책만 담는다. 원래 학습/정책 경로 접근과 `tools`/`experiments` import를 막은 별도 경로에서 합성 입력786개의 행동이 일치했고 Torch를 불러오지 않았다. 근거는 `aircombat-rl/runs/ppo_checkpoint_package_check_20261007/completion.json`이다. 이 합성 입력에서는 기존 초기 정책과도 행동이 모두 같으므로 학습한 행동 변경이나 성능 개선 검증으로 해석하지 않는다. 실제 개발 성능 근거는 별도96경기 결과다.

**최종 채택 모델이나 전체 학습 재현 ZIP이 아니다.** PPO 비평기/옵티마이저와 상대 아카이브는 이 추론 묶음에 없으며, 학습 재개에는 원래 실행의 `learner.zip`·동결 계획·소스/입력 자료가 필요하다. 최신 전체 재현 ZIP 범위는 여전히 앞선 완료 실험까지다. 현재 학습을 수정하거나 추가 비행을 실행하지 않았으며 GitHub에 올리지 않았다.

**2026-10-07 09:05 KST 첫 PPO 개발 개선:** 복구 실행 `league_residual_continuation_20261007`의 RNG3300은 추가2,097,152단계 후 **64승·16무·16패/96경기**를 기록했다. 같은24상대·같은2초기조건·양쪽 좌석에서 기존 기준과 시작 신경망은52승·24무·20패였다. 평균 승률 +12.5%p·코드 그룹 균등 승률 +21.484%p, 하위 사분위 점수 .16667→.27083·열세 상대6→4다. `s3300/audited_chunks_001.json`에서 원시 결과·초기조건/좌석·보존 순위·체크포인트 해시를 재계산했다. 재계산 명령은 `python runs/league_residual_continuation_20261007/audit_completed_checkpoints.py`다. 새 비행을 추가한 분석은 아니다.

**개발 선별 결과이며 최종 승격이 아니다.** 개선은 보존된 여러 제어기 계열을 상대한 승리 증가가 크다. 시간 상대 중 우측 고속 회피와 우측 펄스는 여전히0승4패이고, 좌측 위빙은2승→0승·우측 지연 선회는4승→2승으로 일부 악화됐다. Ace/Pursuit/Lead/Circler는 각각4승을 유지했지만 생존 체력이 낮아졌다. 상대당4경기만으로 일반적 우열을 확정하지 않는다. 후보는 `s3300/checkpoints/step_2097152/submission`에 보존하고, 실제PID50212·4작업자는 다음 구간 학습을 계속한다. 관찰 세션12352(PID52664)가 생성시각과 실제 네이티브 프로세스 핸들을 확인하며45초 간격으로 기록한다. 중복 실행하지 않는다.

첫 구간의 실제 학습 경과시간803.88초·체크포인트 평가16.12초로 두 시간 합계 중 평가는1.97%다. 시작 모델 평가·시작/저장/내보내기 비용을 제외한 구간 수치이며 전체 실험 비용으로 확장하지 않는다. 공유 전반부1,032,200단계는 별도이며 두 후속 RNG에 중복 합산하지 않는다. 기존 동결 예산·선택/최종/홀드아웃 조건은 유지하고 GitHub에 올리지 않는다.

**2026-10-07 08:52 KST 메모리 기준 종료 후 복구:** 이전8작업자 PPO33868은08:44에 메모리 여유 검사로 종료됐다. Windows 메모리 부족 충돌이 아니라 코드의 보호 기준이 발생한 것이며, 정확한 발생 순간의 자원 수치는 저장되지 않았다. `league_residual_ppo_20261007/interruption_analysis.json`을 읽는다. 실제1,032,200단계·772완료 경기·72상대를 경험했고,1,032,192단계의 완전한 rollout 이후 신경망·비평기·옵티마이저가 보존됐다. 마지막8단계는 업데이트 전 표본이며 폐기한다. 이전 실행과 소스/결과는 변경하지 않는다. 관찰 세션86925도 정상 종료했다.

별도 **`aircombat-rl/runs/league_residual_continuation_20261007` 실제PID50212**(런처6164)을 시작했다. **4작업자×1,024단계로 업데이트당4,096표본을 유지**하되, 환경별 GAE 구간이 길어지는 변경은 명시한다. 복원 검사1개와 실제32,768단계 이어 학습·내보내기·저장 복원 검사를 통과했다. 짧은 통합 실행의 최소 커밋 여유는4.54GiB였다. `tools/league_residual_recover.py`와 관련 소스/입력은 이제 동결됐다. 이 검사는 장기 메모리 안정성이나 성능 개선의 증명이 아니다.

독립 후속 RNG3300/3301은 **같은103만 단계 학습 상태를 출발점으로 복사**하고, 새 초기조건240000000/241000000과 새 난수를 사용한다. 처음부터 독립적으로 학습한 두 모델이라고 표현하지 않는다. 공통 전반부 계산량은 전체 비용에서 한 번만 합산한다. 각 후속의 기본 추가 예산8,388,608단계·개발 추세에 따른 최대16,777,216단계, 개발24상대×4경기 간격은 유지한다. 동일한 기준 모델/개발 조건의 기존96경기는 재사용하고 새 실행으로 세지 않는다. 저장된 신경망의 시작 개발 성능을 확인한 뒤 학습하고, 선택/최종/홀드아웃은 이전에 열지 않았던 예약 조건을 사용한다. 이력의33868/8작업자 실행 중 표시는 현재 상태가 아니다. GitHub 업로드 금지를 유지한다.

**2026-10-07 08:42 KST PPO 본 학습 실행 중:** `aircombat-rl/runs/league_residual_ppo_20261007`의 실제 제어기 **PID33868**(런처7132)이 RNG3200 학습을 시작했다. **8개 작업자·72상대·16코드 그룹**이며 다음 독립 RNG3201도 자동 실행한다. 기본은 각8,388,608조종 단계, 2,097,152단계마다24상대×4경기(양쪽 좌석)를 평가하고, 최근 개발 성능이 사전 기준대로 개선되면16,777,216단계까지 연장한다. 신경망 PPO 학습이며 기존 CEM 계수 탐색과 구분한다. 기존 제어기 행동을 초기 prior로 사용하고9개 공식 행동을 모두 학습할 수 있다. 공식 물리·승패와 이전 모델을 보존한다.

실제 비행96회에서 원래 모델/내보낸 초기 정책/학습 환경의 전체 경로·행동·공식 결과가 일치했고, 중간 평가4경기 후에도 원래 학습 환경의 상태/경로가 유지됐다. 학습 실행기 검사3개와 실제32,768단계 PPO 가중치 갱신·저장 복원·Torch/NumPy 비교도 통과했다. 근거는 `league_residual_qualify_20261007/completion.json`, `league_residual_smoke_20261007/completion.json`, 본 실행의 `launch_analysis.json`이다. **새 학습 모델의 성능 향상은 아직 입증되지 않았다.** 소스와 입력을 동결했으므로 실행 중 변경하지 않는다. 학습230000000/231000000, 개발66000000, 선택66010000, 최종67000000, 매개변수 홀드아웃68000000을 분리했다. 최종/홀드아웃은 선택 기준 통과 때만 연다.

**현재 PPO 자원 측정:** 15초 표본 전체CPU27.3%·학습과 작업자19.8%, 최소 커밋 여유2.40GiB·물리 메모리 여유7.35GiB다. 앞선 CEM의CPU92%와 혼동하지 않는다. CPU가 포화된 상태는 아니며 통신/동기화 대기가 병목 후보지만 인과 프로파일링은 아직 하지 않았다. 작업자 수를 무작정 늘리지 않고 현재 학습을 유지한다. `resource_observation_0840.json` 참조. 후속 최적화는 작업자당 여러 환경을 묶어 통신을 줄이는 방식 등의 실제 배속을 비교할 수 있다. 별도 시뮬레이션 풀을 겹치지 않는다. 최신 ZIP에는 이 작업이 아직 포함되지 않으며 GitHub에는 올리지 않는다.

**2026-10-07 08:30 KST 완료 결과·다음 실행:** 추격 조건 CEM은 두 RNG 모두6세대로 끝났고 선택 기준에 실패했다. 실제 학습50,496경기·개발4,140경기·초기 기준1,380경기·선택8,280경기다. 동결 로그의50,688에는 재사용192경기가 있으므로 실제 실행 수로 인용하지 않는다. 선택에서 기준1586승·752무·422패, 두 후보1575승·752무·433패와1588승·747무·425패로 약점 개선을 입증하지 못했다. `aircombat-rl/runs/league_pursuit_context_train_20261007/completed_experiment_analysis.json`에서 원시 기록·양쪽 좌석·동결 해시를 확인했다. 기존 정책을 유지하고 최종64000000·홀드아웃65000000은 열지 않았다.

학습PID41968·관찰 세션4253·대기 비교PID1800은 모두 종료됐다. 유휴 작업자 지원 방식은864회 공식 결과/선택 비행 추적이 같았으나 중앙11.587초→12.539초로 약8.2% 느려 **기존14경량+2일반 설정을 유지**한다. `aircombat-rl/runs/league_flexible_benchmark_20261007/completion.json` 참조. 현재 실행 중인 학습은 없으며, 다음 작업은 준비된 PPO 정책의 실제 비행 검증과 혼합 상대 학습 실행기다. 아래 실행 중 표시는 당시의 이력이다. 최신 ZIP에는 이번 완료 실험/속도 비교/PPO 준비물이 아직 포함되지 않는다. GitHub 자동 업로드 금지를 유지한다.

**2026-10-07 08:20 KST 선택 비교·PPO 준비:** 현재 CEM의 두 RNG는 모두6세대로 종료하고 별도 선택 비교를 진행 중이다. 실제PID41968 종료 후에는 대기된 속도 비교PID1800이 먼저 실행된다. PPO 시제품은 혼합 상대 환경(`residual_env.py`) 검사6개와 제출물 내보내기(`league_residual_export.py`) 검사2개를 추가로 통과했다. 실제 보존 기준 모델을 넣은 초기 Torch/NumPy 정책이 합성 입력786개에서 같은 행동을 냈고, 별도 경로에서 `tools`/`experiments` 접근을 막아도 제출물이 동작하며 Torch를 불러오지 않았다. `runs/residual_policy_prototype_20261007/environment_export_checks.json`·`real_teacher_identity.json`과 `source_snapshot`이 근거다. **실제 비행 검사와 PPO 학습 실행기는 아직 남아 있고 신경망 학습은 시작하지 않았다.**

내보내기 검사 중 Windows `platform._wmi_query`에서 `0x8007000e` 진단이 출력됐으나 검사2개가 끝나고 종료코드0을 확인했다. 당시 후측정 커밋 여유는5.03GiB였고 CEM/대기 프로세스가 모두 살아 있었다. 원인은 확정하지 않았으며 학습 충돌로 해석하지 않는다. 새 준비 코드는 현재 동결 실험에서 사용하지 않는다. 현재 검증 묶음 ZIP에도 아직 포함되지 않는다.

**2026-10-07 08:03 KST 진행·후속 정책 준비:** 현재 학습PID41968은 RNG3101의5세대를 진행한다.4세대 개발 후보가2세대와 같은 `5a5ecfbeabd6`여서716승 결과를 재사용했으며, 새 독립 검증으로 세지 않는다. 대기 속도 비교PID1800이 다음 실행 슬롯을 보유한다. 별도 `experiments/league/residual_features.py`·`residual_ppo_policy.py`의 PPO 핵심 시제품은 합성 검사3개를 통과했다(`runs/residual_policy_prototype_20261007/software_checks.json`). 기존31개 공개 특징에 기준 제어기 행동9개를 더하고, 초기에는 제공된 기준 행동을 그대로 고르며 학습으로9개 조종 중 다른 행동을 고를 수 있다. Torch/NumPy 출력 및 저장·복원도 검사했다. **혼합 상대 환경·학습 실행기·전체 제출물 내보내기·실제 비행 검사는 아직 없고 PPO 학습을 시작한 것이 아니다.** 현재 CEM/대기 비교 종료 후 결과를 보고 후속 예산과 실행을 정한다. 이 시제품도 최신 완료자료 ZIP 범위 밖이다.

**2026-10-07 07:51 KST 두 번째 반복 개발 결과:** RNG3101의2세대 후보 `5a5ecfbeabd6`는 **716승·527무·137패/1,380경기**, 기준은707승·531무·142패다. 평균 승률 +0.652%p·그룹 균등 승률 +0.331%p·시간 상대 평균 점수 +.034375이나 고속 회피 우측 개선0으로 자격 기준에는 미달한다. 원시 요약·양쪽 좌석/초기조건을 확인한 `s3101/g1/development_analysis.json`을 읽는다. PID41968은3세대 학습을 진행 중이며, 대기 속도 비교PID1800은 유지된다. 앞선 RNG3100의4세대에서는 고속 회피20패가16패·4상호격추로, Pursuit20승이9승·11시간초과로 바뀌었다. `s3100/g3/outcome_transition_analysis.json`에69상대의 짝지은 결과 변화와 후속 가설이 있다. 다음 전략에서는 생존/무승부 개선과 승리 증가를 구분한다. PPO 후속은 아직 설계 검토이며 현재 실행 중인 신경망 학습은 없다.

**2026-10-07 첫 반복 완료·두 번째 반복 실행 중:** `league_pursuit_context_train_20261007`의 RNG3100은6세대로 끝났고, 실제 PID41968이 RNG3101의2세대를 진행 중이다. 첫 반복의 **실제 학습24,896경기·별도 개발2,760경기**를 원시 기록으로 재계산했다(초기 공통 기준 평가는 별도). 동결 제어기의 `training_games=25088`에는 재사용192경기가 포함돼 있으므로 실제 실행 수로 인용하지 않는다. `s3100/completed_search_analysis.json`·`analysis_completed_search.py`에 저장 요약/좌석/초기조건/비용/연장 판단 검사가 있다.4→6세대에서 보존된 개발 점수가 오르지 않아 사전 규칙대로 연장하지 않았다. 첫 후보의 개발1승 증가가 시간 상대 약점 보완을 입증하지는 않는다.

**후속 실행 슬롯 예약:** 속도 비교 `league_flexible_benchmark_20261007` 실제PID1800(런처29048)이 학습41968의 생성시각을 확인한 네이티브 핸들로 종료를 기다린다. 현재 시뮬레이션 작업자0개·사유 메모리 약18MiB·NumPy/PyTorch 미수입을 확인했다. **현재 학습이 끝났다고 바로 다른 대결을 시작하지 말고 이 대기 프로세스를 먼저 확인할 것.** 기존14경량+2일반 풀과, 유휴 일반 작업자가 경량 경기를 돕는 별도 `FlexiblePool`을 같은216경기씩 정순/역순 비교한다(합계864실행). 관련 검사6개·동결 해시를 확인했으며 **실제 배속은 아직 미측정**이다. 기록 일치·양 반복에서 빠름·중앙 시간3% 이상 감소·메모리 여유 기준을 모두 충족해야 새 실행에 채택한다. 기존 실행 프로필을 자동 덮어쓰지 않는다. 새 풀/비교기/검사도 이제 동결돼 있어 수정하지 않는다. 학습 관찰 세션4253이 종료하면 대기 비교PID1800의 실제 상태를 확인한다.

현재 최신 ZIP은 직전 완료 실험까지이며, 위 진행 학습과 새 실행 분배 코드는 포함하지 않는다. GitHub 자동 업로드 제외를 유지한다.

**2026-10-07 07:34 KST 완료 자료 보관·학습 계속:** 새 로컬 묶음 `aircombat-rl/experiments/league/replay_snapshot_contextual_20261007.zip`을 완성했다. 이전 분리 조합 평가부터 완료된 시간 상대 보완 학습114,400경기와 속도 예비 실험8,784경기까지 포함한다. 기준 커밋의 새 별도 경로에서 파일2,381개·정책208개·합성 행동163,488개·관련 검사19개와 완료 학습 요약 재계산을 통과했다. 추가 비행/전체 학습 재실행은 하지 않았다. ZIP 19,303,049바이트, SHA256 `e0904cfe3b9ab848cee7a783e1ee5c580fb6ae16ff8a8172e41bf3b36146f3f4`. 범위는 묶음의 `README.md`·`verification_report.json`을 읽는다. **현재 진행 중인5계수 추격 조건 학습은 제외**하며 GitHub에는 올리지 않았다.

현재 학습PID41968·관찰 세션4253은 계속 실행 중이며 RNG3100의5세대가 완료되고6세대를 진행 중이다. 첫 개발 후보는708승·530무·142패,4세대 후보는698승·571무·111패, 기준은707승·531무·142패(각1,380경기)다.4세대에서 일부 시간 상대 점수와 패배 수는 개선됐으나 평균·그룹 균등 승률이 낮아 동결 학습 순위는 첫 후보를 유지했다. `league_pursuit_context_train_20261007/completed_development_audit_g1_g3.json`은 저장 요약과 양쪽 좌석/초기조건을 재계산했다. **최종 채택·독립 시험 결과가 아니다.** 남은 세대와 두 번째 독립 RNG를 계속 실행하고, 사전 규칙에 따라 연장 여부를 판단한다.

07:21 자원 측정은 전체CPU92.4%·학습85.8%,16작업자 중 일반 작업자2개가 해당 구간에 유휴였다. 현재 실행을 동결한 채, 다음 실행에서 유휴 일반 작업자가 NumPy 경기를 돕는 분배의 실익을 확인할 후보로 남겼다. 추가 배속은 아직 측정되지 않았다. `aircombat-rl/runs/resource_analysis_20261007_0723.json` 참조.

**2026-10-07 07:05 KST 새 독립 학습 실행 중:** `aircombat-rl/runs/league_pursuit_context_train_20261007`의 실제 제어기 **PID41968**(런처37008)이 RNG3100의 첫 세대를 실행 중이다. 새 정책은 기존 상황별 속도 조절4계수에 **상대가 자신에게서 멀어지는 방향의 코사인 조건**을 더한5계수다. 공개 위치·속도만 사용하며 조건을 끄면 기존 속도 정책을 포함한다. 탐색 동작·기본 제어기·요격기·공식 물리/승패는 고정했다. 관련 검사6개와 실제 비행 호환성100회+무제한 조건의 기존 정책 일치16회가 통과했고, 실제 경량14+일반2작업자를 확인했다. 소스/입력 해시도 일치한다. **새5계수 정책의 성능 개선은 아직 입증하지 않았다.**

전체 상대는69개·코드 그룹15개이며 세대별32상대를 순환한다. 각 세대에 모든 시간 상대8개·기준 모델·모든 코드 그룹이 포함되고, 두 RNG 모두 기본6세대 안에 전체69상대를 만나도록 실제 목록을 확인했다. 개발/선택/최종 비교는 전체69상대를 사용한다. 독립 RNG3100·3101, 기본6세대·개발 추세에 따라최대10세대, population12, 선별6경기·확인16경기·개발20경기/상대다. 개발63000000·선택63010000·최종64000000·매개변수 홀드아웃65000000으로 분리했다. 최종 비교는 고정된 새 후보와 기준 모델을 전체 상대에 대해 비교하며, 통과 후에만 아직 사용하지 않은 시간 정책 매개변수 조합을 평가한다.

앞선 속도 조절 예비 실험은 **06:47에 정상 종료**했다. 확인 조건에서 기준895승·339무·230패, 적극적c0는827승·400무·237패, 보수적c6는899승·342무·223패(각1,464경기)였다. c0는 고속 회피 우측0승→9승/24경기·우측 위빙5승→23승을 만들었으나 Ace24승→8승·Lead23승→5승으로 악화됐다. c6는 전체 승률 +0.27%p·시간 상대 평균 점수 변화0으로 약점 기준에 미달했다. `league_contextual_speed_pilot_20261007/analysis.json`에서 원시 요약/양쪽 좌석 초기조건을 확인했다. 이 관찰이 새 이동방향 조건의 가설을 뒷받침하지만 인과 효과를 입증한 것은 아니다. 기존 정책과 모든 후보를 보존한다.

**실행 관찰:** 이전 프로세스47900·8368과 관찰 세션93392는 종료됐다. 현재 관찰 세션은 **4253**이며 실제41968의 생성시각이 맞는 네이티브 핸들을 잡고 `observer_events.jsonl`에45초 간격 메타데이터·메모리 상태를 기록한다. 이 세션을 재사용하고 중복 실행하지 말 것. 새 정책/내보내기/학습 소스도 이제 동결돼 있으므로 수정하지 않는다. **남은 보관 작업:** 직전114,400경기 학습과8,784경기 속도 예비 실험을 포함하는 새 재실행 묶음이 아직 필요하다. 기존 ZIP은 이전 분리 조합 평가까지다. 현재 학습과 병행해 완료 자료를 정리하되 추가 시뮬레이션을 겹쳐 실행하지 말 것. GitHub 업로드는 사용자의 새 명시적 요청 전까지 제외한다.

**2026-10-07 06:39 KST 학습 완료·후속 실험 실행 중:** `league_temporal_repair_20261007`은 두 독립 RNG의 **6세대/10세대·학습114,400경기**, 별도 개발 검증7,700경기, 선택 비교4,400경기로 완료됐다. `completed_searches_analysis.json`은 실제 실행된 기록만 합산하고 모든 저장 요약·동일 초기조건 양쪽 좌석·보존된 개발 학습 순위를 재계산했다. 재계산 명령은 `python runs/league_temporal_repair_20261007/analysis_completed_searches.py`다. 선택 조건57010000에서 기준 **1464승·393무·343패**, 8세대 후보 **1469승·374무·357패**로 평균 승률 +0.23%p·패배 +14·열세 상대 +1·하위 사분위 -.00268, 고속 회피 우측 개선0이어서 자격 기준을 통과하지 못했다. **최종60000000·매개변수 홀드아웃61000000은 미사용**이며 기존 분리 조합을 유지한다. 전체 판단은 `completed_experiment_analysis.json`에 있다. 동결 제어기의 완료 상태명 `no_development_improvement`는 선택 실패에도 그대로 남는 구현상 기본값이므로, 두 번째 RNG에서 개발 점수가 개선되지 않았다는 뜻으로 해석하지 말 것.

원래 제어기 **PID47900의 실제 종료를 확인**했고, 대기하던 **PID8368**이 `league_contextual_speed_pilot_20261007`을 자동 시작했다. **실제 비행100회 동작/로더 일치 검사 통과**, 경량14+일반2작업자로 **61상대·속도 정책8개**의 개발 선별을 진행 중이다. 새로 학습한 개발 후보들도 상대 목록에 포함했다. 선별62000000은4,392경기, 상위2개+기준의 확인62010000도최대4,392경기다. **개발용 예비 비교이며 최종 시험·모델 승격이 아니다.** 기존 관찰 세션93392는 남은 PID8368을 계속 감시한다. 중복 실행·동결 소스 변경·GitHub 업로드 금지. 최신 ZIP은 이번 학습/후속을 아직 포함하지 않으므로 완료된 후 새 보관 묶음이 필요하다.

**2026-10-07 06:14 KST 연장 학습 진행:** RNG3001의 8세대·학습 **57,200경기**가 완료됐고 PID47900이 9세대를 진행 중이다. 8세대 후보 `league_temporal_repair_20261007_5258a80e42b1`의 개발 결과는 **714승·243무·143패**다. 6세대보다 평균 승률 -1.27%p이나 하위 사분위 점수 +.02143·열세 상대8→7로 일부 회복했고, 사전 정의한 가중 학습 점수 .595621→.596199로 개발 최고 후보가 갱신됐다. **원래 기준 대비 하위 사분위 -.01071·열세 상대 +1·고속 회피 우측 개선0으로 최종 자격은 아직 실패**한다. `s3001/g7/development_analysis.json`은 원시 승·무·패/점수와 세 모델의 동일 초기조건·양쪽 좌석 짝을 확인한다. 10세대 예산과 대기 후속 PID8368은 그대로다. 별도 최종/홀드아웃 결과가 아니다.

**2026-10-07 05:56 KST 학습 자동 연장:** 두 번째 RNG 3001은 6세대·학습 **43,120경기**를 완료하고, 사전에 정한 예산 규칙을 통과해 **10세대까지 연장**됐다. 실제 제어기 PID 47900이 7세대를 시작했다. 6세대 개발 후보 `league_temporal_repair_20261007_0d8252a3f0fc`는 1,100경기에서 **728승·230무·142패**, 평균 승률66.18%(기준64.00%)였다. 가중 학습 순위의 첫 점수가 .587848→.595621(+.007773)로 기준 .0025 이상 개선되고 평균 승률도 감소하지 않아 연장됐다. `s3001/budget_decision.json`과 `s3001/g5/development_analysis.json`에서 확인한다.

이 후보는 **개발용 최고 후보이며 최종 채택 모델이 아니다**. 하위 사분위 점수 -.03214, 시간 상대 평균 점수 -.01875, 열세 상대6→8, 고속 회피 우측 상대 개선0으로 약점 기준은 아직 실패했다. 원시 승·무·패/점수 및 동일 초기조건의 양쪽 좌석 짝을 확인했다. 이후 개발·선택 결과로 판단하며 최종/매개변수 홀드아웃은 아직 사용하지 않았다. 후속 속도 정책 실험 PID8368은 연장된 원래 프로세스의 종료를 계속 기다린다. 실행 중인 두 실험을 복제하거나 동결 코드를 수정하지 말 것.

**2026-10-07 05:40 KST 개발 결과·실행 비용:** 현재 `league_temporal_repair_20261007`의 두 번째 독립 RNG 3001은 4세대·학습 28,160경기를 완료하고 5세대를 진행 중이다. 4세대 개발은 1,100경기에서 **712승·247무·141패**, 기준은 **704승·271무·125패**였다. 평균 승률 +0.73%p에도 하위 사분위 점수 -.0161, 시간 상대 평균 점수 -.01875, 열세 상대 +1이며 고속 회피 우측 상대 개선은 없다. 동결된 학습 순위상 기존 모델을 유지했다. 원시 승·무·패와 점수, seed/seat 중복 여부를 확인한 `s3001/g3/development_analysis.json`을 참고한다. 6세대 예산 규칙은 변경하지 않았다.

완료된 첫 9세대만 집계한 `completed_prefix_budget_audit_9_generations.json`에서는 **학습 63,800경기·별도 개발 검증 3,300경기**였다. 검증 비중은 경기 수 4.92%, 작업자별 경기 소요시간 합계 4.31%이다. 선별·확인은 CEM 분포를 갱신하는 학습용 대결이며, 재사용 평가 사본과 진행 중인 세대는 제외했다. 이 시간 합계는 실제 경과시간이나 CPU 시간이 아니다. 현재 실험에서 별도 검증이 주된 비용이라는 근거는 없다.

05:28의 15초 관측은 **전체 CPU 약100%, 학습 프로세스 약91.7%, 실제 작업자16개**, 최소 커밋 여유5.41GiB·물리 메모리 여유9.84GiB였다(`runs/resource_observation_20261007_0528.json`). 관측 구간의 수치이며 전체 실행의 평균·최댓값을 뜻하지 않는다. 실행 프로필을 더 바꾸지 않고 학습을 유지한다. 실제 PID와 파일 메타데이터는 다시 확인할 것.

**2026-10-07 05:12 KST 조건부 후속 실행 연결:** 현재 학습 PID 47900은 두 번째 RNG 3001의 2세대를 진행 중이다. 별도 `runs/league_contextual_speed_pilot_20261007`을 실제 대기 제어기 **PID 8368**(런처 37392)으로 연결했다. 원래 학습 프로세스 핸들을 잡고 종료를 기다리며 대결 작업자 0개, NumPy/PyTorch 미사용, 사유 메모리 약 17.54MiB를 확인했다. 현재 실험이 `repair_profile_passed`이면 건너뛰고, 완료된 `no_development_improvement`/`repair_profile_failed`일 때만 실행한다. 비정상·미완료 종료는 자동 재시도하지 않는다.

후속은 먼저 소비된 호환성 조건의 실제 비행 100회로 비활성 정책 동일성 및 원래/경량 로더의 전체 경기 기록을 비교한다. 모두 통과해야 새 코드 서명을 경량 풀에 넣는다. 이어 속도 조절 설정 8개를 전체 보존 상대+이번 개발 후보들과 선별(조건 62000000·상대당 8경기), 상위 2개를 별도 조건에서 확인(62010000·상대당 24경기)한다. **개발용 예비 비교이며 최종 시험·미학습 상대 주장·모델 승격·GitHub 업로드는 없다.** 관련 검사 6개와 현재 학습/대기 계획의 동결 해시를 확인했다. 대기 중인 새 정책·내보내기·후속 실행 소스도 이제 수정하지 말 것. 실제 상태는 `queue_runtime.json`의 PID와 프로세스, 완료/실패 파일로 확인한다.

**2026-10-07 05:03 KST 첫 독립 학습 완료:** `league_temporal_repair_20261007/s3000`은 6세대·학습 **42,240경기**로 끝났고 개발 개선이 없어 **연장하지 않았다**. 개발 승·무·패는 2세대 704/271/125, 4세대 701/261/138, 6세대 704/271/125(각 1,100경기 기록)였다. 6세대는 동일 기준 정책 요청을 재사용한 것으로 새 평가 경기라고 세지 않는다. 기존 분리 조합을 유지했으며 최종 시험은 아직 열지 않았다. 원시 요약 재계산과 예산 판단은 `s3000/completed_search_analysis.json`에 있다. **같은 제어기 PID 47900이 두 번째 독립 RNG 3001을 실제 시작**했고 첫 세대 대결 기록을 확인했다. 첫 검색 관찰용 exec 세션 95284는 목표 파일 생성 후 정상 종료한 보조 관찰기이며 학습 프로세스 종료가 아니다. 현재 메모리 할당 여유 약 5.63GiB, 실패 파일 없음. 두 번째 검색의 완료 결과로 후속 정책 실험 여부를 판단한다.

**2026-10-07 04:53 KST 진행:** 시간 상대 보완 학습 PID 47900·16작업자를 유지한다. 첫 독립 RNG 3000의 4세대까지 학습 27,280경기가 완료됐고 현재 5세대 후보 확인 중이다. 2세대 개발은 기존 모델과 상대별 승·무·패가 같았고, 4세대 개발 후보는 평균 승률 -0.27%p·시간 상대 평균 점수 -.03125·열세 상대 +2여서 기존 모델을 유지했다. 고속 회피 상대 점수 개선도 없다. `s3000/g1/development_analysis.json`, `g3/development_analysis.json`에서 원시 요약을 재계산했다. 6세대 연장 판단과 두 번째 RNG 3001은 아직 대기다. 실제 실행·완료 여부는 다시 확인할 것.

후속 선택지 `experiments/league/contextual_speed.py`와 `tools/league_contextual_speed_export.py`를 별도로 준비했다. 공개 관측의 상대 속도·거리·경과시간 조건에 따라 추적 속도만 높이고, 첫 탐색·요격 경로·발사 기회·기존 방어 동작은 보존하는 4계수 정책이다. 비활성화 시 기존 정책 일치, 조건/방어/사격 보호, 별도 경로 내보내기·시계 초기화 검사 3개 통과. **실제 대결 0회, 성능 미검증, 현재 학습에 적용하지 않음.** 현재 실험이 완료된 결과로 후속 실제 대결 실험이 필요한지 판단한다. 새 코드의 경량 작업자 서명은 아직 실행 프로필에 추가하지 않았다.

**완료 결과 보관:** 직전 분리 조합의 최종·시간 상대 평가까지 담은 새 로컬 묶음은 `aircombat-rl/experiments/league/replay_snapshot_separated_final_20261007.zip`(7,073,386바이트, SHA256 `d159fd62feab0180b7cc54e13e0371b2d00fce1b1cd9066e714838a938865ef4`)이다. 별도 경로에서 파일 523개·정책 59개·합성 행동 10,797회·검사 21개를 확인했다. 묶음 안의 시작 문서와 상태는 그 완료 단계에 맞게 작성했다. **아래 현재 실행 중인 후속 학습과 16작업자 개선은 이 ZIP에서 제외**한다. 현재 학습은 별도로 계속 실행 중이며 실제 PID/메타데이터로 확인한다. GitHub에는 올리지 않았다.

**2026-10-07 04:11 KST 새 학습 시작:** `aircombat-rl/runs/league_temporal_repair_20261007`의 실제 제어기 PID **47900**(런처 48632)이 실행 중이다. 경량 14작업자는 PyTorch 미사용, 일반 2작업자는 PyTorch 사용을 확인했고 실제 대결 파일이 저장됐다. 기존 45상대·분리 조합 2개·소비된 시간 상대 8개, 총 **55상대**를 유지한다. 첫 .5초 탐색·게이트·요격기는 고정하고 이후 기본 제어기의 유효 계수 11개를 CEM으로 학습한다. 짧은 진단 64장면에서 시간 상대들은 모두 이 기본 경로를 사용했다(소비된 조건 재사용, 성능 증거 아님).

독립 RNG **3000·3001**, 기본 각 **6세대**, 개발 개선 시 최대 각 **10세대**다. 학습은 uniform40%·코드그룹30%·소비된 시간 상대 약점30%로 가중한다. 새 개발/선발 조건 57000000/57010000, 조건부 최종 60000000, 별도 상대 파라미터 평가 61000000은 분리했다. 후자는 알려진 시간 행동 가족의 새 계수 조합이며 새 아키텍처라고 부르지 않는다. 개발 최고 후보를 독립 반복별 하나씩 선발하고, 분리 기준 모델 대비 기존 새 프로필+시간 상대 평균 점수 .025·extend_right 점수 .10 개선을 요구한다. 최종은 고정한 후보 하나로 확인하며 실패한 과거 기준을 통과로 바꾸지 않는다. 관련 검사 13개 통과. **현재 동결 실행 소스·입력은 수정하지 말 것.** 아래 이전 PID/완료 표현은 이력이다. GitHub 업로드 금지 유지.

**2026-10-07 04:04 KST 실행 속도 개선 완료:** `league_split_benchmark_20261007`이 정상 종료했고 현재 학습 프로세스는 없다. 다음 새 실험은 `aircombat-rl/experiments/league/execution_profile.json`의 **16작업자(경량 14+일반 2)** 설정을 사용한다. 동일 216경기를 구성별 정순/역순으로 반복한 총 1,728경기에서 경기·피격 기록이 일치했다. 기존 8작업자 중앙 처리 시간 16.305초 → 새 16작업자 11.217초로 처리량 1.454배, 소요 시간 31.2% 감소. 평균 CPU 51.3% → 83.9%, 최소 할당 여유는 3.993 → 5.795GiB다. 전체 장시간 학습의 단축률은 아직 측정하지 않았다. `tools.league_split_pool.SplitPool`은 확인한 NumPy 정책 코드만 경량 작업자로 보내고 신경망/미확인 정책은 일반 작업자로 보낸다. Python 시작 전 3종 스레드 제한을 1로 설정하고 현재 메모리 여유를 다시 확인한다. 기존 동결 실행의 설정은 바꾸지 않는다. **다음 작업은 이 설정으로 소비된 시간 상대 약점을 보완하는 새 학습을 시작하는 것**이며, 아래 실행 중 표현은 과거 기록이다.

**2026-10-07 04:00 KST 업데이트:** 분리 조합의 최종 14,400경기·시간 상대 2,560경기와 원시 기록 재계산 보고가 모두 완료됐다. PID 40404·25636은 종료됐다. 45상대 최종 프로필 통과는 유지되지만 시간 상대에서는 선택 모델 240승·142무·258패/640경기, 기존 검증 모델 243승·140무·257패였다. 승률 차이 -0.47%p(95% 구간 -5.00~+4.06)로 일반화 개선 근거가 없고 `temporal_extend_right`에는 0승·80패다. **전체 Goal은 계속 진행한다.** 이 시간 상대들은 이제 소비된 개발 상대이며 다시 미학습으로 부르지 않는다. 상세: `aircombat-rl/runs/league_separated_validate_20261007/analysis_auto/report.md`.

사용자의 자원·속도 질문에 따라 `runs/league_split_benchmark_20261007`에서 기존 8작업자와 경량/신경망 분리 8·12·16작업자를 같은 경기로 비교 중이다. 시작 전 학습 Python은 없었고 CPU 약 3%, 물리 여유 11.4GiB·할당 여유 7.06GiB였다. 실제 제어기 PID는 `docs/LEAGUE_STATE.json`과 프로세스로 확인할 것. 새 실행기는 기존 동결 코드를 수정하지 않는다. 정확한 경기·피격 기록 일치와 메모리 여유를 확인하기 전 속도 개선을 확정하지 않는다. GitHub 업로드 금지 유지.

**2026-10-07 03:45 KST:** 선택 후보가 14,400경기 최종 시험의 `aggregate_and_absolute_weakness_v1` 기준을 통과했다. 기존 검증 모델 대비 평균 승률 +13.44%p(초기조건 묶음 95% 구간 +11.22~+15.50), refine_2200 대비 +9.50%p(+7.33~+11.64)이고 두 코드 그룹 균등 승률 구간도 양수다. 최악·하위 점수와 열세 상대 수 조건도 통과했다. **기존 상대별 퇴보 제한은 미통과**이며 이를 기존 기준 통과로 표시하지 않는다. 4c 대비 평균 승률은 -.64%p(구간 -1.75~+.44)인 반면 하위 1/4 점수는 +.02448·최악 점수는 +.05다. 4c보다 평균 승률이 높다는 결론은 없다.

같은 PID **40404**·8작업자가 시간 기반 미학습 상대 32묶음 중 16묶음까지 실행했다. **52000000 패널은 이제 개봉/소비됐으므로 다시 미학습 증거로 쓰지 않는다.** 보고 대기 PID 25636도 실제 살아 있다. 최종 판정은 `final_decision.json`, 전체 완료·독립 원시 재계산 보고서는 아직 대기다. 공식 환경·판정 29파일을 원본 클론 커밋 4141a299와 비교해 내용 차이가 없었고 기존 CEM 계수도 원본 보존 파일/공개 기준 커밋과 같았다. `physics_preservation_audit.json`에 범위와 파일 해시가 있다. 설치된 외부 바이너리 전체를 대조한 것은 아니다. 아직 새 후보를 최종 보관 모델로 승격하지 않았다.

**2026-10-07 03:34 KST:** 최종 시험 PID 40404·8작업자가 실제 실행 중이며 180묶음 중 100묶음(8,000/14,400경기)이 저장됐다. 아직 최종 판정·완료·실패 파일은 없다. 새 보고기 `tools.league_validation_report`를 실제 PID **25636**으로 연결했다. `runs/league_separated_validate_20261007/analysis_auto/watch_plan.json`에 소스 해시·대상 프로세스를 동결했고, 대기 중 NumPy/PyTorch를 불러오지 않으며 사유 메모리는 약 16.1MiB다. 45초 간격 자원 관측은 보고기 시작 이후의 표본이며 전체 실행의 정확한 최대값이 아니다. 원래 프로세스 핸들을 유지해 PID 재사용을 혼동하지 않는다. 선행 실행이 완료되면 원시 에피소드에서 요약·쌍별 신뢰구간·피격을 다시 확인하고 `analysis_auto/report.json`, `report.md`를 생성한다. 원시 요약 불일치 거부와 경량 대기 검사는 2개 통과했다. 추가 대결·모델 선택·승격·GitHub 업로드는 하지 않는다. 보고기와 현재 동결 실행 소스는 수정하지 말 것.

**2026-10-07 03:23 KST 현재:** 분리 조합 개발 비교가 완료됐다. `league_separated_probe_20261007_parent_0`(4c 초반 .5초 + refine_2200 이후 기본 제어기 + 기존 요격기)은 45상대·1,800경기에서 **1,293승·365무·142패**, 승률 71.83%·최악 점수 .35·하위 1/4 점수 .48333·열세 상대 3개였다. 같은 소비된 조건의 4c보다 +24승·-28패이며 새 프로필을 두 기준 모델에 대해 모두 통과했다. d8 이후 제어기를 쓴 다른 조합은 통과하지 못했다. 이는 개발 재조합 결과이며 새로운 CEM 계수 학습이나 최종 검증 결과가 아니다.

이 후보 하나를 시험 전에 고정하고 **`runs/league_separated_validate_20261007`**, 실제 PID **40404**·8작업자로 새 최종 시험을 시작했다. 기존 검증 모델·refine_2200·4c·선택 후보를 같은 45상대·각 80경기(총 14,400경기)로 비교한다. 새 band **58000000**은 개봉/사용을 시작했고 선택은 변경하지 않는다. 기존 프로필의 두 기준 모델 비교가 통과하면 같은 후보로 아직 열지 않은 시간 기반 상대 52000000의 2,560경기를 자동 실행한다. 4c와의 차이도 쌍별 불확실성·하위 성적·초기 손상으로 보고한다. 최종 흐름 검사 2개가 통과했다. PID 16576은 완료·종료됐으며 아래 실행 중 표현은 이력이다. 현재 동결 파일은 수정하지 않는다. 최신 ZIP에는 이 후속 분리 조합·최종 시험은 아직 포함되지 않는다.

**2026-10-07 03:15 KST 현재:** `runs/league_separated_probe_20261007`, 실제 PID **16576**, 작업자 8개로 후속 조합 비교를 실행 중이다. 새 `separated_probe` 정책은 4c의 첫 .5초 기동·공개 관측 게이트·요격기를 보존하고, 게이트 선택 뒤 기본 제어기만 refine_2200 또는 d8의 계수로 바꾼다. 같은 제어기를 넣은 대조 조합은 기존 4c의 실제 전투 8경기를 정확히 재현했고, 분리/초기화/내보내기 검사 2개를 통과했다. 새 두 조합×45상대×40경기=3,600경기를 **이미 사용한 선발 조건**에서 개발 비교한다. 기존 두 기준 모델의 3,600경기 기록을 재사용하며 최종·미학습 시험으로 부르지 않는다. 실행 소스·모델·입력을 동결했고 실제 저장 대결이 있다. 완료 결과로 후속 계수 학습 또는 다른 약점 보완을 결정한다. 현재 프로세스를 다시 확인할 것.

완료된 보완 학습·새 선발 평가의 최신 로컬 재현 묶음은 `experiments/league/replay_snapshot_assessment_v2_20261007.zip`(2,204,164바이트, SHA256 `a7bc2b1a983cb30b1bacc680eff25567f13e3fe0b3872b40fce5767de6389630`)이다. 별도 경로 복원에서 468파일·정책 57개·합성 행동 10,431회·검사 15개를 확인했다. 선발 원시 에피소드와 초기 라우팅 진단은 포함하지만, **그 뒤 시작한 위 분리 조합 정책과 결과는 제외**한다. ZIP 안의 README에 재현 명령과 범위가 있다. GitHub에는 올리지 않았다.

**2026-10-07 03:04 KST 완료 결과:** 새 경쟁력 평가의 선발 7,200경기가 끝났고 `no_profile_selection`으로 종료했다. 각 1,800경기에서 기존 검증 모델 972승·357무·471패, refine_2200 1,058승·490무·252패, 후보 d8 1,261승·343무·196패, 후보 4c 1,269승·361무·170패였다. 4c는 기존 검증 모델 대비 승률 +16.5%p·열세 상대 10→3으로 좋아졌으나, refine_2200보다 하위 1/4 점수가 .00625 낮아 새 기준을 통과하지 못했다. d8도 같은 하위 점수 차이와 열세 상대 1개 증가로 실패했다. 최종 56000000·미학습 52000000은 열지 않았다. 실제 PID 39060은 종료됐다. `selection_analysis.json`에 원시 상대별 요약이 있다. 이 결과로 기존 검증 모델을 교체하지 않았다.

`runs/league_routing_probe_20261007`에서 사용된 선발 초기조건을 다시 쓴 **640개 초반 장면**만 진단했다(전체 경기·독립 성능 평가 아님). 기준/4c 모두 Ace·refine_2200·gate_s2302·novel_opening_left에서는 기본 제어기, Evader·Circler·새 후보 둘에서는 요격기를 40/40번 골랐다. 다음 가설은 4c의 초반 .5초를 고정한 채 선택 후 기본 제어기를 기존 모델과 조합하는 것이다. 기존 정책·평가 기준은 보존하고, 새 정책/출력으로 먼저 조합 효과를 검사할 것. 아직 이 조합을 학습하거나 검증하지 않았다.

**2026-10-07 02:55 KST 현재:** 새 별도 평가 `runs/league_tournament_assess_20261007`를 02:54에 시작했다. 실제 PID **39060**, 작업자 **8개**와 첫 저장 대결들을 확인했다. 후보는 627승 `d8adf19d8873`, 637승 `4c4991b87895`; 기존 검증 모델과 refine_2200을 함께 비교한다. 상대 45개(이전 39개+이번 보완 후보 6개), 새 선발 7,200경기이며 통과한 후보 하나만 최종 10,800경기, 최종 통과 시 미학습 시간 상대 2,560경기로 이어진다. 최종·미학습 패널은 아직 열지 않았다. 코드·입력·환경·판정 기준 동결과 검사 12개를 확인했다. 현재 모델을 개발 성적만으로 교체하지 않았다.

속도 비교 v2는 완료했고 672개의 반복 경기 기록이 전부 같았다. 기본/사전 스레드 제한 2작업자 중앙 처리 시간은 31.44/31.22초, 제한 설정 8작업자는 10.45초였다. 0.7% 시간 차이로 실질적인 스레드 설정 속도 개선을 주장하지 않는다. 초기화된 작업자당 사유 메모리는 .842→.370GiB로 줄었다. 12·16작업자는 현재 메모리 할당 여유 검사에서 양 반복 모두 제외됐으므로 그 속도를 비교한 것이 아니다. 8작업자는 두 반복 모두 최소 할당 여유 4.24GiB를 확보했다. 현재 선택은 8개이며 새 평가 시작 후 실제 할당 여유 약 4.81GiB였다. 이는 앞선 CPU 포화 시점과 다른 실행 상태다. 근거는 `runs/league_thread_benchmark_v2_20261007/completion.json`이다.

**2026-10-07 02:52 KST:** 보완 학습은 두 난수 모두 6세대·각 49,608경기, 합계 99,216경기로 종료했다. 마지막 두 번째 후보 `4c4991b87895`는 개발 637승·72무·71패/780경기였지만 기존 상대별 보존 기준은 미통과다. 두 반복 모두 기준 모델을 유지해 선발·최종 시험을 생략했고, 시간 기반 상대 평가도 `not_eligible`로 종료했다. PID 2808·33308은 더 이상 실행 중이 아니다. `runs/league_repair_20261007/completed_searches_analysis.json`이 완료 요약이다.

기존 스레드 비교 PID 48036은 실제 경기 전에 메모리 사전 검사에서 종료했다(필요 할당 여유 8.8GiB, 종료 후 측정 약 7.2GiB). 기존 소스·계획·실패 기록을 보존했다. 별도 `tools.league_thread_benchmark_v2`를 실제 PID **36092**로 실행했다. `runs/league_thread_benchmark_v2_20261007`에서 동일 112경기의 기본/사전 스레드 제한 2작업자 비교와 제한 설정 8·12·16작업자 비교를 정순·역순으로 수행한다. 메모리가 부족한 구성은 명시적으로 건너뛰며, 추천은 양 반복에서 할당 여유 3GiB·물리 여유 2GiB를 확보한 구성 중 처리 시간이 가장 빠른 수준으로 고른다. 실제 프로세스와 완료/실패 파일을 확인하고, 완료 전 새 대결을 겹쳐 실행하지 않는다.

새 평가기 `tools.league_tournament_assess`와 지표 `tools.league_tournament_metrics`는 작성·관련 검사 완료, **아직 실제 시험 전**이다. 기존 개발에서 반복별 후보 하나씩 지명하고 기존 39상대와 보완 후보들을 새 조건에서 대결시킨다. 두 기준 모델 모두에 대해 평균·코드 그룹 균등 승률이 각각 0.5%p 이상 증가하고, 최악·하위 1/4 점수·열세 상대 수·Evader 패배가 악화되지 않아야 한다. 최종은 미리 선택한 하나만 평가하며 두 승률 차이의 초기조건 묶음 부트스트랩 95% 하한도 양수여야 한다. 기존 상대별 퇴보 제한의 통과 여부는 별도로 남긴다. 이는 새로운 목표 프로필이며 기존 탈락을 소급해 통과로 바꾸지 않는다. 선택 55010000·최종 56000000은 실행 계획 동결 전 미사용으로 확인했다. 최종 통과 때만 이전에 열지 않은 시간 기반 상대 패널을 평가한다. 코드 그룹은 실제 학생 정책 분포를 대표한다고 입증되지 않았다. 벤치마크 결과를 반영한 실행 설정으로 시작할 것. 새 검사 총 12개 통과. GitHub 업로드 금지 유지.

이 문서와 저장소의 파일을 기준으로, 이전 대화 없이 프로젝트를 파악하고 이어갈 수 있도록 정리했다.
코드 실행과 저장소 접근이 가능한 환경이 필요하다. 실행 중인 프로세스나 Python 환경 자체가 GitHub로 이동하는 것은 아니다.

**2026-10-07 02:30 KST:** 두 번째 반복의 4세대 후보 `a5a893929803`는 618승·74무·88패/780경기이며 현재 교체 기준은 미통과다. 기존 검증 모델을 유지하고 남은 세대를 진행한다. 학습 2808·평가 대기 33308·실행 설정 비교 대기 48036이 모두 실제 살아 있고 실패 파일은 없다. 627승 후보의 개발 승률 개선은 코드 구조에 따른 12개 그룹을 균등하게 가중한 기술적 비교에서도 71.64→81.14%로 나타났다. 이 그룹은 실제 대회 분포나 서로 독립된 아키텍처라고 입증된 것은 아니며, 새로운 시험 결과도 아니다.

**2026-10-07 02:16 KST 업데이트:** 두 번째 난수 2901의 첫 개발 후보 `c13252f4bd07`는 624승/780경기였지만 현재 교체 기준에서 탈락했다. 기준 모델 대비 최대 퇴보는 15%p, refine_2200 대비 최대 퇴보는 55%p였다. 같은 학습 PID 2808과 조건부 평가 대기 PID 33308을 유지한다.

스레드 설정 비교 `runs/league_thread_benchmark_20261007`를 실제 PID 48036으로 대기시켰다. 학습 2808과 평가 33308이 모두 종료하고 완료 기록이 있을 때만 시작한다. 대기 중 NumPy·PyTorch를 가져오지 않고 작업자 0개·사유 메모리 약 17.2MiB를 확인했다. 같은 공식 112경기를 기존 3작업자·사전 스레드 제한 3작업자·사전 스레드 제한 16작업자로 정순/역순 두 번 비교한다. 동일 경기 기록, 처리 시간, 작업자 사유 메모리, 시스템 할당 여유를 기록한다. 단위 검사 5개와 동결 해시 검증을 통과했으며 실제 비교 경기는 아직 시작하지 않았다. 이 비교가 끝나거나 실패 원인이 해결되기 전에는 새 학습을 겹쳐 시작하지 않는다. 현재 시제품 NumPy 로더는 이 비교에 사용하지 않는다.

평가 목표도 구분해야 한다. 첫 반복의 627승 후보는 개발에서 기준 모델보다 평균 승률(61.54→80.38%), 최악 상대 점수(.10→.30), 하위 1/4 점수(.4175→.495), 열세 상대 수(4→3)가 개선됐다. 점수는 승1·무.5·패0의 로컬 분석 정의다. 특정 상대에 대한 상대적 퇴보 제한에서 탈락한 것이며, 전체 경쟁력이 더 약하다는 뜻은 아니다. 현재 동결 기준은 유지한다. 다음 실험의 목표와 판정 기준은 이 구분을 반영해 사전 명시하고 새 조건에서 확인하며, 기존 탈락을 소급해 통과로 바꾸지 않는다. 개발 근거는 `runs/league_repair_20261007/development_weakness_alignment.json`이다. 실제 학생 정책에 대한 증거는 아직 없다.

**2026-10-07 01:56 KST 현재:** 보완 학습의 첫 난수 2900은 6세대·학습 49,608경기로 끝났고 연장하지 않았다. 개발 승수는 616→627→612/780경기였지만 세 후보 모두 상대별 보호 기준을 통과하지 못했다. 기존 검증 모델을 유지하며 같은 제어기 PID 2808·16작업자가 두 번째 난수 2901을 자동으로 시작했다. 대기 평가 PID 33308은 유지되며 최종 시험과 미학습 상대 평가는 아직 열지 않았다. 실제 상태는 프로세스와 메타데이터로 다시 확인한다. 첫 반복의 근거는 `runs/league_repair_20261007/s2900/completed_search_analysis.json`이다.

다음 실행의 자원 개선 후보도 조사했다. 현행 `Actor`는 NumPy만 쓰는 제출 정책도 PyTorch를 불러오며, 첫 3세대에서 실제 신경망 상대 2개는 합산 작업자 시간의 7.37%였다. 별도 `tools/league_numpy_matches.py` 경량 로더 시제품은 정책 33개·합성 행동 6,039회를 동일하게 재현하고 PyTorch를 불러오지 않았다. 경량 로더의 사유 메모리는 기본 스레드 설정 약 511MiB, Python 실행 전에 `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`을 지정하면 약 28MiB였다. 기존 로더에 같은 사전 스레드 설정을 적용한 합성 검사도 행동이 일치했고 사유 메모리는 약 263MiB였다. 실제 전체 학습의 절감량·배속을 측정한 것은 아니다. 현재 실행은 바꾸지 않았고, 종료 후 먼저 기존 로더의 사전 스레드 설정을 실제 경기와 시스템 할당 여유로 검증한다. 두 풀 분리는 그다음 선택 사항이다. 근거는 `memory_optimization_analysis.json`과 연결된 프로브다. 최신 ZIP에는 이 시제품과 첫 반복 완료 결과가 아직 포함되지 않는다.

**2026-10-07 01:25 KST 보완 학습 2세대 결과:** 실제 학습 PID 2808·16작업자가 계속 실행 중이다. 첫 개발 후보 `b0dc61a86f57`는 39상대·780경기에서 616승·87무·77패, 기존 검증 모델은 480승·172무·128패였다. 전체 승률은 +17.44%p지만 Evader -15%p, round_04·ddqn_s0 -30%p, novel_opening_left -15%p이며 refine_2200보다 시작 후보와의 대결에서 -55%p여서 보존 기준을 통과하지 못했다. 기존 모델을 유지하고 다음 세대를 진행한다. 완료된 2세대 학습 기록은 16,848경기이며 최종 시험은 열지 않았다. 근거는 `runs/league_repair_20261007/s2900/first_checkpoint_analysis.json`이다.

대기 평가 PID 52032는 01:13:59 Windows `ntdll.dll`·예외 `0xc000070a` 충돌로 종료됐다. 학습은 정상 진행했고, 종료 확인·기존 기록 보존·동결 해시와 환경 확인 후 같은 코드를 PID 33308로 재개했다. `recovery_20261007_0117`에 근거가 있다. 물리 RAM 여유와 별개로 01:22 시스템 전체 메모리 할당은 42.92/44.82GiB(95.75%)로 높았다. JUnit 보고서 생성의 Windows 환경 조회에서도 `0x8007000e`가 출력됐고, 보고서 생성을 빼고 재실행한 검사는 정상 통과했다. 이 두 오류의 공통 원인은 아직 확정하지 않았다. 추가 보조 시뮬레이션을 겹치지 말고 다음 작업자 수 결정에는 할당 한도 여유를 반영한다.

최신 로컬 새 실행 재현 ZIP은 `experiments/league/replay_snapshot_repair_20261007.zip`이다. 프로젝트 파일 372개·정책 51개·합성 행동 9,333회·관련 검사 13개와 복원 경로의 새 계획 생성을 확인했다. 현재 학습의 부분 체크포인트나 최종 결과는 제외했다. 자세한 범위는 묶음의 `README.md`와 `verification_report.json`에 있다. GitHub에는 올리지 않았다.

**2026-10-07 01:08 KST 현재 실행:** 후속 보완 학습 `runs/league_repair_20261007`을 01:03에 시작했다. 실제 제어기 PID 2808, 시뮬레이션 작업자 16개, 상대 39개, 난수 2900/2901이다. 개발에서 강했지만 상대별 퇴보로 탈락한 604승 후보 주변의 계수 묶음을 작게 변경하며, 기존 검증 모델과 refine_2200에 대한 보존·선발·최종 기준을 유지한다. 소프트웨어 검사 8개를 통과했고 기준 평가 후 첫 세대 선별 결과가 계속 저장되고 있다. 기본 6세대, 개발 추세에 따른 최대 8세대이며 아직 성능 결론은 없다.

사용자의 자원 점검 요청 당시에는 선행 실험 종료 후 후속 코드 준비 중이라 학습 프로세스가 없고 CPU가 4~7%였다. 실행 후 작업자 16개와 CPU 표본 5회 100%, 별도 10초 시스템 시간 측정 100%를 확인했다. 가용 메모리는 약 7~7.6GiB다. 기존 동일 96경기 비교에서 16작업자가 3작업자 대비 3.18배 빨랐지만, CPU 포화가 전체 구현의 최적성을 입증하지는 않는다. 현재 GPU 학습은 없고, 동결 실행을 유지하면서 실험 사이 공백과 다음 실행의 코드 비용을 줄이는 방향이다. 근거는 `runs/league_repair_20261007/resource_audit.json`이다.

후속 미학습 상대 평가는 `runs/league_temporal_audit_v3_20261007`, 실제 PID 52032로 PID 2808의 종료를 기다린다. 대기 중 시뮬레이션 작업자는 0개이며 앞선 최종 시험을 통과할 때만 실행한다. 아직 열지 않은 52000000 패널을 새 학습 계획에 연결해 `experiments/league/temporal_panel_repair_20261007.json`에 동결했다. 새 평가기에는 같은 제한적 Windows 쓰기 재시도와 불완전 선행 종료 처리를 적용했고 검사 4개를 통과했다. 실제 프로세스를 재확인하고 활성 진행률 파일 대신 `tools.league_live_status`의 메타데이터로 관측한다. 아래 PID·시작 전 문구는 과거 이력이다. GitHub 자동 업로드는 하지 않는다.

**2026-10-07 최신 완료 결과:** 초기조건 확대 실행은 두 난수 모두 6세대에서 종료했고 보존된 학습 경기 기록은 총 75,480개다. 새 선발 조건에서 기준 모델은 971승/1,480경기, 첫 반복 후보는 976승으로 +0.34%p여서 사전 최소 개선 +0.5%p를 통과하지 못했다. 최종 50000000과 시간 기반 상대 패널 52000000은 열지 않았고 후속 평가는 `not_eligible`로 종료했다. 학습 36160·평가 대기 52248 프로세스는 종료됐으므로 아래 실행 중 표현은 이력이다. 현재 검증 모델은 계속 `intercept_s2500_g3_c6`다.

두 번째 반복 마지막 후보 `league_conditions_20261006_d83760860544`는 개발 조건에서 604승·64무·72패/740경기로 강했지만 Evader·일부 학습 정책·초기 기동 상대에 대한 15~25%p 퇴보 때문에 보존 기준을 통과하지 못했다. 모델은 보관했으며 다음 실험은 이 후보 주변에서 일부 계수만 작게 바꿔 취약 상대를 보완하는 방향이다. 후속 학습은 아직 시작 전이다. `runs/league_conditions_20261006/completed_analysis.json`과 `s2801/completed_search_analysis.json`을 읽고 새 계획·동결 소스·출력 폴더로 구현·실행한다. 현재 모델과 기존 CEM, 전체 상대 기록을 보존하고 새 조건을 사용한다. Windows 파일 갱신 재시도도 새 드라이버와 새 평가 함수에 명시적으로 넣고 동결된 기존 소스는 수정하지 않는다.

**2026-10-07 재개 상태:** 현재 학습 제어기는 PID 36160, 후속 평가 대기는 PID 52248이다. 기존 24176·33832는 진행률 파일의 Windows 접근 오류 및 그에 따른 의존 실행 종료로 끝났고, 실패 로그를 보존한 뒤 동일한 동결 코드·계획으로 재개했다. 첫 탐색 6세대·두 번째 탐색 5세대와 저장된 6세대 대결을 재사용했고 6세대 선별 완료 후 확인 학습을 진행 중이다. 실제 프로세스를 재확인하며, 활성 진행률 JSON을 직접 반복해서 열지 말고 `python -m tools.league_live_status runs/league_conditions_20261006`의 파일 목록 기반 상태와 완료된 불변 결과를 확인한다. 아래 초기 PID·중간 경기 수는 당시 기록이다.

2026-10-06 요격기 7계수 학습과 최종 시험 완료: `intercept_s2500_g3_c6`가 같은 3,360경기에서 2,628승으로 기존 refine_2200의 2,402승, 이전 대칭 선택기의 2,571승을 넘어 사전 기준을 통과했다. 별도 8개 계수 변형 상대에서는 기존 부모와 모두 452승/640경기로 같았다. 최신 로컬 모델은 `aircombat-rl/experiments/league/candidates/intercept_s2500_g3_c6`, 기존 공개 묶음은 그대로 보존한다.
기본 기동 탐색은 큰 상대별 퇴보 때문에 선발에 실패했고, 후속 퇴보 감점 탐색도 두 번 × 6세대·학습 31,392경기에서 더 강한 보존 모델을 찾지 못했다. 두 단계 모두 최종 시험은 열지 않았다. 22:51 KST `runs/league_conditions_20261006`에서 후보를 판단하는 학습 초기조건을 늘린 실행을 시작했다. 선별은 2→4초기조건, 확인 학습은 4→12초기조건이며 양 좌석을 모두 사용한다. 기존 감점·보존 기준과 제안 분포를 유지하고 상대 37개·작업자 16개·난수 두 개로 진행한다. 현재 제어기는 시작 당시 PID 24176이다. 모델·원시 결과·물리를 보존하며 재개할 때 실제 명령줄과 완료·실패 파일을 확인한다. 같은 정책 복사본의 추가를 막고, 모든 탐색이 기준 모델만 보존한 경우 새 선발 경기를 생략한다.

초기조건 확대 실행의 첫 난수 2800은 6세대·학습 38,184경기로 완료했다. 2·4·6세대 개발 후보는 474·465·465승/740경기였으며, `a8a6ff6df4bb`의 474승·182무·84패를 보존했다. 기준 요격 모델은 같은 조건에서 463승·185무·92패였다. 6세대 개발 요청은 4세대와 같아 740경기를 재실행하지 않았다. 4→6세대 개선이 없어 연장하지 않았고 두 번째 탐색 난수 2801을 자동 시작했다. 같은 474승 후보의 계수가 이전의 다른 개발 초기조건에서는 뒤처졌으므로 최종 개선이나 학습량 확대 효과로 단정하지 않는다. 현재 최종 검증 모델은 바꾸지 않는다. 근거는 해당 실행의 `checkpoint_review.json`과 `s2800/completed_search_analysis.json`이다.

두 번째 탐색 난수 2801은 4세대·학습 24,864경기를 마쳤고 두 개발 체크포인트 모두 기존 기준 모델이 그대로 남았다. 동일 모델·동일 개발 요청의 740경기씩을 재실행하지 않았다. 두 반복 누적 학습은 63,048경기이며 다음 세대를 진행한다. 첫 반복의 474승 후보가 현재 탐색 전체의 최고 개발 후보지만 아직 최종 검증 성능은 아니다.

노트북 작업자 비교에서는 16개가 3개보다 같은 경기에서 약 3.18배 빨랐고 모든 경기 기록이 같았다. 실제 실행의 10초 CPU 표본은 100%, 가용 메모리는 6.57GiB였다. 측정 근거는 `runs/league_workers_20261006`에 있다. 이전 대기 `runs/league_maneuver_20261006`은 학습 전에 교체했으며 기록을 보존했다.
[후속 실험 보고서](aircombat-rl/experiments/league/adaptive_progress.md)와 [현재 상태](docs/LEAGUE_STATE.json)를 읽고 실제 프로세스를 확인한다. 아래 첫 리그 묶음의 완료 결과와 현재 실행을 구분한다.

GitHub 업로드는 자동화에서 제외했다. 이전 자동 업로드 승인은 철회됐으며, 사용자가 새로 명시적으로 요청할 때만 올린다.
학습·평가·분석과 로컬 결과 보관은 자동으로 계속 진행한다.

23:43 KST 별도 미학습 상대 평가를 대기열에 추가했다. `runs/league_temporal_audit_v2_20261006`, 실제 시작 PID 33832가 현재 학습 PID 24176의 종료를 기다리며 시뮬레이션 작업자는 0개다. 첫 대기 제어기 32792는 상태 파일의 Windows 접근 오류로 종료됐고 실패 기록을 보존한 채 재시도·중복 기록 방지 수정을 별도 v2에 적용했다. 현재 실험이 최종 검증을 통과할 때만 새 상대 8개와 1,920경기를 비교하고, 실패하면 평가를 건너뛴다. 상대 패널과 대기 코드·입력은 동결됐다. 초기 검사 8개·합성 행동 일치 1,600회·복구 검사 3개는 통과했고 실제 경기는 아직 열지 않았다. 자세한 범위는 `aircombat-rl/experiments/league/unseen_temporal_20261006/README.md`를 본다. 이전 재현 ZIP에는 새 평가 파일들이 아직 포함되지 않는다.

## 목표와 현재 방식

목표는 수업의 2D FairFight에서 다양한 참가자 정책을 상대하는 성능이다.
공식 초기조건·JSBSim 물리·고정 고도·20Hz·무장·행동 공간·승패 판정을 보존한다.
지금 검증한 방식은 **17개 제어 계수의 CEM 탐색**이며 신경망 DQN/PPO와 구분한다.
이전 신경망 실험에서는 중간 성능이 마지막 학습에서 무너지는 일이 반복돼 모델 보존·검증 선발을 사용했다.
이후 Ace 전용 7계수 CEM에서 여러 상대를 학습하는 리그로 확장했다.

먼저 읽을 파일:

1. [현재 상태와 다음 분기](docs/LEAGUE_STATE.json): 최종 정책, 완료 증거, 남은 작업.
2. [리그 결과 보고서](aircombat-rl/experiments/league/result.md): 상대별 결과와 한계. 생성 전이라면 상태 파일의 대기 항목을 따른다.
3. [자동 학습 방법](aircombat-rl/experiments/league/README.md): 목적함수·대결·선발 기준.
4. [설치·복원](REPRODUCE.md), [정책 묶음](aircombat-rl/experiments/league/bundle/README.md).
5. [이전 실험 요약](docs/PROGRESS_20261006.md). `HANDOFF.md`의 오래된 PID·현재 진행 표현은 당시 기록이다.

## 실제로 실행한 자동 학습

각 라운드에서 이전 우수 정책을 보존하고 후보 계수를 탐색한다. 같은 상대·같은 초기조건에서 후보를 비교하고,
훈련과 별도인 개발 경기로 모델을 선택한다. 라운드가 끝날 때 기존 챔피언과 동일 상대들에게 대결시킨다.
채택 여부와 무관하게 선택된 새 후보를 다음 라운드의 상대 목록에 추가한다. 이 과정을 6회 실행했다.
그 뒤 모든 보관 후보를 같은 전체 상대 집합으로 재선발하고, 같은 부모에서 난수 3개로 추가 학습했다.
최종 정책은 시험을 열기 전에 동결했다. 시험 결과가 더 높은 다른 모델로 바꿔 선택하지 않는다.

| 구성 | 코드·기록 |
|---|---|
| 17계수 정책과 범위 | `aircombat-rl/experiments/league/controller.py` |
| 양 좌석·동일 초기조건 대결 | `tools/league_matches.py`의 `duel`, `evaluate_jobs` |
| CEM 후보 탐색·상대 가중치·반복 | `tools/league_train.py`의 `search`, `metrics`, `run` |
| 전체 후보 재선발·3회 추가 학습·최종 시험 | `tools/league_validate.py` |
| 새 정책 파일과 직접 대결 | `tools/league_duel.py` |
| 포장·호환성 검사·보고 | `tools/league_package.py`, `league_bundle_check.py`, `league_finish.py` |
| 6라운드 소스·입력·설정 동결 | `runs/league_20261006/main/plan.json`, `source_snapshot/` |
| 추가 학습과 시험 전 선택 | `runs/league_20261006/followup_pool/frozen_selection.json` |

표의 `tools/`와 `runs/`는 `aircombat-rl/` 안의 경로다. 학습 설정과 난수 상태는 각 `search/`, `replication/s*/`에 있다.
처음부터 재실행하는 `league_train --out <새폴더>`와 최신 정책에서 후속 탐색하는 것은 다르다.
현재 `league_train` CLI의 `--rounds`를 이미 완료한 출력 폴더에 바꿔 넣어도 기존 계획의 라운드 수는 늘어나지 않는다.

## 새 컴퓨터에서 확인

2026-10-07 최신 로컬 재현 ZIP은 `aircombat-rl/experiments/league/replay_snapshot_conditions_20261006.zip`이다. 초기조건 확대 학습과 새 상대 평가 v2까지 포함한 273파일 묶음으로, 깨끗한 기준 커밋의 별도 폴더에서 복원·정책 행동 7,320회·관련 검사 18개가 통과했다. 진행 중인 학습의 부분 상태를 그대로 옮기는 묶음은 아니다. ZIP 안의 README에 새 실행과 조건부 후속 평가 명령이 있다. GitHub에는 올리지 않았다.

`REPRODUCE.md`에 따라 Python 3.12 환경과 의존성을 설치하고, 아래 명령은 `aircombat-rl`에서 실행한다.
그래프 생성에는 별도로 `python -m pip install matplotlib`이 필요하다.

```powershell
# 저장된 모델·상대 입력의 해시를 확인하고, 없는 과거 입력만 복원
python -m tools.league_replay --bundle experiments/league/bundle --restore-inputs

# 최종 모델이 포함된 묶음에서 공식 채점 경로와의 일치 확인
python -m tools.league_bundle_check --bundle experiments/league/bundle --out runs/new_machine_check/loader.json

# 실제 2경기로 실행 경로만 점검. 성능 추정용 표본이 아님
python -m tools.league_replay --bundle experiments/league/bundle --out runs/new_machine_replay --band 33010000 --n 2 --smoke

# 다른 사람의 정책과 같은 20초기조건·양 좌석으로 개발 대결
python -m tools.league_duel --a experiments/league/bundle/models/final --b path/to/other_policy --out runs/new_opponent --band 33021000 --n 40
```

같은 출력 폴더가 이미 있으면 새 이름을 쓴다. 예제 개발 band는 미사용 최종 시험으로 주장하지 않는다.
원래 CEM, 각 라운드, 추가 학습 모델과 과거 DDQN을 보존했다. 과거 신경망의 일부 재생 버퍼는 로컬 유지 목록에 있으므로
그 learner의 경험까지 그대로 재개하려면 별도 복사가 필요하다. 현재 CEM 탐색과 저장 정책 평가에는 그 버퍼가 필요하지 않다.

## 이어서 진행할 때의 분기

**같은 컴퓨터에서 아직 실행 중이면:** 상태 파일만 믿지 말고 명령줄·실행 경로가 일치하는 실제 프로세스를 확인한다.
살아 있는 `league_validate`, `league_runner_probe`, `league_finish`를 중복 실행하지 않는다. 과거 PID를 새 PC에서 재사용하지 않는다.
각 실행의 `completion.json`/`failure.json`을 읽는다. 관측 타임아웃만으로 프로세스를 재시작하지 않는다.

**최종 평가와 점검이 완료됐으면:** 완료된 6라운드나 소진된 시험을 재시작하지 않는다.
현재 결과와 취약 상대를 읽은 뒤 다음 학습 목적을 정한다. 실제 참가자 정책 확보가 가장 직접적인 상대 다양화다.

**새 학습이 필요하면:** 기존 파일을 덮어쓰지 않는 새 실험 폴더와 새 seed band를 먼저 기록한다.
`frozen_selection.json`의 `chosen`을 부모로, `opponents`와 `selected`를 합친 목록을 `unique_entrants`로 중복 제거해 출발할 수 있다.
이 목록의 과거 학습 제외 상대는 이제 관측한 상대이므로 다음 실험에서 다시 미관측 상대라고 부르지 않는다.
새 참가자 정책은 `kind="submission"`으로 정책 폴더와 가중치 경로를 기록하고 입력 해시를 동결한다.
기존 `search` API는 부모·상대 목록·학습 seed·훈련 band·개발 band·설정을 받아 추가 탐색할 수 있다.
Windows의 병렬 실행은 실제 Python 스크립트의 `if __name__ == '__main__':` 안에서 시작한다.
후속 제어 스크립트는 새 이름으로 만들고, 동결된 기존 소스를 바꿔 기존 결과인 것처럼 이어 붙이지 않는다.

각 새 후보는 이전 전체 상대 집합과 같은 조건에서 비교한다. 특정 한 정책을 이기는 것만으로 챔피언을 선언하지 않는다.
훈련 종료 후 새 개발 조건으로 선발하고, 선택을 동결한 다음에만 새 최종 조건을 연다.
시험 실패 시 이미 본 시험은 개발 근거로 남기고 새 미래 시험을 예약한다. 성공 여부와 실패한 실험 모두 보관한다.

## 평가에서 지킬 구분

- Ace 상대 200/200승과 다중 상대 결과를 섞지 않는다. 다른 학생들의 실제 정책을 모두 이긴 증거는 없다.
- 3회 미세조정은 한 학습 부모를 공유한다. 독립적인 최초 학습 3회로 표현하지 않는다.
- 양 좌석·여러 상대가 같은 초기조건 seed를 공유하므로 경기 수 전체가 독립 초기조건 수는 아니다.
- 무승부 0.5는 개발 목적함수의 규칙이며 공식 대회 배점 확정이 아니다. 무승부율과 승률을 따로 보고한다.
- Ace 시험 2000000, 이번 리그 시험 40000000은 소진됐다. 이전 여러 신경망 실험의 band도 저장된 계획에서 확인한다.
- 연속 상태의 모든 상황을 경험하거나 모든 상대를 이겼다고 표현하지 않는다. 공식 시작 조건 밖의 강건성은 별도 시험이다.

새 에이전트에 전달할 요청 예시:

> START_HERE.md와 docs/LEAGUE_STATE.json을 읽고 실제 파일·실행 상태를 확인해.
> 완료한 실험은 보존하고 현재 남은 작업부터 이어가. 새 실험은 기존 모든 정책과 비교하고,
> 개발 선발과 최종 시험을 분리해. 결과와 재실행 자료를 기록하고 업로드 상태도 구분해서 보고해.
