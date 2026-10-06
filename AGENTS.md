# Project context for a new agent

Read `START_HERE.md` and `docs/LEAGUE_STATE.json` before continuing experiments.
They identify the objective, current artifacts, completed work, and next branches.
Older sections of `HANDOFF.md` are historical; their PIDs and “running” labels are not live state.

Preserve official FairFight physics, initial conditions, weapon and verdict rules, and the original CEM policy.
Keep prior policies and raw results. Separate CEM parameter search from neural DQN/PPO training in reports.
Inspect actual processes and completion/failure files before launching or resuming a run. Never duplicate a live run.
Use a new output directory for changed settings or a new experiment. Frozen source/input hashes must not be bypassed.
Both seats and shared initial conditions are required for paired policy comparisons. Retain older opponents.
Do not select a model using the final test, or reuse a consumed test as unseen evidence.
Do not infer universal tournament performance from local scripted opponents or from Ace-only results.

Run commands from `aircombat-rl` with the intended Python environment. Default local simulation concurrency is three workers.
Use `REPRODUCE.md` for installation and `experiments/league/bundle/README.md` for policy loading.
An archived Windows PID, absolute runtime path, or installed virtual environment is not portable to another machine.
Published policy weights and CEM search records are included; some historical neural replay buffers are local-only.

Update the current state record and result report after meaningful completed work.
When uploading, include model artifacts and verification evidence, and distinguish a local result from a verified remote commit.
