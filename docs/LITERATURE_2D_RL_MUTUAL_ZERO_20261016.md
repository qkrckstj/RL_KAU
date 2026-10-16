# Literature review: 2D classroom simulation and mutual-zero draws

Reviewed 2026-10-16. This is a literature synthesis and proposed diagnostic sequence, not an implemented training intervention. Original rules, model archives, and admission requirements are unchanged. No GitHub publication.

## Verified local observation

Source: aircombat-rl/runs/seven_league_continue68501_20261016/iteration_000/screen/cells (s69000). Of224 games,197 are draws; all197 have exactly zero terminal health for both participants. All197 record positive head_on_damage and zero rear_damage. This does not establish identical destruction timestamps, causation, or a mathematical equilibrium. These are consumed development conditions.

The local classify function rounds terminal health to four decimals; these zeros are exact, not merely rounded equal. terminal_utility gives draws0. AimTrainingEnv adds actual damage reward0.5 and received-damage cost0.75, as well as potential terms. For a full1.0/1.0 exchange, the undiscounted damage component alone is approximately-0.25. Consequently, saying that mutual destruction has no learning penalty is incorrect. Discounted component returns, timing, and inherited-policy constraints still require diagnosis. The strict opponent-level acceptance criterion is not the same as expected per-episode return optimization.

## Sources and relevance

1. Tasbas, Sahin, Ure (2023), Reinforcement Learning Based Self-play and State Stacking Techniques for Noisy Air Combat Environment. AIAA-related DOI10.2514/6.2023-1077; author version https://arxiv.org/abs/2303.03068 ; full text https://arxiv.org/html/2303.03068v1 . Simplified planar x/y motion, DQN variants, frozen-opponent self-play and stacked noisy observations. Relevant to temporal representation and opponent diversity. It is not evidence that adding memory fixes mutual-zero outcomes in this simulator; geometry, action and reward definitions differ.

2. Ozbek and Koyuncu (2022), Reinforcement Learning based Air Combat Maneuver Generation, https://arxiv.org/abs/2201.05528 . Planar Dubins-like goal reaching with TD3 and HER. Full text describes stationary-target and obstacle tasks; two-agent dogfight is future work. Useful background for sparse goal-reaching learning, not a demonstrated competitive-air-combat solution. HER must not relabel official losses as evaluation wins.

3. Heymann, Ardema, Rajan (1984), A formulation and analysis of combat games, NASA-TM-85927, https://ntrs.nasa.gov/citations/19840014517 . Differential-game formulation with offensive objectives for both participants, distinguishing joint capture from a draw. PDF page38 discusses regions with mutual destruction and unilateral deviations leading to unilateral capture. This is a close conceptual match to mutually destructive outcomes, but not an RL study and not proof that the current policies or rules create such an equilibrium.

4. Chen et al. (2025), Autonomous Dogfight Decision-Making for Air Combat Based on Reinforcement Learning with Automatic Opponent Sampling, Aerospace12(3),265, https://doi.org/10.3390/aerospace12030265 . Full text https://mdpi-res.com/d_attachment/aerospace/aerospace-12-00265/article_deploy/aerospace-12-00265.pdf . AOS-PPO uses a policy pool, meta-solver and gate with curriculum/self-play. This is a higher-fidelity simulation study, not a verified planar match. Relevant to opponent-distribution design; its pool management/gate cannot replace the user's immutable-original-seven and fresh-confirmation admission requirements.

5. Ng, Harada, Russell (1999), Policy invariance under reward transformations: Theory and application to reward shaping, ICML, https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf . Potential-based shaping F=gamma*Phi(next)-Phi(current) can preserve optimal policies under its assumptions. Relevant to distinguishing optimization assistance from changing the actual objective. Local potential terms already use this form with terminal potential zero. This does not certify that all other local reward terms preserve the task, nor guarantee neural PPO convergence. Multi-agent extension: Lu, Schwartz, Givigi, Policy Invariance under Reward Transformations for General-Sum Stochastic Games, author version2014, https://arxiv.org/abs/1401.3907 .

6. Lanctot et al. (2017), A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning, NeurIPS, https://papers.nips.cc/paper_files/paper/2017/hash/3323fe11e9595c09af38fe67567a9394-Abstract.html . PSRO learns approximate responses to policy mixtures and derives meta-strategies from empirical payoffs. Relevant to cyclic matchups and opponent overfitting; it does not promise one policy dominates every opponent. Use ordinary per-game payoffs for game-theoretic analysis; keep the user's strict matchup acceptance statistic separate, since both sides can fail it.

7. Chow, Ghavamzadeh, Janson, Pavone (2018 journal version;2015 preprint), Risk-Constrained Reinforcement Learning with Percentile Risk Criteria, JMLR18(167),1-51, https://jmlr.org/papers/v18/15-636.html . Chance constraints and CVaR constraints target downside risk rather than expected return alone. Relevant to the user's any-loss-fails criterion, but this is not an air-combat experiment and does not guarantee zero losses. A risk constraint alone can favor all-draw policies, which still fail the user's criterion.

## Proposed diagnostic and experiment order

1. Audit native termination semantics and record first-zero timestamps for both players in selected deterministic replays. Determine whether damage is simultaneous, sequential within a decision interval, or accumulated after one health reaches zero. Compare replay and original outcomes; preserve the official rules regardless of findings until a discrepancy is established.
2. Separate both-zero draws, both-positive draws, wins and losses in reports. Report unilateral damage, received damage, existing rear/head-on metrics and component returns. Zero rear_damage is an observation, not proof of its cause.
3. Reconstruct discounted reward components and compare trajectory classes before changing weights. Existing damage cost already penalizes mutual depletion. Potential coefficients alone do not necessarily alter the ideal objective; finite-horizon handling and function approximation matter.
4. Check whether observation history and optimization anchoring prevent leaving the inherited behavior. Compare one intervention at a time in a new run; do not alter the active frozen batch. Preserve original checkpoints and exact loader interfaces.
5. Study opponent-mixture and risk-sensitive training as separate hypotheses, using training data for learning and the consumed screen only as adaptive development evidence. Avoid optimizing directly on unseen confirmation conditions.
6. Keep official224-game screens and conditional fresh448-game confirmations exactly as authorized. Mutual-zero diagnostics are additional measurements, not new win rules. No claim of universal victory or feasibility of all-opponent domination follows from these sources.

## Reading priority

For the immediate symptom: NASA joint-capture formulation, local reward audit informed by Ng et al., and opponent-distribution analysis informed by PSRO/AOS-PPO. For planar state/action representation: Tasbas et al. For strict downside-risk objectives: Chow et al. The TD3/HER goal-reaching paper is background only.
