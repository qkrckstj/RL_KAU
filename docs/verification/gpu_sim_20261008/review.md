# Local code and architecture review

Scope: optional CUDA F-16 physics integration, tensor combat, GPU PPO, CPU policy
loader, fidelity audit and throughput evidence. Original application code and
archived policies/results remain unchanged relative to base 4364d36.

Independent review found no critical control, weapon, GAE or graph-state defect.
The reviewer identified one important packaging issue: GPU tests needed to skip
when the optional physics package is absent. Each affected test now calls
`pytest.importorskip`, while CPU scalar parity tests continue to execute.

Two review suggestions were implemented: capture is checked against all state
buffers, and fidelity was measured in actual float32 training precision in
addition to float64. Both precision runs failed the declared trajectory gate.
The known upstream coordinate/trim differences are not waived by test passage
or by four matching development-game verdicts. Official JSBSim is unchanged.

Additional regression checks cover reset-mask aliasing, target persistence,
masked physics/control reset, invalid-state quarantine, terminal observations,
timeout/kill/invalid precedence, GAE boundaries, finite changed GPU parameters,
safe checkpoint deserialization and loading through the original CPU evaluator.

Architectural boundary: the new simulator and PPO are opt-in experiments. No
archived trainer is redirected, no champion is selected, and no final test band
is opened. CEM inference is tensorized; CEM search and DQN remain unchanged.
Checkpointing is at successful completion; resumable training is not claimed.

See the retained full-suite log and pip-check output for actual validation.
GitHub authentication is invalid; publication and a remote PR remain undone.

Final independent reporting review found no remaining important blocker. It
confirmed the optional-dependency fix, capture checks and the reporting boundary
between trajectory failure, batch throughput and policy quality.
