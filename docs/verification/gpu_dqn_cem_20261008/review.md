# Local code and architecture review

Base: 1071945. Scope: additive native GPU DQN/CEM and run-record helper, tests,
documentation. Original JSBSim, archived policy/training sources and prior GPU
PPO/physics Python sources remain unchanged.

Independent read-only review found no critical error. One important finding:
CEM accepted population=2 but filled both slots with incumbent/mean, leaving no
random challenger. A test first reproduced that the second candidate exactly
equalled the mean. The final sampler reserves the mean only when size>2.
The 64-candidate timing run is unaffected; its exact pre-fix sources are saved.

The review also requested direct DQN terminal-collection coverage. A test now
starts an actual CUDA environment just before timeout, calls the real trainer,
and confirms terminal features/done remain in replay after the environment
resets. Replay unit coverage additionally mutates input observation, next-state
and done buffers after insertion and checks copied values.

Follow-up independent review confirmed both changes and found no additional
important issue. Replay wrap, TD termination, target-copy cadence, candidate/IC/
seat mapping, elite fitting, invalid rejection and source snapshots were reviewed.

Validation: 145 tests passed, four existing warnings, 32.35 s; pip check passed.
Both policies completed original JSBSim games in both physical seats against
Ace and original CEM, on consumed open development seed 33030000. Results are
retained without a policy-quality or promotion claim. Known GPU physics fidelity
failure remains a blocker for replacing the official simulator, not for exposing
separate experimental training entrypoints.

GitHub authentication remains invalid. No remote issue, PR, push or merge.

Final reporting review matched all measurements and execution results to raw
evidence. It found a stale unported-DQN/CEM sentence in the current-state record;
that sentence now points to the new gpu_trainers status.
