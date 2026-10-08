# Local review record

Scope: new runtime audit and optional PPO CPU collection helper; frozen sources remain unchanged.

Independent code review identified callback checkpoint serialization as a blocker.
A CUDA regression first failed because collect_rollouts was saved as an instance
closure. The helper now excludes both temporary method overrides from checkpoint
data and restores the original methods. Callback save, CUDA reload, continued
training, device restoration after errors, and GPU optimizer state tests pass.
The final review found no additional important code issues.

Reporting review requires the CPU baseline beside hybrid CUDA timing, no claim
of equivalent learning trajectories, and rejection of the two-env CPU/CUDA
equal-work comparison because actual optimizer-step counts differ. Source
snapshot hashes were verified externally after that guard rejected the run.
CEM results cover fixed-policy scripted-opponent games, not the entire search.

Validation: 126 tests passed, pip check passed. Original tracked application
files match base 461284b1b046e890dab8436570ef34b71904e36b. No merge performed.
