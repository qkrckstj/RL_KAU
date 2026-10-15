# 가중치·학습 설정 변경 이력 및 모델 파일 목록

작성 시각: 2026-10-15T22:33:23+09:00

## 범위와 읽는 방법

- 현재 `rear-async/aircombat-rl/runs/`의 `rear_cpu_*`, `aim_cpu_*` 계획서 **122개**를 읽었다. 과거 다른 작업 공간의 GPU·DQN·CEM 실험 전체를 포함한 목록은 아니다.
- 추가 학습 단계가 양수인 `READY.json`이 있는 실행은 **90개**다. 이것은 체크포인트 저장 증거이며 전체 실험 완료·성공·채택을 뜻하지 않는다.
- **수동 설정 가중치**는 보상 계수, 정규화 계수, 상대 샘플링 비중 등을 뜻한다. **신경망 가중치**는 학습 과정이 갱신한 텐서 파일이다. 둘을 구분해 기록한다.
- 아래 값은 저장된 계획의 명시값이다. 키가 없으면 `미기록`으로 취급하며 0이나 라이브러리 기본값으로 추정하지 않았다. 상속되어 남은 계획 필드가 모두 실제 실행 경로에서 사용됐다는 뜻은 아니다.
- 실험은 여러 부모에서 분기했다. 표의 나열 순서는 시간상 단일 변경 순서가 아니다. 변경값은 확인 가능한 부모 체크포인트가 속한 실행의 계획과 비교했다.
- 계수의 부호와 적용 방식은 구현에 따른다. 예를 들어 양수로 저장된 패널티 계수를 양의 보상이라고 해석하지 않는다. 이 문서는 기존 기록의 목록이며 설정 추천이나 성능 개선 원인 판정은 하지 않는다.
- 기존 파일을 수정하거나 학습·평가를 실행하지 않았다. 이번 파일은 로컬에만 작성했다.

## 1. 기록된 설정값의 전체 목록

`학습 저장 증거 있음`은 해당 값이 기재된 계획 아래 양수 단계의 READY 파일이 있다는 뜻이다. 계획서만 있는 값은 별도로 표시한다.

| 설정 키 | 학습 저장 증거가 있는 실행의 값 | 그 외 계획에만 나타난 값 |
| --- | --- | --- |
| `actor_features_trainable` | `false` / `true` | — |
| `aim_coefficients.damage` | `0.5` / `0.625` | — |
| `aim_coefficients.potential` | `0.05` | — |
| `aim_coefficients.received` | `0.5` / `0.75` / `0.875` / `1.0` | — |
| `completed_focus_weight` | `0.0` | — |
| `conflict_prior_bias` | `3.0` | — |
| `control_guidance_coefficient` | `0.05` | — |
| `control_guidance_range` | `12000.0` | — |
| `curriculum.fraction` | `0.0` / `0.25` / `0.5` / `1.0` | — |
| `curriculum.stage` | `"disabled"` / `"post_encounter"` / `"quarter"` | — |
| `gradient_clipping.actor_max_norm` | `0.5` | — |
| `gradient_clipping.critic_max_norm` | `0.5` | — |
| `gradient_clipping.mode` | `"separate_actor_critic"` | — |
| `heading_feedback` | `"public_euler_heading_rate_v1"` | — |
| `learner_feature_dim` | `43` / `44` | — |
| `opening_anchor.batch_size` | `64` | — |
| `opening_anchor.coefficient` | `1.0` | — |
| `position_per_second` | `0.0` / `0.0005` / `0.001` / `0.01` / `0.02` | — |
| `ppo.batch_size` | `512` | — |
| `ppo.clip_range` | `0.05` / `0.1` | — |
| `ppo.ent_coef` | `0.0` / `0.001` / `0.003` / `0.005` / `0.006` / `0.01` | — |
| `ppo.gae_lambda` | `0.99` / `0.999` | — |
| `ppo.gamma` | `0.999` / `0.9995` | — |
| `ppo.learning_rate` | `0.0001` / `1e-05` / `1e-06` / `3e-05` / `5e-06` | — |
| `ppo.max_grad_norm` | `0.5` | — |
| `ppo.n_epochs` | `1` / `2` / `4` | — |
| `ppo.n_steps` | `640` | — |
| `ppo.target_kl` | `0.015` | — |
| `prior_bias` | `3.0` / `6.0` | — |
| `rear_damage_coefficient` | `0.0` / `0.05` / `0.1` / `0.5` | — |
| `reference_kl_coefficient` | `0.1` | — |
| `relative_enemy_feedback` | `"public_enemy_los_cos_v1"` / `"public_native_predictive_lead_cos_v1"` | — |
| `relative_potential_coefficient` | `1.0` | — |
| `speed_exploration.rule` | `"preserve_heading_marginal_temper_conditional_speed_v1"` | — |
| `speed_exploration.temperature` | `2.0` | — |
| `tail_coefficients.damage` | `0.0` / `0.125` / `0.25` / `0.5` | — |
| `tail_coefficients.hold` | `0.0` / `0.001` / `0.01` | — |
| `tail_coefficients.potential` | `0.0` / `0.01` / `0.025` / `0.25` | — |
| `tail_coefficients.received` | `0.0` / `0.25` / `0.5` | — |
| `terminal_draw_penalty` | `0.25` / `0.5` | — |

## 2. 선정된 7개 모델의 실제 가중치 파일

학습 재개용 `learner.zip`과 추론용 내보내기를 구분한다. 아래 파일의 존재와 SHA-256을 이번 작성 시점에 확인했다. 텐서의 모든 숫자를 Markdown으로 펼치거나 모델을 재저장하지 않았다.

| 모델 | 파일 역할 | 파일 | 크기(byte) | SHA-256 |
| --- | --- | --- | ---: | --- |
| `s44701_step_002621440` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_002621440/learner.zip) | 275441 | `8d638726e2e23ed46f5800df45f8fde65b15cadb81e5ab23e25e6d559076431d` |
| `s44701_step_002621440` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_002621440/greedy/policy_net.zip) | 29005 | `c252b0a3f090d414469d8dd192ef157c98cee1a23cdc973321fb157ff045f061` |
| `s44701_step_002621440` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_002621440/sampled_4900/policy_net.zip) | 29548 | `e9e99775ec6adb2b7a7b95d77c15aa2a089dd1b59a83faa31b05bbae00b5a03a` |
| `s44701_step_002621440` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_002621440/sampled_4901/policy_net.zip) | 29549 | `991a43f4e349dd3756b61138e525fc1a713fb83214c6756d94c99754b365c373` |
| `s45800_step_002621440` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_002621440/learner.zip) | 257988 | `1f3283a90447075c93531012d5e08ad563d7464c1eb2ea5a6dba5c1ea1f1b4ad` |
| `s45800_step_002621440` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_002621440/greedy/policy_net.zip) | 29912 | `ab0d87389997e1d4e21b059656b5e29a2471448766eb40490cde8a121a6c05c6` |
| `s45800_step_002621440` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_002621440/sampled_4900/policy_net.zip) | 30447 | `eb65d69639189d49bba7460cb1ef2af0d3dc98d2463fbfcf048f7044876044d9` |
| `s45800_step_002621440` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_002621440/sampled_4901/policy_net.zip) | 30447 | `0fafcb7f7ba133c9551fb4de63b1570ac305c5f0e0f0a056171c9d3ffc77c30d` |
| `s47236_step_002621440` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440/learner.zip) | 296272 | `c7b4ee33326465d3a254691db6537d7241089f9c8bfcc8efa00303db8e31d112` |
| `s47236_step_002621440` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440/greedy/policy_net.zip) | 58685 | `c0e46f016fa39b95002f8b296f7da9af4f29ca31b6c9f55fd0195ab6ff5c25c2` |
| `s47236_step_002621440` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440/sampled_4900/policy_net.zip) | 59157 | `52c1d264dcf3c0e7d39c223a2ff5e756921e0133716730d4ebdd6db721e6cd9e` |
| `s47236_step_002621440` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440/sampled_4901/policy_net.zip) | 59157 | `abab826b2e2286100bd12a236aa6083b76ac5754c41e0168a005e50979899f61` |
| `s47238_step_010485760` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_010485760/learner.zip) | 296019 | `3b4f23020661d935e5ac8562fceb78ebe145303cbb2c52d4dd23eaf1ab758ada` |
| `s47238_step_010485760` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_010485760/greedy/policy_net.zip) | 58824 | `d0b2924c6a9e77e9f57ba45b631f3b1b5002fa46da1a0a427dae5404f3c077f4` |
| `s47238_step_010485760` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_010485760/sampled_4900/policy_net.zip) | 59420 | `a9ac98f091dc65fe46b7b54c70a18ce78744dbb87b0ca4e18a5ff3fb18d2ffd2` |
| `s47238_step_010485760` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_010485760/sampled_4901/policy_net.zip) | 59420 | `d3724085e792c653c2b247cd045580beaaa0f66f7e8be05cf7b0a83e4c460e44` |
| `s46201_step_005242880` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_005242880/learner.zip) | 258255 | `e623e053f291f041bc59013c6f4e3289e14466d1c962aac6f0de94886ea76afc` |
| `s46201_step_005242880` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_005242880/greedy/policy_net.zip) | 30014 | `c05f361c04e130a93c44dc8fea16c2b544f93f5fe4d84cb0b20c4a5c336d96aa` |
| `s46201_step_005242880` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_005242880/sampled_4900/policy_net.zip) | 30544 | `7758d76c79bdef666e840da4c429008ff8ee9222e5bcfe7e74feb5304fa30cd1` |
| `s46201_step_005242880` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_005242880/sampled_4901/policy_net.zip) | 30545 | `3964ab578fde8af46826f36b80d100f6c40ce86bdc8e73972da69121894f2af1` |
| `s46401_step_005242880` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_005242880/learner.zip) | 259785 | `0ffd89f797c3c6f48caf56d7bce6fc988fecb6b36e019b56c58941b580fa6c15` |
| `s46401_step_005242880` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_005242880/greedy/policy_net.zip) | 30169 | `d54c6a83464d9e1aaa69f14f5522c7f41001d6da25fe982830f5b5ec58cdcc73` |
| `s46401_step_005242880` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_005242880/sampled_4900/policy_net.zip) | 30698 | `68539610551acf519685721f0cb9b82d5196598e771f94d2ed87b7ff5ecd7e32` |
| `s46401_step_005242880` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_005242880/sampled_4901/policy_net.zip) | 30698 | `31176a58c4735ab75d43c411271c83a3c2789f5273339c10653e48b183979d5f` |
| `s46500_step_005242880` | 학습 상태 | [learner.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_005242880/learner.zip) | 259788 | `32aefafd5c99310910ee2fd89781a9f849a2a00b336c1fabe7122c664fd0f244` |
| `s46500_step_005242880` | greedy 추론 | [greedy/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_005242880/greedy/policy_net.zip) | 30193 | `adac930a2f40c5030bcbc55c0f39e33719c9d40d14ddaa79b99c7ec541dc0107` |
| `s46500_step_005242880` | 행동 난수 4900 추론 | [sampled_4900/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_005242880/sampled_4900/policy_net.zip) | 30726 | `cc0a898f15fdbcb5dc978cc67fd78cb9fc5c825e050ff772c87e4519141f6141` |
| `s46500_step_005242880` | 행동 난수 4901 추론 | [sampled_4901/policy_net.zip](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_005242880/sampled_4901/policy_net.zip) | 30726 | `1fc3d75d11429518fbedf4e5aad32e3e427605f97df5269317ee7b9f711d09dd` |

## 3. 선정된 7개 모델의 계획에 기록된 계수

승률 순위 파일과 같은 7개 모델이다. 각 모델의 출처 계획을 연결했다. 개별 모델마다 학습 단계가 다르므로 이 표만으로 같은 학습 상태라고 볼 수 없다.

### s44701_step_002621440

출처: [rear_cpu_clip005_win585_rear12_continue_20261013/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/plan.json)

```json
{
  "ppo.learning_rate": 0.0001,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.0,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.025,
  "tail_coefficients.hold": 0.001,
  "tail_coefficients.damage": 0.5,
  "tail_coefficients.received": 0.5,
  "position_per_second": 0.001,
  "rear_damage_coefficient": 0.05,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

### s45800_step_002621440

출처: [aim_cpu_control_guidance_20261013/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/plan.json)

```json
{
  "ppo.learning_rate": 3e-05,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.003,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.0,
  "tail_coefficients.hold": 0.0,
  "tail_coefficients.damage": 0.0,
  "tail_coefficients.received": 0.0,
  "position_per_second": 0.0,
  "rear_damage_coefficient": 0.0,
  "aim_coefficients.potential": 0.05,
  "aim_coefficients.damage": 0.5,
  "aim_coefficients.received": 0.5,
  "relative_potential_coefficient": 1.0,
  "control_guidance_coefficient": 0.05,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

### s47236_step_002621440

출처: [aim_cpu_encounter_actor_v2_20261015/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/plan.json)

```json
{
  "ppo.learning_rate": 1e-06,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.003,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.0,
  "tail_coefficients.hold": 0.0,
  "tail_coefficients.damage": 0.0,
  "tail_coefficients.received": 0.0,
  "position_per_second": 0.0,
  "rear_damage_coefficient": 0.0,
  "aim_coefficients.potential": 0.05,
  "aim_coefficients.damage": 0.5,
  "aim_coefficients.received": 0.75,
  "relative_potential_coefficient": 1.0,
  "control_guidance_coefficient": 0.05,
  "control_guidance_range": 12000.0,
  "prior_bias": 6.0,
  "terminal_draw_penalty": 0.25,
  "reference_kl_coefficient": 0.1,
  "completed_focus_weight": 0.0,
  "curriculum.fraction": 0.5,
  "curriculum.stage": "post_encounter",
  "actor_features_trainable": true,
  "learner_feature_dim": 44,
  "heading_feedback": "public_euler_heading_rate_v1",
  "relative_enemy_feedback": "public_enemy_los_cos_v1",
  "opening_anchor.coefficient": 1.0,
  "opening_anchor.batch_size": 64
}
```

### s47238_step_010485760

출처: [aim_cpu_encounter_speed_20261015/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/plan.json)

```json
{
  "ppo.learning_rate": 1e-06,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.003,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.0,
  "tail_coefficients.hold": 0.0,
  "tail_coefficients.damage": 0.0,
  "tail_coefficients.received": 0.0,
  "position_per_second": 0.0,
  "rear_damage_coefficient": 0.0,
  "aim_coefficients.potential": 0.05,
  "aim_coefficients.damage": 0.5,
  "aim_coefficients.received": 0.75,
  "relative_potential_coefficient": 1.0,
  "control_guidance_coefficient": 0.05,
  "control_guidance_range": 12000.0,
  "prior_bias": 6.0,
  "terminal_draw_penalty": 0.25,
  "reference_kl_coefficient": 0.1,
  "completed_focus_weight": 0.0,
  "curriculum.fraction": 0.5,
  "curriculum.stage": "post_encounter",
  "actor_features_trainable": true,
  "learner_feature_dim": 44,
  "heading_feedback": "public_euler_heading_rate_v1",
  "relative_enemy_feedback": "public_enemy_los_cos_v1",
  "opening_anchor.coefficient": 1.0,
  "opening_anchor.batch_size": 64,
  "speed_exploration.temperature": 2.0,
  "speed_exploration.rule": "preserve_heading_marginal_temper_conditional_speed_v1"
}
```

### s46201_step_005242880

출처: [aim_cpu_state_prior_20261013/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/plan.json)

```json
{
  "ppo.learning_rate": 3e-05,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.003,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.0,
  "tail_coefficients.hold": 0.0,
  "tail_coefficients.damage": 0.0,
  "tail_coefficients.received": 0.0,
  "position_per_second": 0.0,
  "rear_damage_coefficient": 0.0,
  "aim_coefficients.potential": 0.05,
  "aim_coefficients.damage": 0.5,
  "aim_coefficients.received": 0.5,
  "relative_potential_coefficient": 1.0,
  "control_guidance_coefficient": 0.05,
  "control_guidance_range": 12000.0,
  "prior_bias": 6.0,
  "conflict_prior_bias": 3.0,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

### s46401_step_005242880

출처: [aim_cpu_rate_input_20261014/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/plan.json)

```json
{
  "ppo.learning_rate": 3e-05,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.003,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.0,
  "tail_coefficients.hold": 0.0,
  "tail_coefficients.damage": 0.0,
  "tail_coefficients.received": 0.0,
  "position_per_second": 0.0,
  "rear_damage_coefficient": 0.0,
  "aim_coefficients.potential": 0.05,
  "aim_coefficients.damage": 0.5,
  "aim_coefficients.received": 0.5,
  "relative_potential_coefficient": 1.0,
  "control_guidance_coefficient": 0.05,
  "control_guidance_range": 12000.0,
  "prior_bias": 6.0,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5,
  "learner_feature_dim": 44,
  "heading_feedback": "public_euler_heading_rate_v1"
}
```

### s46500_step_005242880

출처: [aim_cpu_heading_memory_20261014/plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/plan.json)

```json
{
  "ppo.learning_rate": 3e-05,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.05,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.003,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.0,
  "tail_coefficients.hold": 0.0,
  "tail_coefficients.damage": 0.0,
  "tail_coefficients.received": 0.0,
  "position_per_second": 0.0,
  "rear_damage_coefficient": 0.0,
  "aim_coefficients.potential": 0.05,
  "aim_coefficients.damage": 0.5,
  "aim_coefficients.received": 0.5,
  "relative_potential_coefficient": 1.0,
  "control_guidance_coefficient": 0.05,
  "control_guidance_range": 12000.0,
  "prior_bias": 6.0,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5,
  "learner_feature_dim": 44,
  "heading_feedback": "public_euler_heading_rate_v1"
}
```

## 4. 실행별 부모 대비 설정 변경 및 저장 모델 목록

신경망 내부 수치 변경분이 아닌 **계획의 설정 차이**다. 부모 출처를 확인할 수 없으면 임의의 앞 실행과 비교하지 않았다. READY 목록은 현재 발견한 추가 학습 체크포인트를 모두 적었다.

### aim_cpu_actor_head_only_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/plan.json)
- 계획 SHA-256: `eb762987ced5e5183db729d226231b635c1793b1cdf9acebf4a9cbf5ef0b3aec`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **14개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `actor_features_trainable` | `"미기록"` | `false` |
| `reference_kl_coefficient` | `"미기록"` | `0.1` |

저장된 추가 학습 모델:

- [s47080_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_000655360/READY.json).
- [s47080_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_001310720/READY.json).
- [s47080_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_001966080/READY.json).
- [s47080_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_002621440/READY.json).
- [s47080_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_003276800/READY.json).
- [s47080_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47080_step_003932160/READY.json).
- [s47081_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_000655360/READY.json).
- [s47081_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_001310720/READY.json).
- [s47081_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_001966080/READY.json).
- [s47081_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_002621440/READY.json).
- [s47081_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003276800/READY.json).
- [s47081_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160/READY.json).
- [s47081_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_004587520/READY.json).
- [s47081_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_005242880/READY.json).

### aim_cpu_best498_continue_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/plan.json)
- 계획 SHA-256: `9b5f6f1cda40cc9c9441d0a18eba515848295a02d6cb66b85aeda7d510b5e3aa`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s46000_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_000655360/READY.json).
- [s46000_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_001310720/READY.json).
- [s46000_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_001966080/READY.json).
- [s46000_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_002621440/READY.json).
- [s46000_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_003276800/READY.json).
- [s46000_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_003932160/READY.json).
- [s46000_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_004587520/READY.json).
- [s46000_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46000_step_005242880/READY.json).
- [s46001_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_000655360/READY.json).
- [s46001_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_001310720/READY.json).
- [s46001_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_001966080/READY.json).
- [s46001_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_002621440/READY.json).
- [s46001_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_003276800/READY.json).
- [s46001_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_003932160/READY.json).
- [s46001_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_004587520/READY.json).
- [s46001_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_best498_continue_20261013/checkpoints/s46001_step_005242880/READY.json).

### aim_cpu_contact_anchor_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/plan.json)
- 계획 SHA-256: `2ef1a85e0a05cf7beac98b3303c449e120cadb950a6ecacdbaca441882c7b015`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47235_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_000655360/READY.json).
- [s47235_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_001310720/READY.json).
- [s47235_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_001966080/READY.json).
- [s47235_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_002621440/READY.json).
- [s47235_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_003276800/READY.json).
- [s47235_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_003932160/READY.json).
- [s47235_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_004587520/READY.json).
- [s47235_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_005242880/READY.json).
- [s47235_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_005898240/READY.json).
- [s47235_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_006553600/READY.json).
- [s47235_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_007208960/READY.json).
- [s47235_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_007864320/READY.json).
- [s47235_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_008519680/READY.json).
- [s47235_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_009175040/READY.json).
- [s47235_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_009830400/READY.json).
- [s47235_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/checkpoints/s47235_step_010485760/READY.json).

### aim_cpu_control_guidance_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/plan.json)
- 계획 SHA-256: `442db363fc84d2a481dc7923b87c3557a1d0316cfaf80be43c2a9df4458b88c2`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000000000`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_predictive_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `control_guidance_coefficient` | `"미기록"` | `0.05` |
| `curriculum.fraction` | `0.0` | `0.5` |
| `curriculum.stage` | `"disabled"` | `"quarter"` |
| `ppo.ent_coef` | `0.0` | `0.003` |
| `relative_potential_coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s45800_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_000655360/READY.json).
- [s45800_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_001310720/READY.json).
- [s45800_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_001966080/READY.json).
- [s45800_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_002621440/READY.json).
- [s45800_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_003276800/READY.json).
- [s45800_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_003932160/READY.json).
- [s45800_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_004587520/READY.json).
- [s45800_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45800_step_005242880/READY.json).
- [s45801_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_000655360/READY.json).
- [s45801_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_001310720/READY.json).
- [s45801_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_001966080/READY.json).
- [s45801_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_002621440/READY.json).
- [s45801_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_003276800/READY.json).
- [s45801_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_003932160/READY.json).
- [s45801_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_004587520/READY.json).
- [s45801_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_control_guidance_20261013/checkpoints/s45801_step_005242880/READY.json).

### aim_cpu_current_anchor504_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/plan.json)
- 계획 SHA-256: `ac1dcb9402a9c5aa4a326f60eaac55c45671a5c14a8926655ee4f5ed1c019e5d`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_actor_head_only_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47100_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_000655360/READY.json).
- [s47100_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_001310720/READY.json).
- [s47100_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_001966080/READY.json).
- [s47100_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_002621440/READY.json).
- [s47100_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_003276800/READY.json).
- [s47100_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47100_step_003932160/READY.json).
- [s47101_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_000655360/READY.json).
- [s47101_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_001310720/READY.json).
- [s47101_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_001966080/READY.json).
- [s47101_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_002621440/READY.json).
- [s47101_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_003276800/READY.json).
- [s47101_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor504_20261014/checkpoints/s47101_step_003932160/READY.json).

### aim_cpu_current_anchor508_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/plan.json)
- 계획 SHA-256: `d3bd19e5067099645ee3b704e4bfbafd79352476f21f97c05e0eaae4ac8a2295`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001310720`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_head_rate5e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47130_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47130_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47130_step_000655360/READY.json).
- [s47130_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47130_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47130_step_001310720/READY.json).
- [s47130_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47130_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47130_step_001966080/READY.json).
- [s47131_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47131_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47131_step_000655360/READY.json).
- [s47131_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47131_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47131_step_001310720/READY.json).
- [s47131_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47131_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_current_anchor508_20261014/checkpoints/s47131_step_001966080/READY.json).

### aim_cpu_dealt0625_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/plan.json)
- 계획 SHA-256: `874ada6f7a9f175c1815416b174eedb840beda4191f07d2cedc884706f6cefcc`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.damage` | `0.5` | `0.625` |

저장된 추가 학습 모델:

- [s47180_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47180_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47180_step_000655360/READY.json).
- [s47180_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47180_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47180_step_001310720/READY.json).
- [s47180_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47180_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47180_step_001966080/READY.json).
- [s47181_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47181_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47181_step_000655360/READY.json).
- [s47181_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47181_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47181_step_001310720/READY.json).
- [s47181_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47181_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_dealt0625_20261014/checkpoints/s47181_step_001966080/READY.json).

### aim_cpu_draw_peak_assessment_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_peak_assessment_20261014/plan.json)
- 계획 SHA-256: `5560d03e47a7668314f25b487e90587eb6ea8a397441c975dee9d6b17e2b29f5`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `heading_feedback` | `"미기록"` | `"public_euler_heading_rate_v1"` |
| `learner_feature_dim` | `"미기록"` | `43` |
| `prior_bias` | `"미기록"` | `6.0` |
| `terminal_draw_penalty` | `"미기록"` | `0.25` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_draw_strength050_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/plan.json)
- 계획 SHA-256: `063cd102e5e669b93d09e84583ba385369b59d10b7add20444db7aa23d1e7c99`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `3e-05` | `1e-05` |
| `relative_enemy_feedback` | `"미기록"` | `"public_native_predictive_lead_cos_v1"` |
| `terminal_draw_penalty` | `0.25` | `0.5` |

저장된 추가 학습 모델:

- [s46900_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_000655360/READY.json).
- [s46900_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_001310720/READY.json).
- [s46900_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_001966080/READY.json).
- [s46900_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_002621440/READY.json).
- [s46900_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_003276800/READY.json).
- [s46900_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_003932160/READY.json).
- [s46900_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_004587520/READY.json).
- [s46900_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46900_step_005242880/READY.json).
- [s46901_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_000655360/READY.json).
- [s46901_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_001310720/READY.json).
- [s46901_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_001966080/READY.json).
- [s46901_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_002621440/READY.json).
- [s46901_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_003276800/READY.json).
- [s46901_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_003932160/READY.json).
- [s46901_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_004587520/READY.json).
- [s46901_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_strength050_20261014/checkpoints/s46901_step_005242880/READY.json).

### aim_cpu_draw_utility025_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_20261014/plan.json)
- 계획 SHA-256: `26dbbdee2d7c8fde124ab17ad944b846395f2a782a7c8084c068e5bdfe05e293`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `heading_feedback` | `"미기록"` | `"public_euler_heading_rate_v1"` |
| `learner_feature_dim` | `"미기록"` | `43` |
| `prior_bias` | `"미기록"` | `6.0` |
| `terminal_draw_penalty` | `"미기록"` | `0.25` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_draw_utility025_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/plan.json)
- 계획 SHA-256: `5560d03e47a7668314f25b487e90587eb6ea8a397441c975dee9d6b17e2b29f5`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `heading_feedback` | `"미기록"` | `"public_euler_heading_rate_v1"` |
| `learner_feature_dim` | `"미기록"` | `43` |
| `prior_bias` | `"미기록"` | `6.0` |
| `terminal_draw_penalty` | `"미기록"` | `0.25` |

저장된 추가 학습 모델:

- [s46610_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_000655360/READY.json).
- [s46610_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_001310720/READY.json).
- [s46610_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_001966080/READY.json).
- [s46610_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_002621440/READY.json).
- [s46610_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003276800/READY.json).
- [s46610_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160/READY.json).
- [s46610_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_004587520/READY.json).
- [s46610_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_005242880/READY.json).
- [s46611_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_000655360/READY.json).
- [s46611_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_001310720/READY.json).
- [s46611_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_001966080/READY.json).
- [s46611_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_002621440/READY.json).
- [s46611_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_003276800/READY.json).
- [s46611_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_003932160/READY.json).
- [s46611_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_004587520/READY.json).
- [s46611_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46611_step_005242880/READY.json).

### aim_cpu_early59_complete_qualification_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_early59_complete_qualification_20261014/plan.json)
- 계획 SHA-256: `71f5b93650a81260aee00cda94b4bb50a27b139b985fea105570286fc4480ae0`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_actor_head_only_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `1e-05` | `5e-06` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_early59_qualification_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_early59_qualification_20261014/plan.json)
- 계획 SHA-256: `cb552573a1fafe156ceafd9bf29de953958979b8740d9a7a42fd2caea60c47d2`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_actor_head_only_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `1e-05` | `5e-06` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_encounter_actor_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_20261015/plan.json)
- 계획 SHA-256: `b2c72e5ab65f719f21a20a25e3927ad7ce5ff03c9e527a5b81f9cd92a7ef7989`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `learner_feature_dim` | `43` | `44` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_encounter_actor_v2_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/plan.json)
- 계획 SHA-256: `0912cfd2c9f023ac60a134c086cfc352b95b2136558b64630667d5a13fc1b7de`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `learner_feature_dim` | `43` | `44` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47236_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_000655360/READY.json).
- [s47236_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_001310720/READY.json).
- [s47236_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_001966080/READY.json).
- [s47236_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440/READY.json).
- [s47236_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_003276800/READY.json).
- [s47236_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_003932160/READY.json).
- [s47236_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_004587520/READY.json).
- [s47236_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_005242880/READY.json).
- [s47236_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_005898240/READY.json).
- [s47236_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_006553600/READY.json).
- [s47236_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_007208960/READY.json).
- [s47236_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_007864320/READY.json).
- [s47236_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_008519680/READY.json).
- [s47236_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_009175040/READY.json).
- [s47236_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_009830400/READY.json).
- [s47236_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_010485760/READY.json).

### aim_cpu_encounter_loss_focus_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/plan.json)
- 계획 SHA-256: `6b8af82cc6a7420e6669ddc884aeaf1dcb1b7489ac92a68caffd12f20601ab80`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P3**.

부모 실행 `aim_cpu_encounter_actor_v2_20261015`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47237_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_000655360/READY.json).
- [s47237_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_001310720/READY.json).
- [s47237_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_001966080/READY.json).
- [s47237_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_002621440/READY.json).
- [s47237_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_003276800/READY.json).
- [s47237_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_003932160/READY.json).
- [s47237_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_004587520/READY.json).
- [s47237_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_005242880/READY.json).
- [s47237_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_005898240/READY.json).
- [s47237_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_006553600/READY.json).
- [s47237_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_007208960/READY.json).
- [s47237_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_007864320/READY.json).
- [s47237_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_008519680/READY.json).
- [s47237_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_009175040/READY.json).
- [s47237_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_009830400/READY.json).
- [s47237_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/checkpoints/s47237_step_010485760/READY.json).

### aim_cpu_encounter_speed_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/plan.json)
- 계획 SHA-256: `d7e61a71a64332478958016a81d8b7150641c6e8dc82c797a3a0f8fdf4224172`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_encounter_actor_v2_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `speed_exploration.rule` | `"미기록"` | `"preserve_heading_marginal_temper_conditional_speed_v1"` |
| `speed_exploration.temperature` | `"미기록"` | `2.0` |

저장된 추가 학습 모델:

- [s47238_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_000655360/READY.json).
- [s47238_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_001310720/READY.json).
- [s47238_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_001966080/READY.json).
- [s47238_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_002621440/READY.json).
- [s47238_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_003276800/READY.json).
- [s47238_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_003932160/READY.json).
- [s47238_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_004587520/READY.json).
- [s47238_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_005242880/READY.json).
- [s47238_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_005898240/READY.json).
- [s47238_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_006553600/READY.json).
- [s47238_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_007208960/READY.json).
- [s47238_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_007864320/READY.json).
- [s47238_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_008519680/READY.json).
- [s47238_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_009175040/READY.json).
- [s47238_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_009830400/READY.json).
- [s47238_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015/checkpoints/s47238_step_010485760/READY.json).

### aim_cpu_encounter_speed_20261015_unlaunched_band_collision

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_speed_20261015_unlaunched_band_collision/plan.json)
- 계획 SHA-256: `147d76aa3001488995ec360713c51d6e803c303e609034db9a8da676ab98eaf9`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_actor_v2_20261015/checkpoints/s47236_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_encounter_actor_v2_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `speed_exploration.rule` | `"미기록"` | `"preserve_heading_marginal_temper_conditional_speed_v1"` |
| `speed_exploration.temperature` | `"미기록"` | `2.0` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_enemy_los_control_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/plan.json)
- 계획 SHA-256: `aff66b4ca43bbafa0a25853b70081af3f4535aaa7ed64f275f48de4c29ee2537`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `3e-05` | `1e-05` |
| `relative_enemy_feedback` | `"미기록"` | `"public_enemy_los_cos_v1"` |

저장된 추가 학습 모델:

- [s47000_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_000655360/READY.json).
- [s47000_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_001310720/READY.json).
- [s47000_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_001966080/READY.json).
- [s47000_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_002621440/READY.json).
- [s47000_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_003276800/READY.json).
- [s47000_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_003932160/READY.json).
- [s47000_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_004587520/READY.json).
- [s47000_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47000_step_005242880/READY.json).
- [s47001_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_000655360/READY.json).
- [s47001_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_001310720/READY.json).
- [s47001_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_001966080/READY.json).
- [s47001_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_002621440/READY.json).
- [s47001_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_003276800/READY.json).
- [s47001_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_003932160/READY.json).
- [s47001_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_004587520/READY.json).
- [s47001_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_enemy_los_control_20261014/checkpoints/s47001_step_005242880/READY.json).

### aim_cpu_entropy001_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/plan.json)
- 계획 SHA-256: `f524129abecaff71e5cd311d521696b1b657fcfeff7546408de9f2ea55a007bf`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.003` | `0.001` |

저장된 추가 학습 모델:

- [s47050_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_000655360/READY.json).
- [s47050_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_001310720/READY.json).
- [s47050_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_001966080/READY.json).
- [s47050_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_002621440/READY.json).
- [s47050_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_003276800/READY.json).
- [s47050_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47050_step_003932160/READY.json).
- [s47051_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_000655360/READY.json).
- [s47051_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_001310720/READY.json).
- [s47051_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_001966080/READY.json).
- [s47051_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_002621440/READY.json).
- [s47051_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_003276800/READY.json).
- [s47051_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy001_20261014/checkpoints/s47051_step_003932160/READY.json).

### aim_cpu_entropy003_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/plan.json)
- 계획 SHA-256: `ad429afad7c00c30a1aaad3512cde6e37b4b5d19e6739abc39773d94a50c9295`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000000000`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_predictive_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.0` | `0.5` |
| `curriculum.stage` | `"disabled"` | `"quarter"` |
| `ppo.ent_coef` | `0.0` | `0.003` |
| `relative_potential_coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s45600_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_000655360/READY.json).
- [s45600_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_001310720/READY.json).
- [s45600_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_001966080/READY.json).
- [s45600_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_002621440/READY.json).
- [s45600_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_003276800/READY.json).
- [s45600_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_003932160/READY.json).
- [s45600_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_004587520/READY.json).
- [s45600_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45600_step_005242880/READY.json).
- [s45601_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_000655360/READY.json).
- [s45601_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_001310720/READY.json).
- [s45601_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_001966080/READY.json).
- [s45601_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_002621440/READY.json).
- [s45601_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_003276800/READY.json).
- [s45601_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_003932160/READY.json).
- [s45601_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_004587520/READY.json).
- [s45601_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy003_20261013/checkpoints/s45601_step_005242880/READY.json).

### aim_cpu_entropy005_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_20261014/plan.json)
- 계획 SHA-256: `7abbdf513483824fa8ef6c0bbe63f679fafac63f068b5fcd8701fe229bfa3c22`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.003` | `0.005` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_entropy005_resume_v2_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/plan.json)
- 계획 SHA-256: `d7c5210757759fc3ce3009d9fbb05da5f98df877d6f13a2686c6f4f8671e9efa`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.003` | `0.005` |

저장된 추가 학습 모델:

- [s47192_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47192_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47192_step_000655360/READY.json).
- [s47192_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47192_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47192_step_001310720/READY.json).
- [s47192_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47192_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47192_step_001966080/READY.json).
- [s47193_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47193_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47193_step_000655360/READY.json).
- [s47193_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47193_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47193_step_001310720/READY.json).
- [s47193_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47193_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_entropy005_resume_v2_20261014/checkpoints/s47193_step_001966080/READY.json).

### aim_cpu_feature_train1e6_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/plan.json)
- 계획 SHA-256: `c4efd9dade26a01472d609bbb0335c9577350ee80667cfa575e6fee7f168cd85`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001310720`.
- 추가 학습 READY: **9개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_head_rate5e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `actor_features_trainable` | `false` | `true` |
| `ppo.learning_rate` | `5e-06` | `1e-06` |

저장된 추가 학습 모델:

- [s47150_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_000655360/READY.json).
- [s47150_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001310720/READY.json).
- [s47150_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080/READY.json).
- [s47150_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_002621440/READY.json).
- [s47150_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_003276800/READY.json).
- [s47150_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_003932160/READY.json).
- [s47151_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47151_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47151_step_000655360/READY.json).
- [s47151_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47151_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47151_step_001310720/READY.json).
- [s47151_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47151_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47151_step_001966080/READY.json).

### aim_cpu_guidance_range12k_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/plan.json)
- 계획 SHA-256: `620bff996e2bf1b9168a9b44e99ed99b9aff33d4f4e374990b9d0d7712190703`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000000000`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_predictive_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `control_guidance_coefficient` | `"미기록"` | `0.05` |
| `control_guidance_range` | `"미기록"` | `12000.0` |
| `curriculum.fraction` | `0.0` | `0.5` |
| `curriculum.stage` | `"disabled"` | `"quarter"` |
| `ppo.ent_coef` | `0.0` | `0.003` |
| `relative_potential_coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s45900_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_000655360/READY.json).
- [s45900_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_001310720/READY.json).
- [s45900_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_001966080/READY.json).
- [s45900_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_002621440/READY.json).
- [s45900_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_003276800/READY.json).
- [s45900_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_003932160/READY.json).
- [s45900_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_004587520/READY.json).
- [s45900_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_005242880/READY.json).
- [s45900_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_005898240/READY.json).
- [s45900_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_006553600/READY.json).
- [s45900_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_007208960/READY.json).
- [s45900_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_007864320/READY.json).
- [s45900_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_008519680/READY.json).
- [s45900_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_009175040/READY.json).
- [s45900_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_009830400/READY.json).
- [s45900_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760/READY.json).
- [s45901_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_000655360/READY.json).
- [s45901_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_001310720/READY.json).
- [s45901_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_001966080/READY.json).
- [s45901_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_002621440/READY.json).
- [s45901_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_003276800/READY.json).
- [s45901_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_003932160/READY.json).
- [s45901_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_004587520/READY.json).
- [s45901_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45901_step_005242880/READY.json).

### aim_cpu_head_rate1e6_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/plan.json)
- 계획 SHA-256: `75b76f23da41428c9261014a9e457107e5315a19444762240815acfb20c7bba7`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001310720`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_head_rate5e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `5e-06` | `1e-06` |

저장된 추가 학습 모델:

- [s47140_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47140_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47140_step_000655360/READY.json).
- [s47140_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47140_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47140_step_001310720/READY.json).
- [s47140_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47140_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47140_step_001966080/READY.json).
- [s47141_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47141_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47141_step_000655360/READY.json).
- [s47141_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47141_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47141_step_001310720/READY.json).
- [s47141_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47141_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate1e6_20261014/checkpoints/s47141_step_001966080/READY.json).

### aim_cpu_head_rate5e6_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/plan.json)
- 계획 SHA-256: `e83e2429e07886ece59a7c446d7707fbcd5db864baed52b331266ef0252bbe1f`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_actor_head_only_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `1e-05` | `5e-06` |

저장된 추가 학습 모델:

- [s47110_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_000655360/READY.json).
- [s47110_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_001310720/READY.json).
- [s47110_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_001966080/READY.json).
- [s47110_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_002621440/READY.json).
- [s47110_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_003276800/READY.json).
- [s47110_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47110_step_003932160/READY.json).
- [s47111_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_000655360/READY.json).
- [s47111_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001310720/READY.json).
- [s47111_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001966080/READY.json).
- [s47111_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_002621440/READY.json).
- [s47111_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_003276800/READY.json).
- [s47111_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_003932160/READY.json).

### aim_cpu_head_trace_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_trace_20261015/plan.json)
- 계획 SHA-256: `5846a62cb5f8f255b5cd420f137a81d91fe6ea68fef55b2de67be026f251f4aa`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **0개**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{}
```

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_heading_memory_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/plan.json)
- 계획 SHA-256: `1616788ba1885fce1b1e6f187902b8835af4550e07106bafc90765ba8248239b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `heading_feedback` | `"미기록"` | `"public_euler_heading_rate_v1"` |
| `learner_feature_dim` | `"미기록"` | `44` |
| `prior_bias` | `"미기록"` | `6.0` |

저장된 추가 학습 모델:

- [s46500_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_000655360/READY.json).
- [s46500_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_001310720/READY.json).
- [s46500_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_001966080/READY.json).
- [s46500_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_002621440/READY.json).
- [s46500_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_003276800/READY.json).
- [s46500_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_003932160/READY.json).
- [s46500_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_004587520/READY.json).
- [s46500_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46500_step_005242880/READY.json).
- [s46501_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_000655360/READY.json).
- [s46501_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_001310720/READY.json).
- [s46501_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_001966080/READY.json).
- [s46501_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_002621440/READY.json).
- [s46501_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_003276800/READY.json).
- [s46501_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_003932160/READY.json).
- [s46501_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_004587520/READY.json).
- [s46501_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_memory_20261014/checkpoints/s46501_step_005242880/READY.json).

### aim_cpu_heading_rate_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/plan.json)
- 계획 SHA-256: `4cfe7d61f281868b426b9f534ba024feecb1a61c2baeeccbe6016afa8d27e6b6`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `heading_feedback` | `"미기록"` | `"public_euler_heading_rate_v1"` |
| `prior_bias` | `"미기록"` | `6.0` |

저장된 추가 학습 모델:

- [s46300_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_000655360/READY.json).
- [s46300_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_001310720/READY.json).
- [s46300_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_001966080/READY.json).
- [s46300_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_002621440/READY.json).
- [s46300_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_003276800/READY.json).
- [s46300_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_003932160/READY.json).
- [s46300_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_004587520/READY.json).
- [s46300_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46300_step_005242880/READY.json).
- [s46301_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_000655360/READY.json).
- [s46301_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_001310720/READY.json).
- [s46301_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_001966080/READY.json).
- [s46301_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_002621440/READY.json).
- [s46301_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_003276800/READY.json).
- [s46301_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_003932160/READY.json).
- [s46301_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_004587520/READY.json).
- [s46301_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_heading_rate_20261013/checkpoints/s46301_step_005242880/READY.json).

### aim_cpu_known_focus_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/plan.json)
- 계획 SHA-256: `e2e4626fb2ea5ec514e826fd65e9d8dd1fef338a3bc42c216b0ffb2322459b2c`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P4**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47222_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_000655360/READY.json).
- [s47222_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_001310720/READY.json).
- [s47222_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_001966080/READY.json).
- [s47222_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_002621440/READY.json).
- [s47222_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_003276800/READY.json).
- [s47222_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_003932160/READY.json).
- [s47222_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_004587520/READY.json).
- [s47222_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_005242880/READY.json).
- [s47222_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_005898240/READY.json).
- [s47222_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_006553600/READY.json).
- [s47222_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_007208960/READY.json).
- [s47222_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_007864320/READY.json).
- [s47222_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_008519680/READY.json).
- [s47222_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_009175040/READY.json).
- [s47222_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_009830400/READY.json).
- [s47222_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/checkpoints/s47222_step_010485760/READY.json).

### aim_cpu_known_long2_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/plan.json)
- 계획 SHA-256: `5a76f76d771ee5c1767c09e192b1a209b823c2ae2dfb46193285abdd64b9e937`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **64개**.
- 상대 샘플링 비중: **P5**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47221_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_000655360/READY.json).
- [s47221_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_001310720/READY.json).
- [s47221_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_001966080/READY.json).
- [s47221_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_002621440/READY.json).
- [s47221_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_003276800/READY.json).
- [s47221_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_003932160/READY.json).
- [s47221_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_004587520/READY.json).
- [s47221_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_005242880/READY.json).
- [s47221_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_005898240/READY.json).
- [s47221_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_006553600/READY.json).
- [s47221_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_007208960/READY.json).
- [s47221_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_007864320/READY.json).
- [s47221_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_008519680/READY.json).
- [s47221_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_009175040/READY.json).
- [s47221_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_009830400/READY.json).
- [s47221_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_010485760/READY.json).
- [s47221_step_011141120](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_011141120) — 추가 11,141,120단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_011141120/READY.json).
- [s47221_step_011796480](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_011796480) — 추가 11,796,480단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_011796480/READY.json).
- [s47221_step_012451840](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_012451840) — 추가 12,451,840단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_012451840/READY.json).
- [s47221_step_013107200](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_013107200) — 추가 13,107,200단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_013107200/READY.json).
- [s47221_step_013762560](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_013762560) — 추가 13,762,560단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_013762560/READY.json).
- [s47221_step_014417920](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_014417920) — 추가 14,417,920단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_014417920/READY.json).
- [s47221_step_015073280](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_015073280) — 추가 15,073,280단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_015073280/READY.json).
- [s47221_step_015728640](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_015728640) — 추가 15,728,640단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_015728640/READY.json).
- [s47221_step_016384000](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_016384000) — 추가 16,384,000단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_016384000/READY.json).
- [s47221_step_017039360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_017039360) — 추가 17,039,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_017039360/READY.json).
- [s47221_step_017694720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_017694720) — 추가 17,694,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_017694720/READY.json).
- [s47221_step_018350080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_018350080) — 추가 18,350,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_018350080/READY.json).
- [s47221_step_019005440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_019005440) — 추가 19,005,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_019005440/READY.json).
- [s47221_step_019660800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_019660800) — 추가 19,660,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_019660800/READY.json).
- [s47221_step_020316160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_020316160) — 추가 20,316,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_020316160/READY.json).
- [s47221_step_020971520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_020971520) — 추가 20,971,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_020971520/READY.json).
- [s47221_step_021626880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_021626880) — 추가 21,626,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_021626880/READY.json).
- [s47221_step_022282240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_022282240) — 추가 22,282,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_022282240/READY.json).
- [s47221_step_022937600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_022937600) — 추가 22,937,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_022937600/READY.json).
- [s47221_step_023592960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_023592960) — 추가 23,592,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_023592960/READY.json).
- [s47221_step_024248320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_024248320) — 추가 24,248,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_024248320/READY.json).
- [s47221_step_024903680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_024903680) — 추가 24,903,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_024903680/READY.json).
- [s47221_step_025559040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_025559040) — 추가 25,559,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_025559040/READY.json).
- [s47221_step_026214400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_026214400) — 추가 26,214,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_026214400/READY.json).
- [s47221_step_026869760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_026869760) — 추가 26,869,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_026869760/READY.json).
- [s47221_step_027525120](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_027525120) — 추가 27,525,120단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_027525120/READY.json).
- [s47221_step_028180480](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_028180480) — 추가 28,180,480단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_028180480/READY.json).
- [s47221_step_028835840](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_028835840) — 추가 28,835,840단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_028835840/READY.json).
- [s47221_step_029491200](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_029491200) — 추가 29,491,200단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_029491200/READY.json).
- [s47221_step_030146560](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_030146560) — 추가 30,146,560단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_030146560/READY.json).
- [s47221_step_030801920](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_030801920) — 추가 30,801,920단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_030801920/READY.json).
- [s47221_step_031457280](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_031457280) — 추가 31,457,280단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_031457280/READY.json).
- [s47221_step_032112640](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_032112640) — 추가 32,112,640단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_032112640/READY.json).
- [s47221_step_032768000](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_032768000) — 추가 32,768,000단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_032768000/READY.json).
- [s47221_step_033423360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_033423360) — 추가 33,423,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_033423360/READY.json).
- [s47221_step_034078720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_034078720) — 추가 34,078,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_034078720/READY.json).
- [s47221_step_034734080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_034734080) — 추가 34,734,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_034734080/READY.json).
- [s47221_step_035389440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_035389440) — 추가 35,389,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_035389440/READY.json).
- [s47221_step_036044800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_036044800) — 추가 36,044,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_036044800/READY.json).
- [s47221_step_036700160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_036700160) — 추가 36,700,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_036700160/READY.json).
- [s47221_step_037355520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_037355520) — 추가 37,355,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_037355520/READY.json).
- [s47221_step_038010880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_038010880) — 추가 38,010,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_038010880/READY.json).
- [s47221_step_038666240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_038666240) — 추가 38,666,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_038666240/READY.json).
- [s47221_step_039321600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_039321600) — 추가 39,321,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_039321600/READY.json).
- [s47221_step_039976960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_039976960) — 추가 39,976,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_039976960/READY.json).
- [s47221_step_040632320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_040632320) — 추가 40,632,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_040632320/READY.json).
- [s47221_step_041287680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_041287680) — 추가 41,287,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_041287680/READY.json).
- [s47221_step_041943040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_041943040) — 추가 41,943,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/checkpoints/s47221_step_041943040/READY.json).

### aim_cpu_known_long_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/plan.json)
- 계획 SHA-256: `6d94cb1b01449fd0d035b77517b12579771756a10d30ceb6af607fcd3f1d0321`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P5**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47220_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_000655360/READY.json).
- [s47220_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_001310720/READY.json).
- [s47220_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_001966080/READY.json).
- [s47220_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_002621440/READY.json).
- [s47220_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_003276800/READY.json).
- [s47220_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_003932160/READY.json).
- [s47220_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_004587520/READY.json).
- [s47220_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_005242880/READY.json).
- [s47220_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_005898240/READY.json).
- [s47220_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_006553600/READY.json).
- [s47220_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_007208960/READY.json).
- [s47220_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_007864320/READY.json).
- [s47220_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_008519680/READY.json).
- [s47220_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_009175040/READY.json).
- [s47220_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_009830400/READY.json).
- [s47220_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_010485760/READY.json).
- [s47220_step_011141120](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_011141120) — 추가 11,141,120단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_011141120/READY.json).
- [s47220_step_011796480](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_011796480) — 추가 11,796,480단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_011796480/READY.json).
- [s47220_step_012451840](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_012451840) — 추가 12,451,840단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_012451840/READY.json).
- [s47220_step_013107200](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_013107200) — 추가 13,107,200단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_013107200/READY.json).
- [s47220_step_013762560](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_013762560) — 추가 13,762,560단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_013762560/READY.json).
- [s47220_step_014417920](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_014417920) — 추가 14,417,920단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_014417920/READY.json).
- [s47220_step_015073280](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_015073280) — 추가 15,073,280단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_015073280/READY.json).
- [s47220_step_015728640](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_015728640) — 추가 15,728,640단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_015728640/READY.json).
- [s47220_step_016384000](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_016384000) — 추가 16,384,000단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_016384000/READY.json).
- [s47220_step_017039360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_017039360) — 추가 17,039,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_017039360/READY.json).
- [s47220_step_017694720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_017694720) — 추가 17,694,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_017694720/READY.json).
- [s47220_step_018350080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_018350080) — 추가 18,350,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_018350080/READY.json).
- [s47220_step_019005440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_019005440) — 추가 19,005,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_019005440/READY.json).
- [s47220_step_019660800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_019660800) — 추가 19,660,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_019660800/READY.json).
- [s47220_step_020316160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_020316160) — 추가 20,316,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_020316160/READY.json).
- [s47220_step_020971520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_020971520) — 추가 20,971,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long_20261014/checkpoints/s47220_step_020971520/READY.json).

### aim_cpu_known_mutual_focus_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/plan.json)
- 계획 SHA-256: `14e4870ee095f2aa64173436df7d7d83e8b46056269c62724399e6b7ca8ad4d4`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47223_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_000655360/READY.json).
- [s47223_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_001310720/READY.json).
- [s47223_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_001966080/READY.json).
- [s47223_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_002621440/READY.json).
- [s47223_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_003276800/READY.json).
- [s47223_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_003932160/READY.json).
- [s47223_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_004587520/READY.json).
- [s47223_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_005242880/READY.json).
- [s47223_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_005898240/READY.json).
- [s47223_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_006553600/READY.json).
- [s47223_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_007208960/READY.json).
- [s47223_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_007864320/READY.json).
- [s47223_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_008519680/READY.json).
- [s47223_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_009175040/READY.json).
- [s47223_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_009830400/READY.json).
- [s47223_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760/READY.json).

### aim_cpu_long_credit0999_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/plan.json)
- 계획 SHA-256: `a25d4413bc98528ca18a5d81cb40bf72540a3bd90de83273f58f75a0fa9cda0f`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.gae_lambda` | `0.99` | `0.999` |
| `reference_kl_coefficient` | `"미기록"` | `0.1` |

저장된 추가 학습 모델:

- [s47070_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_000655360/READY.json).
- [s47070_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_001310720/READY.json).
- [s47070_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_001966080/READY.json).
- [s47070_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_002621440/READY.json).
- [s47070_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_003276800/READY.json).
- [s47070_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47070_step_003932160/READY.json).
- [s47071_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_000655360/READY.json).
- [s47071_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_001310720/READY.json).
- [s47071_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_001966080/READY.json).
- [s47071_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_002621440/READY.json).
- [s47071_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_003276800/READY.json).
- [s47071_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_credit0999_20261014/checkpoints/s47071_step_003932160/READY.json).

### aim_cpu_long_horizon_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_20261015/plan.json)
- 계획 SHA-256: `fbdf87452a5a7baa6d7db242b581f99956781c4ea4647e643805c119f7bcb587`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.gamma` | `0.999` | `0.9995` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_long_horizon_v2_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/plan.json)
- 계획 SHA-256: `db3d009a5f08976f1d06db8d8833e550c119ba968e7e36438cfa5230d90e82b8`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.gamma` | `0.999` | `0.9995` |

저장된 추가 학습 모델:

- [s47227_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_000655360/READY.json).
- [s47227_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_001310720/READY.json).
- [s47227_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_001966080/READY.json).
- [s47227_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_002621440/READY.json).
- [s47227_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_003276800/READY.json).
- [s47227_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_003932160/READY.json).
- [s47227_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_004587520/READY.json).
- [s47227_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_005242880/READY.json).
- [s47227_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_005898240/READY.json).
- [s47227_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_006553600/READY.json).
- [s47227_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_007208960/READY.json).
- [s47227_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_007864320/READY.json).
- [s47227_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_008519680/READY.json).
- [s47227_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_009175040/READY.json).
- [s47227_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_009830400/READY.json).
- [s47227_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_long_horizon_v2_20261015/checkpoints/s47227_step_010485760/READY.json).

### aim_cpu_los_regression_trace_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_los_regression_trace_20261014/plan.json)
- 계획 SHA-256: `976f367c52bd839704865f89c2d43f67616e58c365e7ab466009d1a09fcf8a29`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **0개**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{}
```

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_mixed_start_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/plan.json)
- 계획 SHA-256: `b3b7fb8fe21b4b7aaf9d59939e1e5f27bedeea0d39a49e5009a9201163d7146f`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003932160`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_lr3e5_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.damage` | `"미기록"` | `0.5` |
| `aim_coefficients.potential` | `"미기록"` | `0.05` |
| `aim_coefficients.received` | `"미기록"` | `0.5` |
| `position_per_second` | `0.001` | `0.0` |
| `rear_damage_coefficient` | `0.05` | `0.0` |
| `tail_coefficients.damage` | `0.5` | `0.0` |
| `tail_coefficients.hold` | `0.001` | `0.0` |
| `tail_coefficients.potential` | `0.025` | `0.0` |
| `tail_coefficients.received` | `0.5` | `0.0` |

저장된 추가 학습 모델:

- [s45400_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_000655360/READY.json).
- [s45400_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_001310720/READY.json).
- [s45400_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_001966080/READY.json).
- [s45400_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_002621440/READY.json).
- [s45400_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_003276800/READY.json).
- [s45400_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_003932160/READY.json).
- [s45400_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_004587520/READY.json).
- [s45400_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45400_step_005242880/READY.json).
- [s45401_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_000655360/READY.json).
- [s45401_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_001310720/READY.json).
- [s45401_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_001966080/READY.json).
- [s45401_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_002621440/READY.json).
- [s45401_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_003276800/READY.json).
- [s45401_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_003932160/READY.json).
- [s45401_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_004587520/READY.json).
- [s45401_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_20261013/checkpoints/s45401_step_005242880/READY.json).

### aim_cpu_mixed_start_early393_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mixed_start_early393_20261013/plan.json)
- 계획 SHA-256: `b3b7fb8fe21b4b7aaf9d59939e1e5f27bedeea0d39a49e5009a9201163d7146f`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003932160`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_lr3e5_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.damage` | `"미기록"` | `0.5` |
| `aim_coefficients.potential` | `"미기록"` | `0.05` |
| `aim_coefficients.received` | `"미기록"` | `0.5` |
| `position_per_second` | `0.001` | `0.0` |
| `rear_damage_coefficient` | `0.05` | `0.0` |
| `tail_coefficients.damage` | `0.5` | `0.0` |
| `tail_coefficients.hold` | `0.001` | `0.0` |
| `tail_coefficients.potential` | `0.025` | `0.0` |
| `tail_coefficients.received` | `0.5` | `0.0` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_mode_probe_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mode_probe_20261014/plan.json)
- 계획 SHA-256: `462399c67ea4806631f8cc666603efcd5f8b6e67faa3e111d74f73ff4f9c2f8a`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **0개**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{}
```

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_mutual_continue_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/plan.json)
- 계획 SHA-256: `e57c803cea7751fc1fbc24df3baaa9d7263c731c2d887c0bb3b01ef0cb0e7412`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47224_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_000655360/READY.json).
- [s47224_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_001310720/READY.json).
- [s47224_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_001966080/READY.json).
- [s47224_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_002621440/READY.json).
- [s47224_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_003276800/READY.json).
- [s47224_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_003932160/READY.json).
- [s47224_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_004587520/READY.json).
- [s47224_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_005242880/READY.json).
- [s47224_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_005898240/READY.json).
- [s47224_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_006553600/READY.json).
- [s47224_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_007208960/READY.json).
- [s47224_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_007864320/READY.json).
- [s47224_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_008519680/READY.json).
- [s47224_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_009175040/READY.json).
- [s47224_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_009830400/READY.json).
- [s47224_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_continue_20261015/checkpoints/s47224_step_010485760/READY.json).

### aim_cpu_mutual_head_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/plan.json)
- 계획 SHA-256: `dd134079125388e7d1fd003a9f363d96f17cbbc5af84b32b1a93a93de649a8ed`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `actor_features_trainable` | `true` | `false` |

저장된 추가 학습 모델:

- [s47225_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_000655360/READY.json).
- [s47225_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_001310720/READY.json).
- [s47225_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_001966080/READY.json).
- [s47225_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_002621440/READY.json).
- [s47225_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_003276800/READY.json).
- [s47225_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_003932160/READY.json).
- [s47225_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_004587520/READY.json).
- [s47225_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_005242880/READY.json).
- [s47225_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_005898240/READY.json).
- [s47225_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_006553600/READY.json).
- [s47225_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_007208960/READY.json).
- [s47225_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_007864320/READY.json).
- [s47225_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_008519680/READY.json).
- [s47225_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_009175040/READY.json).
- [s47225_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_009830400/READY.json).
- [s47225_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_head_20261015/checkpoints/s47225_step_010485760/READY.json).

### aim_cpu_mutual_replica_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/plan.json)
- 계획 SHA-256: `459e06c08698bb900653e7c3192b2627d7a1439b12148f1ee67731ad47b0a2fe`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47226_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_000655360/READY.json).
- [s47226_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_001310720/READY.json).
- [s47226_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_001966080/READY.json).
- [s47226_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_002621440/READY.json).
- [s47226_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_003276800/READY.json).
- [s47226_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_003932160/READY.json).
- [s47226_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_004587520/READY.json).
- [s47226_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_005242880/READY.json).
- [s47226_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_005898240/READY.json).
- [s47226_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_006553600/READY.json).
- [s47226_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_007208960/READY.json).
- [s47226_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_007864320/READY.json).
- [s47226_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_008519680/READY.json).
- [s47226_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_009175040/READY.json).
- [s47226_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_009830400/READY.json).
- [s47226_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_replica_20261015/checkpoints/s47226_step_010485760/READY.json).

### aim_cpu_mutual_trace_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_mutual_trace_20261015/plan.json)
- 계획 SHA-256: `b7bc4b64f7058b414219db066a9fbfddfe1ba6c0286655cac9d079c8012686e5`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **0개**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{}
```

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_opening_coverage_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/plan.json)
- 계획 SHA-256: `9670b6f5d9c07ed6dd86674c120987cc0bb1da3f71358522c11d6dd2a5d9d868`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47232_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_000655360/READY.json).
- [s47232_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_001310720/READY.json).
- [s47232_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_001966080/READY.json).
- [s47232_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_002621440/READY.json).
- [s47232_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_003276800/READY.json).
- [s47232_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_003932160/READY.json).
- [s47232_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_004587520/READY.json).
- [s47232_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_005242880/READY.json).
- [s47232_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_005898240/READY.json).
- [s47232_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_006553600/READY.json).
- [s47232_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_007208960/READY.json).
- [s47232_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_007864320/READY.json).
- [s47232_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_008519680/READY.json).
- [s47232_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_009175040/READY.json).
- [s47232_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_009830400/READY.json).
- [s47232_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_coverage_20261015/checkpoints/s47232_step_010485760/READY.json).

### aim_cpu_opening_rehearsal_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/plan.json)
- 계획 SHA-256: `1b6075ec859ac7ecf3b4c1495f6b7cf89b14b8bb72f6e821c73959c2cd1b8fe6`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47229_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_000655360/READY.json).
- [s47229_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_001310720/READY.json).
- [s47229_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_001966080/READY.json).
- [s47229_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_002621440/READY.json).
- [s47229_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_003276800/READY.json).
- [s47229_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_003932160/READY.json).
- [s47229_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_004587520/READY.json).
- [s47229_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_005242880/READY.json).
- [s47229_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_005898240/READY.json).
- [s47229_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_006553600/READY.json).
- [s47229_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_007208960/READY.json).
- [s47229_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_007864320/READY.json).
- [s47229_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_008519680/READY.json).
- [s47229_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_009175040/READY.json).
- [s47229_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_009830400/READY.json).
- [s47229_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_opening_rehearsal_20261015/checkpoints/s47229_step_010485760/READY.json).

### aim_cpu_post_encounter_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_20261015/plan.json)
- 계획 SHA-256: `a67d5d7a2ba7cee1ee1e3ccf04fe9221e56525aacd4b0f4723e239b569f5f7b7`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_post_encounter_v2_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/plan.json)
- 계획 SHA-256: `d669c872b33070312f6f13f1dac66cce413eae348e4d33950ab79ebfb5083c64`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |

저장된 추가 학습 모델:

- [s47228_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_000655360/READY.json).
- [s47228_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_001310720/READY.json).
- [s47228_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_001966080/READY.json).
- [s47228_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_002621440/READY.json).
- [s47228_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_003276800/READY.json).
- [s47228_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_003932160/READY.json).
- [s47228_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_004587520/READY.json).
- [s47228_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_005242880/READY.json).
- [s47228_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_005898240/READY.json).
- [s47228_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_006553600/READY.json).
- [s47228_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_007208960/READY.json).
- [s47228_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_007864320/READY.json).
- [s47228_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_008519680/READY.json).
- [s47228_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_009175040/READY.json).
- [s47228_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_009830400/READY.json).
- [s47228_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_post_encounter_v2_20261015/checkpoints/s47228_step_010485760/READY.json).

### aim_cpu_predictive_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/plan.json)
- 계획 SHA-256: `265742ddb6235ad91960f60d7c20338314a6c63c54503464979ea2f64a234764`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003932160`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_lr3e5_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.damage` | `"미기록"` | `0.5` |
| `aim_coefficients.potential` | `"미기록"` | `0.05` |
| `aim_coefficients.received` | `"미기록"` | `0.5` |
| `curriculum.fraction` | `0.5` | `0.0` |
| `curriculum.stage` | `"quarter"` | `"disabled"` |
| `position_per_second` | `0.001` | `0.0` |
| `rear_damage_coefficient` | `0.05` | `0.0` |
| `tail_coefficients.damage` | `0.5` | `0.0` |
| `tail_coefficients.hold` | `0.001` | `0.0` |
| `tail_coefficients.potential` | `0.025` | `0.0` |
| `tail_coefficients.received` | `0.5` | `0.0` |

저장된 추가 학습 모델:

- [s45300_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000655360/READY.json).
- [s45300_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_001310720/READY.json).
- [s45300_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_001966080/READY.json).
- [s45300_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_002621440/READY.json).
- [s45300_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_003276800/READY.json).
- [s45300_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_003932160/READY.json).
- [s45300_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_004587520/READY.json).
- [s45300_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_005242880/READY.json).
- [s45301_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_000655360/READY.json).
- [s45301_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_001310720/READY.json).
- [s45301_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_001966080/READY.json).
- [s45301_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_002621440/READY.json).
- [s45301_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_003276800/READY.json).
- [s45301_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_003932160/READY.json).
- [s45301_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_004587520/READY.json).
- [s45301_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45301_step_005242880/READY.json).

### aim_cpu_predictive_enemy_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/plan.json)
- 계획 SHA-256: `c46edd2ab6ef2d4c5580d8ef67c15f9c2274be6bba54264d208beeb256c83d8e`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `relative_enemy_feedback` | `"미기록"` | `"public_native_predictive_lead_cos_v1"` |

저장된 추가 학습 모델:

- [s46700_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_000655360/READY.json).
- [s46700_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_001310720/READY.json).
- [s46700_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_001966080/READY.json).
- [s46700_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_002621440/READY.json).
- [s46700_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_003276800/READY.json).
- [s46700_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_003932160/READY.json).
- [s46700_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_004587520/READY.json).
- [s46700_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46700_step_005242880/READY.json).
- [s46701_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_000655360/READY.json).
- [s46701_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_001310720/READY.json).
- [s46701_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_001966080/READY.json).
- [s46701_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_002621440/READY.json).
- [s46701_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_003276800/READY.json).
- [s46701_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_003932160/READY.json).
- [s46701_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_004587520/READY.json).
- [s46701_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_20261014/checkpoints/s46701_step_005242880/READY.json).

### aim_cpu_predictive_enemy_lr1e5_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_20261014/plan.json)
- 계획 SHA-256: `76f5e9514e9c1e6ddbdca70fabb2c9b227b80d8b1cfeb5f848d3cfb47a1f2f6b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `3e-05` | `1e-05` |
| `relative_enemy_feedback` | `"미기록"` | `"public_native_predictive_lead_cos_v1"` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_predictive_enemy_lr1e5_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/plan.json)
- 계획 SHA-256: `a897a2ea60a662c44c08e52edcbc71aef4e871cc8ed29192d64e7412759ebe1e`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `3e-05` | `1e-05` |
| `relative_enemy_feedback` | `"미기록"` | `"public_native_predictive_lead_cos_v1"` |

저장된 추가 학습 모델:

- [s46810_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_000655360/READY.json).
- [s46810_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_001310720/READY.json).
- [s46810_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_001966080/READY.json).
- [s46810_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_002621440/READY.json).
- [s46810_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_003276800/READY.json).
- [s46810_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_003932160/READY.json).
- [s46810_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_004587520/READY.json).
- [s46810_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46810_step_005242880/READY.json).
- [s46811_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_000655360/READY.json).
- [s46811_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_001310720/READY.json).
- [s46811_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_001966080/READY.json).
- [s46811_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_002621440/READY.json).
- [s46811_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_003276800/READY.json).
- [s46811_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_003932160/READY.json).
- [s46811_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_004587520/READY.json).
- [s46811_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_enemy_lr1e5_resume_20261014/checkpoints/s46811_step_005242880/READY.json).

### aim_cpu_prefix_diversity_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/plan.json)
- 계획 SHA-256: `a2aae839bf08d25bb00ef19c6b6959d53553693ce2edfbe487218e59783690d1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47230_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_000655360/READY.json).
- [s47230_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_001310720/READY.json).
- [s47230_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_001966080/READY.json).
- [s47230_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_002621440/READY.json).
- [s47230_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_003276800/READY.json).
- [s47230_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_003932160/READY.json).
- [s47230_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_004587520/READY.json).
- [s47230_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_005242880/READY.json).
- [s47230_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_005898240/READY.json).
- [s47230_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_006553600/READY.json).
- [s47230_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_007208960/READY.json).
- [s47230_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_007864320/READY.json).
- [s47230_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_008519680/READY.json).
- [s47230_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_009175040/READY.json).
- [s47230_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_009830400/READY.json).
- [s47230_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prefix_diversity_20261015/checkpoints/s47230_step_010485760/READY.json).

### aim_cpu_prior3_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/plan.json)
- 계획 SHA-256: `c4177d26a7c91ba03de22a90bae1207a6a09145cd23808f329e7ef3eac0cf7bf`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `prior_bias` | `"미기록"` | `3.0` |

저장된 추가 학습 모델:

- [s46100_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_000655360/READY.json).
- [s46100_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_001310720/READY.json).
- [s46100_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_001966080/READY.json).
- [s46100_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_002621440/READY.json).
- [s46100_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_003276800/READY.json).
- [s46100_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_003932160/READY.json).
- [s46100_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_004587520/READY.json).
- [s46100_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46100_step_005242880/READY.json).
- [s46101_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_000655360/READY.json).
- [s46101_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_001310720/READY.json).
- [s46101_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_001966080/READY.json).
- [s46101_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_002621440/READY.json).
- [s46101_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_003276800/READY.json).
- [s46101_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_003932160/READY.json).
- [s46101_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_004587520/READY.json).
- [s46101_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_prior3_20261013/checkpoints/s46101_step_005242880/READY.json).

### aim_cpu_rate_input_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/plan.json)
- 계획 SHA-256: `a0b29da37c47f3a86c48944d61b6b099900c15d35f0ca675b6aa785bb72412ed`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `heading_feedback` | `"미기록"` | `"public_euler_heading_rate_v1"` |
| `learner_feature_dim` | `"미기록"` | `44` |
| `prior_bias` | `"미기록"` | `6.0` |

저장된 추가 학습 모델:

- [s46400_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_000655360/READY.json).
- [s46400_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_001310720/READY.json).
- [s46400_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_001966080/READY.json).
- [s46400_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_002621440/READY.json).
- [s46400_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_003276800/READY.json).
- [s46400_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_003932160/READY.json).
- [s46400_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_004587520/READY.json).
- [s46400_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46400_step_005242880/READY.json).
- [s46401_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_000655360/READY.json).
- [s46401_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_001310720/READY.json).
- [s46401_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_001966080/READY.json).
- [s46401_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_002621440/READY.json).
- [s46401_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_003276800/READY.json).
- [s46401_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_003932160/READY.json).
- [s46401_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_004587520/READY.json).
- [s46401_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_rate_input_20261014/checkpoints/s46401_step_005242880/READY.json).

### aim_cpu_received075_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_20261014/plan.json)
- 계획 SHA-256: `5971816edf352b6dd07ea0e34312952d296155f0ddf985cde0d71a21b73b8a15`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.received` | `0.5` | `0.75` |
| `ppo.learning_rate` | `3e-05` | `1e-05` |
| `relative_enemy_feedback` | `"미기록"` | `"public_enemy_los_cos_v1"` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_received075_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/plan.json)
- 계획 SHA-256: `5cc57c838e168a043f415afa06b94e9d2f8729c4cd4b744f70579911b9d1b562`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_draw_utility025_resume_20261014/checkpoints/s46610_step_003932160`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_draw_utility025_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.received` | `0.5` | `0.75` |
| `ppo.learning_rate` | `3e-05` | `1e-05` |
| `relative_enemy_feedback` | `"미기록"` | `"public_enemy_los_cos_v1"` |

저장된 추가 학습 모델:

- [s47020_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_000655360/READY.json).
- [s47020_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_001310720/READY.json).
- [s47020_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_001966080/READY.json).
- [s47020_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_002621440/READY.json).
- [s47020_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_003276800/READY.json).
- [s47020_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_003932160/READY.json).
- [s47020_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_004587520/READY.json).
- [s47020_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47020_step_005242880/READY.json).
- [s47021_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_000655360/READY.json).
- [s47021_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_001310720/READY.json).
- [s47021_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_001966080/READY.json).
- [s47021_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_002621440/READY.json).
- [s47021_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_003276800/READY.json).
- [s47021_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_003932160/READY.json).
- [s47021_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_004587520/READY.json).
- [s47021_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880/READY.json).
- [s47021_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005898240/READY.json).
- [s47021_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_006553600/READY.json).
- [s47021_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_007208960/READY.json).
- [s47021_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_007864320/READY.json).
- [s47021_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_008519680/READY.json).
- [s47021_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_009175040/READY.json).
- [s47021_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_009830400/READY.json).
- [s47021_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_010485760/READY.json).

### aim_cpu_received0875_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/plan.json)
- 계획 SHA-256: `ddfa0a93426762917498e5614dcdeecc03b55530a05ba06e7d173a32a229d979`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.received` | `0.75` | `0.875` |

저장된 추가 학습 모델:

- [s47170_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47170_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47170_step_000655360/READY.json).
- [s47170_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47170_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47170_step_001310720/READY.json).
- [s47170_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47170_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47170_step_001966080/READY.json).
- [s47171_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47171_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47171_step_000655360/READY.json).
- [s47171_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47171_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47171_step_001310720/READY.json).
- [s47171_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47171_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received0875_20261014/checkpoints/s47171_step_001966080/READY.json).

### aim_cpu_received1_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/plan.json)
- 계획 SHA-256: `a9edc426ebcc6687507e874d29194e5754233a3a5a7488b46736019f20c3de96`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000000000`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_predictive_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `aim_coefficients.received` | `0.5` | `1.0` |
| `curriculum.fraction` | `0.0` | `0.5` |
| `curriculum.stage` | `"disabled"` | `"quarter"` |
| `ppo.ent_coef` | `0.0` | `0.003` |
| `relative_potential_coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s45700_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_000655360/READY.json).
- [s45700_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_001310720/READY.json).
- [s45700_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_001966080/READY.json).
- [s45700_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_002621440/READY.json).
- [s45700_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_003276800/READY.json).
- [s45700_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_003932160/READY.json).
- [s45700_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_004587520/READY.json).
- [s45700_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45700_step_005242880/READY.json).
- [s45701_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_000655360/READY.json).
- [s45701_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_001310720/READY.json).
- [s45701_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_001966080/READY.json).
- [s45701_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_002621440/READY.json).
- [s45701_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_003276800/READY.json).
- [s45701_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_003932160/READY.json).
- [s45701_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_004587520/READY.json).
- [s45701_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received1_20261013/checkpoints/s45701_step_005242880/READY.json).

### aim_cpu_reengagement_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/plan.json)
- 계획 SHA-256: `895b51970d2651cb0cb7a685458ad03b9951ec9a9b10613c329eced9a39f00c1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47231_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_000655360/READY.json).
- [s47231_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_001310720/READY.json).
- [s47231_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_001966080/READY.json).
- [s47231_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_002621440/READY.json).
- [s47231_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_003276800/READY.json).
- [s47231_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_003932160/READY.json).
- [s47231_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_004587520/READY.json).
- [s47231_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_005242880/READY.json).
- [s47231_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_005898240/READY.json).
- [s47231_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_006553600/READY.json).
- [s47231_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_007208960/READY.json).
- [s47231_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_007864320/READY.json).
- [s47231_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_008519680/READY.json).
- [s47231_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_009175040/READY.json).
- [s47231_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_009830400/READY.json).
- [s47231_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_20261015/checkpoints/s47231_step_010485760/READY.json).

### aim_cpu_reengagement_focus_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/plan.json)
- 계획 SHA-256: `14e35e4589bb5ec738a2b7134dcd39604d604e4f45ed18561089613d97e9baf1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.5` | `1.0` |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47233_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_000655360/READY.json).
- [s47233_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_001310720/READY.json).
- [s47233_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_001966080/READY.json).
- [s47233_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_002621440/READY.json).
- [s47233_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_003276800/READY.json).
- [s47233_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_003932160/READY.json).
- [s47233_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_004587520/READY.json).
- [s47233_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_005242880/READY.json).
- [s47233_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_005898240/READY.json).
- [s47233_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_006553600/READY.json).
- [s47233_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_007208960/READY.json).
- [s47233_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_007864320/READY.json).
- [s47233_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_008519680/READY.json).
- [s47233_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_009175040/READY.json).
- [s47233_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_009830400/READY.json).
- [s47233_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reengagement_focus_20261015/checkpoints/s47233_step_010485760/READY.json).

### aim_cpu_reference510_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_20261014/plan.json)
- 계획 SHA-256: `0fd2af4a75d8167bf8a25d836d69c2b3efefa398f2c0e47dc26511f449f23718`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_reference510_v2_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/plan.json)
- 계획 SHA-256: `c940678a204659d918fc0333262c26fd8478e0dc1d7ff14a1c96779e96e0375e`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47202_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47202_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47202_step_000655360/READY.json).
- [s47202_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47202_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47202_step_001310720/READY.json).
- [s47202_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47202_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47202_step_001966080/READY.json).
- [s47203_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47203_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47203_step_000655360/READY.json).
- [s47203_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47203_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47203_step_001310720/READY.json).
- [s47203_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47203_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference510_v2_20261014/checkpoints/s47203_step_001966080/READY.json).

### aim_cpu_reference_anchor01_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/plan.json)
- 계획 SHA-256: `dfe53907c0da507d7868ba114f3ef05492c8421948a75a9ee01381f7f578fe4b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `reference_kl_coefficient` | `"미기록"` | `0.1` |

저장된 추가 학습 모델:

- [s47060_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_000655360/READY.json).
- [s47060_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_001310720/READY.json).
- [s47060_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_001966080/READY.json).
- [s47060_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_002621440/READY.json).
- [s47060_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_003276800/READY.json).
- [s47060_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47060_step_003932160/READY.json).
- [s47061_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_000655360/READY.json).
- [s47061_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_001310720/READY.json).
- [s47061_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_001966080/READY.json).
- [s47061_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_002621440/READY.json).
- [s47061_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_003276800/READY.json).
- [s47061_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_anchor01_20261014/checkpoints/s47061_step_003932160/READY.json).

### aim_cpu_reference_midpoint_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_reference_midpoint_20261014/plan.json)
- 계획 SHA-256: `a26847fd35d597f3e42a5765a49f6efe35522252cf93cf947f0bc0b46c074e89`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `reference_kl_coefficient` | `"미기록"` | `0.1` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_relative_reward_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/plan.json)
- 계획 SHA-256: `cafe8449aa46a139d3ba802e521e504e530ec615244284a6c4736bd43405229b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_predictive_20261013/checkpoints/s45300_step_000000000`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_predictive_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.0` | `0.5` |
| `curriculum.stage` | `"disabled"` | `"quarter"` |
| `relative_potential_coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s45500_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_000655360/READY.json).
- [s45500_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_001310720/READY.json).
- [s45500_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_001966080/READY.json).
- [s45500_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_002621440/READY.json).
- [s45500_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_003276800/READY.json).
- [s45500_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_003932160/READY.json).
- [s45500_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_004587520/READY.json).
- [s45500_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45500_step_005242880/READY.json).
- [s45501_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_000655360/READY.json).
- [s45501_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_001310720/READY.json).
- [s45501_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_001966080/READY.json).
- [s45501_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_002621440/READY.json).
- [s45501_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_003276800/READY.json).
- [s45501_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_003932160/READY.json).
- [s45501_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_004587520/READY.json).
- [s45501_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_relative_reward_20261013/checkpoints/s45501_step_005242880/READY.json).

### aim_cpu_selected504_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/plan.json)
- 계획 SHA-256: `a485e3902010f630e64c576595afcc591133d15f26c588fa447021c949b595ed`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/checkpoints/s47081_step_003932160`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_actor_head_only_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47090_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_000655360/READY.json).
- [s47090_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_001310720/READY.json).
- [s47090_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_001966080/READY.json).
- [s47090_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_002621440/READY.json).
- [s47090_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_003276800/READY.json).
- [s47090_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47090_step_003932160/READY.json).
- [s47091_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_000655360/READY.json).
- [s47091_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_001310720/READY.json).
- [s47091_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_001966080/READY.json).
- [s47091_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_002621440/READY.json).
- [s47091_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_003276800/READY.json).
- [s47091_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected504_resume_20261014/checkpoints/s47091_step_003932160/READY.json).

### aim_cpu_selected510_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/plan.json)
- 계획 SHA-256: `8e4956b620de4588eb53c27c45f202c61088dd77027a580c4ce19e532f480d5c`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47160_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47160_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47160_step_000655360/READY.json).
- [s47160_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47160_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47160_step_001310720/READY.json).
- [s47160_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47160_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47160_step_001966080/READY.json).
- [s47161_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47161_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47161_step_000655360/READY.json).
- [s47161_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47161_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47161_step_001310720/READY.json).
- [s47161_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47161_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_selected510_resume_20261014/checkpoints/s47161_step_001966080/READY.json).

### aim_cpu_separate_clip_20261015

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/plan.json)
- 계획 SHA-256: `92639ef0263de103cf8dd68ceb2db9773ffa9c677863a964ca8ca05b3a5c1ca0`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_mutual_focus_20261015/checkpoints/s47223_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P2**.

부모 실행 `aim_cpu_known_mutual_focus_20261015`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.stage` | `"quarter"` | `"post_encounter"` |
| `gradient_clipping.actor_max_norm` | `"미기록"` | `0.5` |
| `gradient_clipping.critic_max_norm` | `"미기록"` | `0.5` |
| `gradient_clipping.mode` | `"미기록"` | `"separate_actor_critic"` |
| `opening_anchor.batch_size` | `"미기록"` | `64` |
| `opening_anchor.coefficient` | `"미기록"` | `1.0` |

저장된 추가 학습 모델:

- [s47234_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_000655360/READY.json).
- [s47234_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_001310720/READY.json).
- [s47234_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_001966080/READY.json).
- [s47234_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_002621440/READY.json).
- [s47234_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_003276800/READY.json).
- [s47234_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_003932160/READY.json).
- [s47234_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_004587520/READY.json).
- [s47234_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_005242880/READY.json).
- [s47234_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_005898240/READY.json).
- [s47234_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_006553600/READY.json).
- [s47234_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_007208960/READY.json).
- [s47234_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_007864320/READY.json).
- [s47234_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_008519680/READY.json).
- [s47234_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_009175040/READY.json).
- [s47234_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_009830400/READY.json).
- [s47234_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_separate_clip_20261015/checkpoints/s47234_step_010485760/READY.json).

### aim_cpu_short500_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_20261014/plan.json)
- 계획 SHA-256: `f237c69256e494c22e46de15c7d542917d2a59fba39da219d4c7143f1593f56f`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### aim_cpu_short500_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/plan.json)
- 계획 SHA-256: `c89cc5d701f28e4175c8ed5b4c81a0c6b10ffe31228f2e8c16dd06f2ada97032`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_received075_resume_20261014/checkpoints/s47021_step_005242880`.
- 추가 학습 READY: **12개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_received075_resume_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47040_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_000655360/READY.json).
- [s47040_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_001310720/READY.json).
- [s47040_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_001966080/READY.json).
- [s47040_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_002621440/READY.json).
- [s47040_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_003276800/READY.json).
- [s47040_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47040_step_003932160/READY.json).
- [s47041_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_000655360/READY.json).
- [s47041_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_001310720/READY.json).
- [s47041_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_001966080/READY.json).
- [s47041_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_002621440/READY.json).
- [s47041_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_003276800/READY.json).
- [s47041_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short500_resume_20261014/checkpoints/s47041_step_003932160/READY.json).

### aim_cpu_short508_resume_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/plan.json)
- 계획 SHA-256: `5b7e184f4e44f738a858f1366e8b503dc89bc7160931482314577e4f23b80f1a`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_head_rate5e6_20261014/checkpoints/s47111_step_001310720`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_head_rate5e6_20261014`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s47120_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47120_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47120_step_000655360/READY.json).
- [s47120_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47120_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47120_step_001310720/READY.json).
- [s47120_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47120_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47120_step_001966080/READY.json).
- [s47121_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47121_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47121_step_000655360/READY.json).
- [s47121_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47121_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47121_step_001310720/READY.json).
- [s47121_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47121_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_short508_resume_20261014/checkpoints/s47121_step_001966080/READY.json).

### aim_cpu_start025_20261014

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/plan.json)
- 계획 SHA-256: `4ab41d625339b9caaf6d4af2a8ef23d473c24dd778c8697bb04b3da1d7fb4413`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080`.
- 추가 학습 READY: **6개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_feature_train1e6_20261014`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.5` | `0.25` |

저장된 추가 학습 모델:

- [s47210_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47210_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47210_step_000655360/READY.json).
- [s47210_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47210_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47210_step_001310720/READY.json).
- [s47210_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47210_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47210_step_001966080/READY.json).
- [s47211_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47211_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47211_step_000655360/READY.json).
- [s47211_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47211_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47211_step_001310720/READY.json).
- [s47211_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47211_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_start025_20261014/checkpoints/s47211_step_001966080/READY.json).

### aim_cpu_state_prior_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/plan.json)
- 계획 SHA-256: `3a56442ea58f5a3b7ac9cc9ae13718e6766b60ddb037ffdefd844048df3ebb0e`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_guidance_range12k_20261013/checkpoints/s45900_step_010485760`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `aim_cpu_guidance_range12k_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `conflict_prior_bias` | `"미기록"` | `3.0` |
| `prior_bias` | `"미기록"` | `6.0` |

저장된 추가 학습 모델:

- [s46200_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_000655360/READY.json).
- [s46200_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_001310720/READY.json).
- [s46200_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_001966080/READY.json).
- [s46200_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_002621440/READY.json).
- [s46200_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_003276800/READY.json).
- [s46200_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_003932160/READY.json).
- [s46200_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_004587520/READY.json).
- [s46200_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46200_step_005242880/READY.json).
- [s46201_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_000655360/READY.json).
- [s46201_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_001310720/READY.json).
- [s46201_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_001966080/READY.json).
- [s46201_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_002621440/READY.json).
- [s46201_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_003276800/READY.json).
- [s46201_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_003932160/READY.json).
- [s46201_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_004587520/READY.json).
- [s46201_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_state_prior_20261013/checkpoints/s46201_step_005242880/READY.json).

### rear_cpu_actualrear010_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/plan.json)
- 계획 SHA-256: `79adb048feba1dcaad1ba56b8a07aa6dbd39ce8ce2485d26d5090e3cf942d7a5`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `rear_damage_coefficient` | `0.05` | `0.1` |

저장된 추가 학습 모델:

- [s43900_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_000655360/READY.json).
- [s43900_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_001310720/READY.json).
- [s43900_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_001966080/READY.json).
- [s43900_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_002621440/READY.json).
- [s43900_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_003276800/READY.json).
- [s43900_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_003932160/READY.json).
- [s43900_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_004587520/READY.json).
- [s43900_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43900_step_005242880/READY.json).
- [s43901_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_000655360/READY.json).
- [s43901_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_001310720/READY.json).
- [s43901_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_001966080/READY.json).
- [s43901_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_002621440/READY.json).
- [s43901_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_003276800/READY.json).
- [s43901_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_003932160/READY.json).
- [s43901_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_004587520/READY.json).
- [s43901_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_20261013/checkpoints/s43901_step_005242880/READY.json).

### rear_cpu_actualrear010_early131_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_actualrear010_early131_20261013/plan.json)
- 계획 SHA-256: `79adb048feba1dcaad1ba56b8a07aa6dbd39ce8ce2485d26d5090e3cf942d7a5`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `rear_damage_coefficient` | `0.05` | `0.1` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_async_smoke_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_20261012/plan.json)
- 계획 SHA-256: `acc4214bcd7f715c3cea763b556306ee805957573e96b116ea3c2f8ca4fbbf0e`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **2개**.
- 상대 샘플링 비중: **P1**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{
  "ppo.learning_rate": 0.0001,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.1,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.0,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.25,
  "tail_coefficients.hold": 0.01,
  "tail_coefficients.damage": 0.25,
  "tail_coefficients.received": 0.5,
  "position_per_second": 0.01,
  "rear_damage_coefficient": 0.5,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

저장된 추가 학습 모델:

- [s42990_step_000020480](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_20261012/checkpoints/s42990_step_000020480) — 추가 20,480단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_20261012/checkpoints/s42990_step_000020480/READY.json).
- [s42990_step_000040960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_20261012/checkpoints/s42990_step_000040960) — 추가 40,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_20261012/checkpoints/s42990_step_000040960/READY.json).

### rear_cpu_async_smoke_v2_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_v2_20261012/plan.json)
- 계획 SHA-256: `c27dc75c7a1962eaae17ba0feb5237b99e74046e46e71e23428fd42627cd43f0`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **2개**.
- 상대 샘플링 비중: **P1**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{
  "ppo.learning_rate": 0.0001,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.1,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.0,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.25,
  "tail_coefficients.hold": 0.01,
  "tail_coefficients.damage": 0.25,
  "tail_coefficients.received": 0.5,
  "position_per_second": 0.01,
  "rear_damage_coefficient": 0.5,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

저장된 추가 학습 모델:

- [s42991_step_000020480](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_v2_20261012/checkpoints/s42991_step_000020480) — 추가 20,480단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_v2_20261012/checkpoints/s42991_step_000020480/READY.json).
- [s42991_step_000040960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_v2_20261012/checkpoints/s42991_step_000040960) — 추가 40,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_async_smoke_v2_20261012/checkpoints/s42991_step_000040960/READY.json).

### rear_cpu_balanced582_clip005_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/plan.json)
- 계획 SHA-256: `a403f4ce7718fb89348747d48a487ace12dff464e11621c8d9971bb7e3b4aa5e`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_006553600`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage050_win585_continue_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.clip_range` | `0.1` | `0.05` |

저장된 추가 학습 모델:

- [s44600_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_000655360/READY.json).
- [s44600_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_001310720/READY.json).
- [s44600_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_001966080/READY.json).
- [s44600_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440/READY.json).
- [s44600_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_003276800/READY.json).
- [s44600_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_003932160/READY.json).
- [s44600_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_004587520/READY.json).
- [s44600_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_005242880/READY.json).
- [s44601_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_000655360/READY.json).
- [s44601_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_001310720/READY.json).
- [s44601_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_001966080/READY.json).
- [s44601_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_002621440/READY.json).
- [s44601_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_003276800/READY.json).
- [s44601_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_003932160/READY.json).
- [s44601_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_004587520/READY.json).
- [s44601_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44601_step_005242880/READY.json).

### rear_cpu_balanced582_clip005_early262_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_early262_20261013/plan.json)
- 계획 SHA-256: `a403f4ce7718fb89348747d48a487ace12dff464e11621c8d9971bb7e3b4aa5e`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_006553600`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage050_win585_continue_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.clip_range` | `0.1` | `0.05` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_balanced582_continue_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/plan.json)
- 계획 SHA-256: `cce50ece9c7a048a8d6d1fc582d5a2b12fd06dd52a87619957a9fe9349d70ed8`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_006553600`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage050_win585_continue_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s44500_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_000655360/READY.json).
- [s44500_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_001310720/READY.json).
- [s44500_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_001966080/READY.json).
- [s44500_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_002621440/READY.json).
- [s44500_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_003276800/READY.json).
- [s44500_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_003932160/READY.json).
- [s44500_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_004587520/READY.json).
- [s44500_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44500_step_005242880/READY.json).
- [s44501_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_000655360/READY.json).
- [s44501_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_001310720/READY.json).
- [s44501_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_001966080/READY.json).
- [s44501_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_002621440/READY.json).
- [s44501_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_003276800/READY.json).
- [s44501_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_003932160/READY.json).
- [s44501_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_004587520/READY.json).
- [s44501_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_continue_20261013/checkpoints/s44501_step_005242880/READY.json).

### rear_cpu_balanced582_early262_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_early262_20261013/plan.json)
- 계획 SHA-256: `cce50ece9c7a048a8d6d1fc582d5a2b12fd06dd52a87619957a9fe9349d70ed8`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_006553600`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage050_win585_continue_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_clip005_win585_rear12_continue_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/plan.json)
- 계획 SHA-256: `cccc85929e772e2ab8877064d02d9a59b5c16e7f3e81f155985bf22d03d0dc40`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s44700_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_000655360/READY.json).
- [s44700_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_001310720/READY.json).
- [s44700_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_001966080/READY.json).
- [s44700_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_002621440/READY.json).
- [s44700_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_003276800/READY.json).
- [s44700_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_003932160/READY.json).
- [s44700_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_004587520/READY.json).
- [s44700_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44700_step_005242880/READY.json).
- [s44701_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_000655360/READY.json).
- [s44701_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_001310720/READY.json).
- [s44701_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_001966080/READY.json).
- [s44701_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_002621440/READY.json).
- [s44701_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_003276800/READY.json).
- [s44701_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_003932160/READY.json).
- [s44701_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_004587520/READY.json).
- [s44701_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_clip005_win585_rear12_continue_20261013/checkpoints/s44701_step_005242880/READY.json).

### rear_cpu_curriculum025_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/plan.json)
- 계획 SHA-256: `3ce5c2cc320225f8a04e3c1904853391bc3eacb5e1d01b32799ee6506a2417cf`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.5` | `0.25` |

저장된 추가 학습 모델:

- [s43300_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_000655360/READY.json).
- [s43300_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_001310720/READY.json).
- [s43300_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_001966080/READY.json).
- [s43300_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_002621440/READY.json).
- [s43300_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_003276800/READY.json).
- [s43300_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_003932160/READY.json).
- [s43300_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_004587520/READY.json).
- [s43300_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43300_step_005242880/READY.json).
- [s43301_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_000655360/READY.json).
- [s43301_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_001310720/READY.json).
- [s43301_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_001966080/READY.json).
- [s43301_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_002621440/READY.json).
- [s43301_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_003276800/READY.json).
- [s43301_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_003932160/READY.json).
- [s43301_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_004587520/READY.json).
- [s43301_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_20261012/checkpoints/s43301_step_005242880/READY.json).

### rear_cpu_curriculum025_early393_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_curriculum025_early393_20261012/plan.json)
- 계획 SHA-256: `3ce5c2cc320225f8a04e3c1904853391bc3eacb5e1d01b32799ee6506a2417cf`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.5` | `0.25` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_damage0125_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/plan.json)
- 계획 SHA-256: `8395f675df13b00a24c92b7984229151355b9a0d6ebadf738476b36fd6969152`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_005242880`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_entropy01_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `tail_coefficients.damage` | `0.25` | `0.125` |

저장된 추가 학습 모델:

- [s42700_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_000655360/READY.json).
- [s42700_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_001310720/READY.json).
- [s42700_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_001966080/READY.json).
- [s42700_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_002621440/READY.json).
- [s42700_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_003276800/READY.json).
- [s42700_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_003932160/READY.json).
- [s42700_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_004587520/READY.json).
- [s42700_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_005242880/READY.json).
- [s42700_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_005898240/READY.json).
- [s42700_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_006553600/READY.json).
- [s42700_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_007208960/READY.json).
- [s42700_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_007864320/READY.json).
- [s42700_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_008519680/READY.json).
- [s42700_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_009175040/READY.json).
- [s42700_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_009830400/READY.json).
- [s42700_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_010485760/READY.json).
- [s42701_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_000655360/READY.json).
- [s42701_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_001310720/READY.json).
- [s42701_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_001966080/READY.json).
- [s42701_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_002621440/READY.json).
- [s42701_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_003276800/READY.json).
- [s42701_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_003932160/READY.json).
- [s42701_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_004587520/READY.json).
- [s42701_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_005242880/READY.json).
- [s42701_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_005898240/READY.json).
- [s42701_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_006553600/READY.json).
- [s42701_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_007208960/READY.json).
- [s42701_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_007864320/READY.json).
- [s42701_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_008519680/READY.json).
- [s42701_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_009175040/READY.json).
- [s42701_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_009830400/READY.json).
- [s42701_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42701_step_010485760/READY.json).

### rear_cpu_damage0125_continue_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/plan.json)
- 계획 SHA-256: `a9da89d6495c2c2b2fe08f331c6b4032d4464b3ebc8b9588ac1a5489db4b8b20`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_20261012/checkpoints/s42700_step_010485760`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_20261012`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s42800_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_000655360/READY.json).
- [s42800_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_001310720/READY.json).
- [s42800_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_001966080/READY.json).
- [s42800_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_002621440/READY.json).
- [s42800_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_003276800/READY.json).
- [s42800_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_003932160/READY.json).
- [s42800_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_004587520/READY.json).
- [s42800_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880/READY.json).
- [s42800_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005898240/READY.json).
- [s42800_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_006553600/READY.json).
- [s42800_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_007208960/READY.json).
- [s42800_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_007864320/READY.json).
- [s42800_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_008519680/READY.json).
- [s42800_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_009175040/READY.json).
- [s42800_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_009830400/READY.json).
- [s42800_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_010485760/READY.json).
- [s42801_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_000655360/READY.json).
- [s42801_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_001310720/READY.json).
- [s42801_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_001966080/READY.json).
- [s42801_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_002621440/READY.json).
- [s42801_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_003276800/READY.json).
- [s42801_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_003932160/READY.json).
- [s42801_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_004587520/READY.json).
- [s42801_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42801_step_005242880/READY.json).

### rear_cpu_damage0125_lr3e5_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/plan.json)
- 계획 SHA-256: `c06fc23d623eca0e91518d52533db222813ad08d341feabf5b141af5dfc1f554`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `0.0001` | `3e-05` |

저장된 추가 학습 모델:

- [s42900_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_000655360/READY.json).
- [s42900_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_001310720/READY.json).
- [s42900_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_001966080/READY.json).
- [s42900_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_002621440/READY.json).
- [s42900_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_003276800/READY.json).
- [s42900_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_003932160/READY.json).
- [s42900_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_004587520/READY.json).
- [s42900_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42900_step_005242880/READY.json).
- [s42901_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_000655360/READY.json).
- [s42901_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_001310720/READY.json).
- [s42901_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_001966080/READY.json).
- [s42901_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_002621440/READY.json).
- [s42901_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_003276800/READY.json).
- [s42901_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_003932160/READY.json).
- [s42901_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_004587520/READY.json).
- [s42901_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_lr3e5_20261012/checkpoints/s42901_step_005242880/READY.json).

### rear_cpu_damage050_win585_continue_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/plan.json)
- 계획 SHA-256: `727abf769e7511963b975ae9ae72102fd732287e5a31038f020f7200f2be547a`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_005242880`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear_damage050_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s44400_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_000655360/READY.json).
- [s44400_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_001310720/READY.json).
- [s44400_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_001966080/READY.json).
- [s44400_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_002621440/READY.json).
- [s44400_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_003276800/READY.json).
- [s44400_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_003932160/READY.json).
- [s44400_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_004587520/READY.json).
- [s44400_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_005242880/READY.json).
- [s44400_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_005898240/READY.json).
- [s44400_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_006553600/READY.json).
- [s44400_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_007208960/READY.json).
- [s44400_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_007864320/READY.json).
- [s44400_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_008519680/READY.json).
- [s44400_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_009175040/READY.json).
- [s44400_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_009830400/READY.json).
- [s44400_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44400_step_010485760/READY.json).
- [s44401_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_000655360/READY.json).
- [s44401_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_001310720/READY.json).
- [s44401_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_001966080/READY.json).
- [s44401_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_002621440/READY.json).
- [s44401_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_003276800/READY.json).
- [s44401_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_003932160/READY.json).
- [s44401_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_004587520/READY.json).
- [s44401_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage050_win585_continue_20261013/checkpoints/s44401_step_005242880/READY.json).

### rear_cpu_entropy001_early393_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy001_early393_20261013/plan.json)
- 계획 SHA-256: `1e2dc46e364ef7047636b1b1730e22faf18b0ca6a69976eae8410b8047bc85d1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.0` | `0.001` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_entropy003_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/plan.json)
- 계획 SHA-256: `eb773fee3c8a7734a370c0082de349fbd69e8e4f68ee83d2395f1ed7c7946c3a`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_010485760`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_entropy01_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.01` | `0.003` |

저장된 추가 학습 모델:

- [s42300_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_000655360/READY.json).
- [s42300_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_001310720/READY.json).
- [s42300_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_001966080/READY.json).
- [s42300_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_002621440/READY.json).
- [s42300_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_003276800/READY.json).
- [s42300_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_003932160/READY.json).
- [s42300_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_004587520/READY.json).
- [s42300_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_005242880/READY.json).
- [s42300_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_005898240/READY.json).
- [s42300_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_006553600/READY.json).
- [s42300_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_007208960/READY.json).
- [s42300_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_007864320/READY.json).
- [s42300_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_008519680/READY.json).
- [s42300_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_009175040/READY.json).
- [s42300_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_009830400/READY.json).
- [s42300_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42300_step_010485760/READY.json).
- [s42301_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_000655360/READY.json).
- [s42301_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_001310720/READY.json).
- [s42301_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_001966080/READY.json).
- [s42301_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_002621440/READY.json).
- [s42301_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_003276800/READY.json).
- [s42301_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_003932160/READY.json).
- [s42301_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_004587520/READY.json).
- [s42301_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_005242880/READY.json).
- [s42301_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_005898240/READY.json).
- [s42301_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_006553600/READY.json).
- [s42301_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_007208960/READY.json).
- [s42301_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_007864320/READY.json).
- [s42301_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_008519680/READY.json).
- [s42301_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_009175040/READY.json).
- [s42301_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_009830400/READY.json).
- [s42301_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy003_20261012/checkpoints/s42301_step_010485760/READY.json).

### rear_cpu_entropy005_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/plan.json)
- 계획 SHA-256: `9e31c60c5240bba017cee0e17d3c45030c0c7d82f4e8684b7e27a061dec00633`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.01` | `0.005` |

저장된 추가 학습 모델:

- [s43100_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_000655360/READY.json).
- [s43100_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_001310720/READY.json).
- [s43100_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_001966080/READY.json).
- [s43100_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_002621440/READY.json).
- [s43100_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_003276800/READY.json).
- [s43100_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_003932160/READY.json).
- [s43100_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_004587520/READY.json).
- [s43100_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_005242880/READY.json).
- [s43100_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_005898240/READY.json).
- [s43100_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_006553600/READY.json).
- [s43100_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_007208960/READY.json).
- [s43100_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_007864320/READY.json).
- [s43100_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_008519680/READY.json).
- [s43100_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_009175040/READY.json).
- [s43100_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_009830400/READY.json).
- [s43100_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43100_step_010485760/READY.json).
- [s43101_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_000655360/READY.json).
- [s43101_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_001310720/READY.json).
- [s43101_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_001966080/READY.json).
- [s43101_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_002621440/READY.json).
- [s43101_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_003276800/READY.json).
- [s43101_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_003932160/READY.json).
- [s43101_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_004587520/READY.json).
- [s43101_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy005_20261012/checkpoints/s43101_step_005242880/READY.json).

### rear_cpu_entropy006_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/plan.json)
- 계획 SHA-256: `2efcd4e8b3da7e94d29d057cad17e0d9940c56c9c2009556a45df4661f8dd405`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_005242880`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_lr3e5_continue_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.01` | `0.006` |

저장된 추가 학습 모델:

- [s42600_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_000655360/READY.json).
- [s42600_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_001310720/READY.json).
- [s42600_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_001966080/READY.json).
- [s42600_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_002621440/READY.json).
- [s42600_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_003276800/READY.json).
- [s42600_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_003932160/READY.json).
- [s42600_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_004587520/READY.json).
- [s42600_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_005242880/READY.json).
- [s42600_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_005898240/READY.json).
- [s42600_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_006553600/READY.json).
- [s42600_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_007208960/READY.json).
- [s42600_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_007864320/READY.json).
- [s42600_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_008519680/READY.json).
- [s42600_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_009175040/READY.json).
- [s42600_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_009830400/READY.json).
- [s42600_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42600_step_010485760/READY.json).
- [s42601_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_000655360/READY.json).
- [s42601_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_001310720/READY.json).
- [s42601_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_001966080/READY.json).
- [s42601_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_002621440/READY.json).
- [s42601_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_003276800/READY.json).
- [s42601_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_003932160/READY.json).
- [s42601_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_004587520/READY.json).
- [s42601_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_005242880/READY.json).
- [s42601_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_005898240/READY.json).
- [s42601_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_006553600/READY.json).
- [s42601_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_007208960/READY.json).
- [s42601_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_007864320/READY.json).
- [s42601_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_008519680/READY.json).
- [s42601_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_009175040/READY.json).
- [s42601_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_009830400/READY.json).
- [s42601_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy006_20261012/checkpoints/s42601_step_010485760/READY.json).

### rear_cpu_entropy01_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/plan.json)
- 계획 SHA-256: `b70d2a361bc59facb95f0bc7c328fa57028a73127f6abff981c676a39f981008`.
- 기록된 부모: `experiments/league/rear_branch_20261012/rear_candidate/learner.zip`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P1**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{
  "ppo.learning_rate": 0.0001,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.1,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.01,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.25,
  "tail_coefficients.hold": 0.01,
  "tail_coefficients.damage": 0.25,
  "tail_coefficients.received": 0.5,
  "position_per_second": 0.01,
  "rear_damage_coefficient": 0.5,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

저장된 추가 학습 모델:

- [s42200_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_000655360/READY.json).
- [s42200_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_001310720/READY.json).
- [s42200_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_001966080/READY.json).
- [s42200_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_002621440/READY.json).
- [s42200_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_003276800/READY.json).
- [s42200_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_003932160/READY.json).
- [s42200_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_004587520/READY.json).
- [s42200_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_005242880/READY.json).
- [s42200_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_005898240/READY.json).
- [s42200_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_006553600/READY.json).
- [s42200_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_007208960/READY.json).
- [s42200_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_007864320/READY.json).
- [s42200_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_008519680/READY.json).
- [s42200_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_009175040/READY.json).
- [s42200_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_009830400/READY.json).
- [s42200_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42200_step_010485760/READY.json).
- [s42201_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_000655360/READY.json).
- [s42201_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_001310720/READY.json).
- [s42201_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_001966080/READY.json).
- [s42201_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_002621440/READY.json).
- [s42201_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_003276800/READY.json).
- [s42201_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_003932160/READY.json).
- [s42201_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_004587520/READY.json).
- [s42201_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_005242880/READY.json).
- [s42201_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_005898240/READY.json).
- [s42201_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_006553600/READY.json).
- [s42201_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_007208960/READY.json).
- [s42201_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_007864320/READY.json).
- [s42201_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_008519680/READY.json).
- [s42201_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_009175040/READY.json).
- [s42201_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_009830400/READY.json).
- [s42201_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_010485760/READY.json).

### rear_cpu_joint585_continue_early393_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_continue_early393_20261013/plan.json)
- 계획 SHA-256: `cccc85929e772e2ab8877064d02d9a59b5c16e7f3e81f155985bf22d03d0dc40`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_joint585_epochs2_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/plan.json)
- 계획 SHA-256: `6f3b92890efb27f1a632554e535e800514fb00222560981efbadddbc24d75828`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.n_epochs` | `4` | `2` |

저장된 추가 학습 모델:

- [s44800_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_000655360/READY.json).
- [s44800_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_001310720/READY.json).
- [s44800_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_001966080/READY.json).
- [s44800_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_002621440/READY.json).
- [s44800_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_003276800/READY.json).
- [s44800_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_003932160/READY.json).
- [s44800_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_004587520/READY.json).
- [s44800_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44800_step_005242880/READY.json).
- [s44801_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_000655360/READY.json).
- [s44801_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_001310720/READY.json).
- [s44801_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_001966080/READY.json).
- [s44801_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_002621440/READY.json).
- [s44801_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_003276800/READY.json).
- [s44801_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_003932160/READY.json).
- [s44801_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_004587520/READY.json).
- [s44801_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_005242880/READY.json).

### rear_cpu_joint585_epochs2_early393_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_early393_20261013/plan.json)
- 계획 SHA-256: `6f3b92890efb27f1a632554e535e800514fb00222560981efbadddbc24d75828`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.n_epochs` | `4` | `2` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_joint585_lr3e5_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/plan.json)
- 계획 SHA-256: `2e9c6706231f657e67fbe27b8246606385bae1e8b8e0cf6d31ec9dfb7448f539`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `0.0001` | `3e-05` |

저장된 추가 학습 모델:

- [s45200_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_000655360/READY.json).
- [s45200_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_001310720/READY.json).
- [s45200_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_001966080/READY.json).
- [s45200_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_002621440/READY.json).
- [s45200_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003276800/READY.json).
- [s45200_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_003932160/READY.json).
- [s45200_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_004587520/READY.json).
- [s45200_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45200_step_005242880/READY.json).
- [s45201_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_000655360/READY.json).
- [s45201_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_001310720/READY.json).
- [s45201_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_001966080/READY.json).
- [s45201_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_002621440/READY.json).
- [s45201_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_003276800/READY.json).
- [s45201_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_003932160/READY.json).
- [s45201_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_004587520/READY.json).
- [s45201_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_20261013/checkpoints/s45201_step_005242880/READY.json).

### rear_cpu_joint585_lr3e5_early393_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_lr3e5_early393_20261013/plan.json)
- 계획 SHA-256: `2e9c6706231f657e67fbe27b8246606385bae1e8b8e0cf6d31ec9dfb7448f539`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `0.0001` | `3e-05` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_joint585_potential010_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/plan.json)
- 계획 SHA-256: `af033ac7cdbf43f4a58a73602cfe98a3c35ca3351123e6ebeb170d6b9fb832e1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `tail_coefficients.potential` | `0.025` | `0.01` |

저장된 추가 학습 모델:

- [s45100_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_000655360/READY.json).
- [s45100_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_001310720/READY.json).
- [s45100_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_001966080/READY.json).
- [s45100_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_002621440/READY.json).
- [s45100_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_003276800/READY.json).
- [s45100_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_003932160/READY.json).
- [s45100_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_004587520/READY.json).
- [s45100_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45100_step_005242880/READY.json).
- [s45101_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_000655360/READY.json).
- [s45101_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_001310720/READY.json).
- [s45101_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_001966080/READY.json).
- [s45101_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_002621440/READY.json).
- [s45101_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_003276800/READY.json).
- [s45101_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_003932160/READY.json).
- [s45101_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_004587520/READY.json).
- [s45101_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_20261013/checkpoints/s45101_step_005242880/READY.json).

### rear_cpu_joint585_potential010_early262_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_potential010_early262_20261013/plan.json)
- 계획 SHA-256: `af033ac7cdbf43f4a58a73602cfe98a3c35ca3351123e6ebeb170d6b9fb832e1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_balanced582_clip005_20261013/checkpoints/s44600_step_002621440`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_balanced582_clip005_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `tail_coefficients.potential` | `0.025` | `0.01` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_lr3e5_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/plan.json)
- 계획 SHA-256: `488d4a1669c38dc39d8df593924a9864202375ad681c1c336e77347ba2167727`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_entropy01_20261012/checkpoints/s42201_step_010485760`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_entropy01_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `0.0001` | `3e-05` |

저장된 추가 학습 모델:

- [s42400_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_000655360/READY.json).
- [s42400_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_001310720/READY.json).
- [s42400_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_001966080/READY.json).
- [s42400_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_002621440/READY.json).
- [s42400_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_003276800/READY.json).
- [s42400_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_003932160/READY.json).
- [s42400_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_004587520/READY.json).
- [s42400_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_005242880/READY.json).
- [s42400_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_005898240/READY.json).
- [s42400_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_006553600/READY.json).
- [s42400_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_007208960/READY.json).
- [s42400_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_007864320/READY.json).
- [s42400_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_008519680/READY.json).
- [s42400_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_009175040/READY.json).
- [s42400_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_009830400/READY.json).
- [s42400_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_010485760/READY.json).
- [s42401_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_000655360/READY.json).
- [s42401_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_001310720/READY.json).
- [s42401_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_001966080/READY.json).
- [s42401_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_002621440/READY.json).
- [s42401_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_003276800/READY.json).
- [s42401_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_003932160/READY.json).
- [s42401_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_004587520/READY.json).
- [s42401_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_005242880/READY.json).
- [s42401_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_005898240/READY.json).
- [s42401_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_006553600/READY.json).
- [s42401_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_007208960/READY.json).
- [s42401_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_007864320/READY.json).
- [s42401_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_008519680/READY.json).
- [s42401_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_009175040/READY.json).
- [s42401_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_009830400/READY.json).
- [s42401_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42401_step_010485760/READY.json).

### rear_cpu_lr3e5_continue_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/plan.json)
- 계획 SHA-256: `b271af63814c4c43e6b8495bccf8da360615620f9776e84baa8625b9cbb4dc5b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_20261012/checkpoints/s42400_step_010485760`.
- 추가 학습 READY: **32개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_lr3e5_20261012`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s42500_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_000655360/READY.json).
- [s42500_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_001310720/READY.json).
- [s42500_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_001966080/READY.json).
- [s42500_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_002621440/READY.json).
- [s42500_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_003276800/READY.json).
- [s42500_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_003932160/READY.json).
- [s42500_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_004587520/READY.json).
- [s42500_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_005242880/READY.json).
- [s42500_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_005898240/READY.json).
- [s42500_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_006553600/READY.json).
- [s42500_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_007208960/READY.json).
- [s42500_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_007864320/READY.json).
- [s42500_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_008519680/READY.json).
- [s42500_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_009175040/READY.json).
- [s42500_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_009830400/READY.json).
- [s42500_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42500_step_010485760/READY.json).
- [s42501_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_000655360/READY.json).
- [s42501_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_001310720/READY.json).
- [s42501_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_001966080/READY.json).
- [s42501_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_002621440/READY.json).
- [s42501_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_003276800/READY.json).
- [s42501_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_003932160/READY.json).
- [s42501_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_004587520/READY.json).
- [s42501_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_005242880/READY.json).
- [s42501_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_005898240/READY.json).
- [s42501_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_006553600/READY.json).
- [s42501_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_007208960/READY.json).
- [s42501_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_007864320/READY.json).
- [s42501_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_008519680/READY.json).
- [s42501_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_009175040/READY.json).
- [s42501_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_009830400/READY.json).
- [s42501_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_lr3e5_continue_20261012/checkpoints/s42501_step_010485760/READY.json).

### rear_cpu_main_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/plan.json)
- 계획 SHA-256: `f369f38c97cceb209a76d07cfcc47de9f105c2dd7ff7d68c373501650f70e72e`.
- 기록된 부모: `체크포인트 출처 미기록`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

비교 가능한 부모 실행 계획을 확인하지 못했으므로 변경 전 값을 추정하지 않았다.

```json
{
  "ppo.learning_rate": 0.0001,
  "ppo.n_steps": 640,
  "ppo.batch_size": 512,
  "ppo.n_epochs": 4,
  "ppo.gamma": 0.999,
  "ppo.gae_lambda": 0.99,
  "ppo.clip_range": 0.1,
  "ppo.target_kl": 0.015,
  "ppo.ent_coef": 0.0,
  "ppo.max_grad_norm": 0.5,
  "tail_coefficients.potential": 0.25,
  "tail_coefficients.hold": 0.01,
  "tail_coefficients.damage": 0.25,
  "tail_coefficients.received": 0.5,
  "position_per_second": 0.01,
  "rear_damage_coefficient": 0.5,
  "completed_focus_weight": 0.0,
  "curriculum.stage": "quarter",
  "curriculum.fraction": 0.5
}
```

저장된 추가 학습 모델:

- [s42100_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_000655360/READY.json).
- [s42100_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_001310720/READY.json).
- [s42100_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_001966080/READY.json).
- [s42100_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_002621440/READY.json).
- [s42100_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_003276800/READY.json).
- [s42100_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_003932160/READY.json).
- [s42100_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_004587520/READY.json).
- [s42100_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_005242880/READY.json).
- [s42101_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_000655360/READY.json).
- [s42101_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_001310720/READY.json).
- [s42101_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_001966080/READY.json).
- [s42101_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_002621440/READY.json).
- [s42101_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_003276800/READY.json).
- [s42101_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_003932160/READY.json).
- [s42101_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_004587520/READY.json).
- [s42101_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42101_step_005242880/READY.json).

### rear_cpu_position0005_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/plan.json)
- 계획 SHA-256: `de81e8a207b9ce94e9c01d8753a26019938427f49f0508e8109532c9e2994d66`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `position_per_second` | `0.001` | `0.0005` |

저장된 추가 학습 모델:

- [s44000_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_000655360/READY.json).
- [s44000_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_001310720/READY.json).
- [s44000_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_001966080/READY.json).
- [s44000_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_002621440/READY.json).
- [s44000_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_003276800/READY.json).
- [s44000_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_003932160/READY.json).
- [s44000_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_004587520/READY.json).
- [s44000_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44000_step_005242880/READY.json).
- [s44001_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_000655360/READY.json).
- [s44001_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_001310720/READY.json).
- [s44001_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_001966080/READY.json).
- [s44001_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_002621440/READY.json).
- [s44001_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_003276800/READY.json).
- [s44001_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_003932160/READY.json).
- [s44001_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_004587520/READY.json).
- [s44001_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position0005_20261013/checkpoints/s44001_step_005242880/READY.json).

### rear_cpu_position002_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/plan.json)
- 계획 SHA-256: `e6d7815975f4bfbbbae2cf158e0528d2e409919bbb027eedc68db993a96662fc`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `position_per_second` | `0.01` | `0.02` |

저장된 추가 학습 모델:

- [s43000_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_000655360/READY.json).
- [s43000_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_001310720/READY.json).
- [s43000_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_001966080/READY.json).
- [s43000_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_002621440/READY.json).
- [s43000_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_003276800/READY.json).
- [s43000_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_003932160/READY.json).
- [s43000_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_004587520/READY.json).
- [s43000_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43000_step_005242880/READY.json).
- [s43001_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_000655360/READY.json).
- [s43001_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_001310720/READY.json).
- [s43001_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_001966080/READY.json).
- [s43001_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_002621440/READY.json).
- [s43001_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_003276800/READY.json).
- [s43001_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_003932160/READY.json).
- [s43001_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_004587520/READY.json).
- [s43001_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_position002_20261012/checkpoints/s43001_step_005242880/READY.json).

### rear_cpu_rearwin025_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/plan.json)
- 계획 SHA-256: `bcba78a78e803c7b759bc12f92c753da3181bc1a83903458b48be3367292c6a9`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s43200_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_000655360/READY.json).
- [s43200_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_001310720/READY.json).
- [s43200_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_001966080/READY.json).
- [s43200_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_002621440/READY.json).
- [s43200_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_003276800/READY.json).
- [s43200_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_003932160/READY.json).
- [s43200_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_004587520/READY.json).
- [s43200_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43200_step_005242880/READY.json).
- [s43201_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_000655360/READY.json).
- [s43201_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_001310720/READY.json).
- [s43201_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_001966080/READY.json).
- [s43201_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_002621440/READY.json).
- [s43201_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_003276800/READY.json).
- [s43201_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_003932160/READY.json).
- [s43201_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_004587520/READY.json).
- [s43201_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin025_20261012/checkpoints/s43201_step_005242880/READY.json).

### rear_cpu_rearwin075_20261012

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/plan.json)
- 계획 SHA-256: `47a0eee7bd1e4a4e42c217b9ca5ae2db33ef127a02cabb579b44886e217ed4c7`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_damage0125_continue_20261012/checkpoints/s42800_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_damage0125_continue_20261012`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s43400_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_000655360/READY.json).
- [s43400_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_001310720/READY.json).
- [s43400_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_001966080/READY.json).
- [s43400_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_002621440/READY.json).
- [s43400_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_003276800/READY.json).
- [s43400_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_003932160/READY.json).
- [s43400_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_004587520/READY.json).
- [s43400_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43400_step_005242880/READY.json).
- [s43401_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_000655360/READY.json).
- [s43401_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_001310720/READY.json).
- [s43401_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_001966080/READY.json).
- [s43401_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_002621440/READY.json).
- [s43401_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_003276800/READY.json).
- [s43401_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_003932160/READY.json).
- [s43401_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_004587520/READY.json).
- [s43401_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_rearwin075_20261012/checkpoints/s43401_step_005242880/READY.json).

### rear_cpu_weakrear010_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/plan.json)
- 계획 SHA-256: `94cc935d6a683d4ff0774a867c7769392be6bd002ef697820303bfc3372f829b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_main_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `position_per_second` | `0.01` | `0.001` |
| `rear_damage_coefficient` | `0.5` | `0.05` |
| `tail_coefficients.hold` | `0.01` | `0.001` |
| `tail_coefficients.potential` | `0.25` | `0.025` |

저장된 추가 학습 모델:

- [s43500_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_000655360/READY.json).
- [s43500_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_001310720/READY.json).
- [s43500_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_001966080/READY.json).
- [s43500_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440/READY.json).
- [s43500_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_003276800/READY.json).
- [s43500_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_003932160/READY.json).
- [s43500_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_004587520/READY.json).
- [s43500_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_005242880/READY.json).
- [s43501_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_000655360/READY.json).
- [s43501_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_001310720/READY.json).
- [s43501_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_001966080/READY.json).
- [s43501_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_002621440/READY.json).
- [s43501_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_003276800/READY.json).
- [s43501_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_003932160/READY.json).
- [s43501_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_004587520/READY.json).
- [s43501_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43501_step_005242880/READY.json).

### rear_cpu_weakrear010_continue_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/plan.json)
- 계획 SHA-256: `2d5f149e6925a49c34a2abc67efab0cbb692061ec27be93ad4443d3ede235eca`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s43600_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_000655360/READY.json).
- [s43600_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_001310720/READY.json).
- [s43600_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_001966080/READY.json).
- [s43600_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_002621440/READY.json).
- [s43600_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_003276800/READY.json).
- [s43600_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_003932160/READY.json).
- [s43600_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_004587520/READY.json).
- [s43600_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43600_step_005242880/READY.json).
- [s43601_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_000655360/READY.json).
- [s43601_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_001310720/READY.json).
- [s43601_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_001966080/READY.json).
- [s43601_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_002621440/READY.json).
- [s43601_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_003276800/READY.json).
- [s43601_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_003932160/READY.json).
- [s43601_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_004587520/READY.json).
- [s43601_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_005242880/READY.json).
- [s43601_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_005898240/READY.json).
- [s43601_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_006553600/READY.json).
- [s43601_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_007208960/READY.json).
- [s43601_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_007864320/READY.json).
- [s43601_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_008519680/READY.json).
- [s43601_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_009175040/READY.json).
- [s43601_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_009830400/READY.json).
- [s43601_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_continue_20261013/checkpoints/s43601_step_010485760/READY.json).

### rear_cpu_weakrear010_early262_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_early262_20261013/plan.json)
- 계획 SHA-256: `94cc935d6a683d4ff0774a867c7769392be6bd002ef697820303bfc3372f829b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_main_20261012/checkpoints/s42100_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_main_20261012`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `position_per_second` | `0.01` | `0.001` |
| `rear_damage_coefficient` | `0.5` | `0.05` |
| `tail_coefficients.hold` | `0.01` | `0.001` |
| `tail_coefficients.potential` | `0.25` | `0.025` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_weakrear010_entropy001_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/plan.json)
- 계획 SHA-256: `1e2dc46e364ef7047636b1b1730e22faf18b0ca6a69976eae8410b8047bc85d1`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.ent_coef` | `0.0` | `0.001` |

저장된 추가 학습 모델:

- [s43800_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_000655360/READY.json).
- [s43800_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_001310720/READY.json).
- [s43800_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_001966080/READY.json).
- [s43800_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_002621440/READY.json).
- [s43800_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_003276800/READY.json).
- [s43800_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_003932160/READY.json).
- [s43800_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_004587520/READY.json).
- [s43800_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43800_step_005242880/READY.json).
- [s43801_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_000655360/READY.json).
- [s43801_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_001310720/READY.json).
- [s43801_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_001966080/READY.json).
- [s43801_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_002621440/READY.json).
- [s43801_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_003276800/READY.json).
- [s43801_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_003932160/READY.json).
- [s43801_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_004587520/READY.json).
- [s43801_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_entropy001_20261013/checkpoints/s43801_step_005242880/READY.json).

### rear_cpu_weakrear010_lr3e5_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/plan.json)
- 계획 SHA-256: `7103ecae2818e3dabde090284e7a23fdb2c1fb3906d14778d2b608cbcd0595cb`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.learning_rate` | `0.0001` | `3e-05` |

저장된 추가 학습 모델:

- [s43700_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_000655360/READY.json).
- [s43700_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_001310720/READY.json).
- [s43700_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_001966080/READY.json).
- [s43700_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_002621440/READY.json).
- [s43700_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_003276800/READY.json).
- [s43700_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_003932160/READY.json).
- [s43700_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_004587520/READY.json).
- [s43700_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43700_step_005242880/READY.json).
- [s43701_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_000655360/READY.json).
- [s43701_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_001310720/READY.json).
- [s43701_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_001966080/READY.json).
- [s43701_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_002621440/READY.json).
- [s43701_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_003276800/READY.json).
- [s43701_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_003932160/READY.json).
- [s43701_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_004587520/READY.json).
- [s43701_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_lr3e5_20261013/checkpoints/s43701_step_005242880/READY.json).

### rear_cpu_weakrear_curriculum025_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/plan.json)
- 계획 SHA-256: `7f87d8a24c0bb1dd947c7b0489a3ca72f32deaa5c06237a7b8bcde4f70a56ec3`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **24개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `curriculum.fraction` | `0.5` | `0.25` |

저장된 추가 학습 모델:

- [s44100_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_000655360/READY.json).
- [s44100_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_001310720/READY.json).
- [s44100_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_001966080/READY.json).
- [s44100_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_002621440/READY.json).
- [s44100_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_003276800/READY.json).
- [s44100_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_003932160/READY.json).
- [s44100_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_004587520/READY.json).
- [s44100_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44100_step_005242880/READY.json).
- [s44101_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_000655360/READY.json).
- [s44101_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_001310720/READY.json).
- [s44101_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_001966080/READY.json).
- [s44101_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_002621440/READY.json).
- [s44101_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_003276800/READY.json).
- [s44101_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_003932160/READY.json).
- [s44101_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_004587520/READY.json).
- [s44101_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_005242880/READY.json).
- [s44101_step_005898240](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_005898240) — 추가 5,898,240단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_005898240/READY.json).
- [s44101_step_006553600](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_006553600) — 추가 6,553,600단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_006553600/READY.json).
- [s44101_step_007208960](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_007208960) — 추가 7,208,960단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_007208960/READY.json).
- [s44101_step_007864320](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_007864320) — 추가 7,864,320단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_007864320/READY.json).
- [s44101_step_008519680](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_008519680) — 추가 8,519,680단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_008519680/READY.json).
- [s44101_step_009175040](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_009175040) — 추가 9,175,040단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_009175040/READY.json).
- [s44101_step_009830400](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_009830400) — 추가 9,830,400단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_009830400/READY.json).
- [s44101_step_010485760](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_010485760) — 추가 10,485,760단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_curriculum025_20261013/checkpoints/s44101_step_010485760/READY.json).

### rear_cpu_weakrear_damage050_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/plan.json)
- 계획 SHA-256: `54458cafe3615e64887f0f49c874d0e48334dc441d3eb831ea9ee3e3d58bc4cc`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `tail_coefficients.damage` | `0.25` | `0.5` |

저장된 추가 학습 모델:

- [s44300_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_000655360/READY.json).
- [s44300_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_001310720/READY.json).
- [s44300_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_001966080/READY.json).
- [s44300_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_002621440/READY.json).
- [s44300_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_003276800/READY.json).
- [s44300_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_003932160/READY.json).
- [s44300_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_004587520/READY.json).
- [s44300_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_005242880/READY.json).
- [s44301_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_000655360/READY.json).
- [s44301_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_001310720/READY.json).
- [s44301_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_001966080/READY.json).
- [s44301_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_002621440/READY.json).
- [s44301_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_003276800/READY.json).
- [s44301_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_003932160/READY.json).
- [s44301_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_004587520/READY.json).
- [s44301_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44301_step_005242880/READY.json).

### rear_cpu_weakrear_received025_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/plan.json)
- 계획 SHA-256: `ef7c45713fa02ba9b1129408af095ca44edf777fa9d59c653223a58d90d8a2eb`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear010_20261013/checkpoints/s43500_step_002621440`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear010_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `tail_coefficients.received` | `0.5` | `0.25` |

저장된 추가 학습 모델:

- [s44200_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_000655360/READY.json).
- [s44200_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_001310720/READY.json).
- [s44200_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_001966080/READY.json).
- [s44200_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_002621440/READY.json).
- [s44200_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_003276800/READY.json).
- [s44200_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_003932160/READY.json).
- [s44200_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_004587520/READY.json).
- [s44200_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44200_step_005242880/READY.json).
- [s44201_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_000655360/READY.json).
- [s44201_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_001310720/READY.json).
- [s44201_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_001966080/READY.json).
- [s44201_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_002621440/READY.json).
- [s44201_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_003276800/READY.json).
- [s44201_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_003932160/READY.json).
- [s44201_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_004587520/READY.json).
- [s44201_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_received025_20261013/checkpoints/s44201_step_005242880/READY.json).

### rear_cpu_win585_continue_early655_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win585_continue_early655_20261013/plan.json)
- 계획 SHA-256: `727abf769e7511963b975ae9ae72102fd732287e5a31038f020f7200f2be547a`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_weakrear_damage050_20261013/checkpoints/s44300_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_weakrear_damage050_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_win586_continue_early131_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_continue_early131_20261013/plan.json)
- 계획 SHA-256: `26d7631b4b56e3948bd135bbf5309c41c54ddfbc5a2242a3f4c1e36d408eb505`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_epochs2_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_win586_epochs1_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/plan.json)
- 계획 SHA-256: `774ef9f8e13ae36bf58ab436b559327cb9ab590bfa06ab28c9dc63a3af9ad20b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_epochs2_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.n_epochs` | `2` | `1` |

저장된 추가 학습 모델:

- [s45000_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_000655360/READY.json).
- [s45000_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_001310720/READY.json).
- [s45000_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_001966080/READY.json).
- [s45000_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_002621440/READY.json).
- [s45000_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_003276800/READY.json).
- [s45000_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_003932160/READY.json).
- [s45000_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_004587520/READY.json).
- [s45000_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45000_step_005242880/READY.json).
- [s45001_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_000655360/READY.json).
- [s45001_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_001310720/READY.json).
- [s45001_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_001966080/READY.json).
- [s45001_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_002621440/READY.json).
- [s45001_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_003276800/READY.json).
- [s45001_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_003932160/READY.json).
- [s45001_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_004587520/READY.json).
- [s45001_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_20261013/checkpoints/s45001_step_005242880/READY.json).

### rear_cpu_win586_epochs1_early393_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs1_early393_20261013/plan.json)
- 계획 SHA-256: `774ef9f8e13ae36bf58ab436b559327cb9ab590bfa06ab28c9dc63a3af9ad20b`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_005242880`.
- 추가 학습 READY: **0개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_epochs2_20261013`의 계획 대비:

| 설정 | 부모 계획 | 이번 계획 |
| --- | --- | --- |
| `ppo.n_epochs` | `2` | `1` |

저장된 추가 학습 모델:

- 양수 추가 학습 단계의 READY가 발견되지 않음. 준비·진단 전용이거나 다른 위치에 결과가 있을 수 있다.

### rear_cpu_win586_epochs2_continue_20261013

- 계획: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/plan.json)
- 계획 SHA-256: `26d7631b4b56e3948bd135bbf5309c41c54ddfbc5a2242a3f4c1e36d408eb505`.
- 기록된 부모: `/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_joint585_epochs2_20261013/checkpoints/s44801_step_005242880`.
- 추가 학습 READY: **16개**.
- 상대 샘플링 비중: **P1**.

부모 실행 `rear_cpu_joint585_epochs2_20261013`의 계획 대비:

추출 대상 설정 키의 변경 없음. 상대 구성·학습 예산·신경망 내부 텐서의 동일성을 뜻하지 않는다.

저장된 추가 학습 모델:

- [s44900_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_000655360/READY.json).
- [s44900_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_001310720/READY.json).
- [s44900_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_001966080/READY.json).
- [s44900_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_002621440/READY.json).
- [s44900_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_003276800/READY.json).
- [s44900_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_003932160/READY.json).
- [s44900_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_004587520/READY.json).
- [s44900_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44900_step_005242880/READY.json).
- [s44901_step_000655360](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_000655360) — 추가 655,360단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_000655360/READY.json).
- [s44901_step_001310720](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_001310720) — 추가 1,310,720단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_001310720/READY.json).
- [s44901_step_001966080](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_001966080) — 추가 1,966,080단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_001966080/READY.json).
- [s44901_step_002621440](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_002621440) — 추가 2,621,440단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_002621440/READY.json).
- [s44901_step_003276800](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_003276800) — 추가 3,276,800단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_003276800/READY.json).
- [s44901_step_003932160](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_003932160) — 추가 3,932,160단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_003932160/READY.json).
- [s44901_step_004587520](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_004587520) — 추가 4,587,520단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_004587520/READY.json).
- [s44901_step_005242880](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_005242880) — 추가 5,242,880단계; learner.zip 있음; [READY](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/rear_cpu_win586_epochs2_continue_20261013/checkpoints/s44901_step_005242880/READY.json).

## 5. 상대 추출 비중 목록

계획서의 `opponents`와 `probabilities`를 같은 인덱스로 연결했다. 각 벡터의 길이와 합(허용 오차 1e-6)을 확인했다. 표는 소수점 10자리 표시이며 원래 정밀도는 연결한 계획서에 남아 있다.

### P1 — 48개 상대

상대 ID·비중 쌍의 canonical JSON SHA-256: `d034587677a6e9c34ed5cc0f60e603ab12f1d97ba0f619bee68f73c69b5dca48`.

사용 실행: `aim_cpu_actor_head_only_20261014`, `aim_cpu_best498_continue_20261013`, `aim_cpu_control_guidance_20261013`, `aim_cpu_current_anchor504_20261014`, `aim_cpu_current_anchor508_20261014`, `aim_cpu_dealt0625_20261014`, `aim_cpu_draw_peak_assessment_20261014`, `aim_cpu_draw_strength050_20261014`, `aim_cpu_draw_utility025_20261014`, `aim_cpu_draw_utility025_resume_20261014`, `aim_cpu_early59_complete_qualification_20261014`, `aim_cpu_early59_qualification_20261014`, `aim_cpu_enemy_los_control_20261014`, `aim_cpu_entropy001_20261014`, `aim_cpu_entropy003_20261013`, `aim_cpu_entropy005_resume_20261014`, `aim_cpu_entropy005_resume_v2_20261014`, `aim_cpu_feature_train1e6_20261014`, `aim_cpu_guidance_range12k_20261013`, `aim_cpu_head_rate1e6_20261014`, `aim_cpu_head_rate5e6_20261014`, `aim_cpu_heading_memory_20261014`, `aim_cpu_heading_rate_20261013`, `aim_cpu_long_credit0999_20261014`, `aim_cpu_mixed_start_20261013`, `aim_cpu_mixed_start_early393_20261013`, `aim_cpu_predictive_20261013`, `aim_cpu_predictive_enemy_20261014`, `aim_cpu_predictive_enemy_lr1e5_20261014`, `aim_cpu_predictive_enemy_lr1e5_resume_20261014`, `aim_cpu_prior3_20261013`, `aim_cpu_rate_input_20261014`, `aim_cpu_received075_20261014`, `aim_cpu_received075_resume_20261014`, `aim_cpu_received0875_20261014`, `aim_cpu_received1_20261013`, `aim_cpu_reference510_20261014`, `aim_cpu_reference510_v2_20261014`, `aim_cpu_reference_anchor01_20261014`, `aim_cpu_reference_midpoint_20261014`, `aim_cpu_relative_reward_20261013`, `aim_cpu_selected504_resume_20261014`, `aim_cpu_selected510_resume_20261014`, `aim_cpu_short500_20261014`, `aim_cpu_short500_resume_20261014`, `aim_cpu_short508_resume_20261014`, `aim_cpu_start025_20261014`, `aim_cpu_state_prior_20261013`, `rear_cpu_actualrear010_20261013`, `rear_cpu_actualrear010_early131_20261013`, `rear_cpu_async_smoke_20261012`, `rear_cpu_async_smoke_v2_20261012`, `rear_cpu_balanced582_clip005_20261013`, `rear_cpu_balanced582_clip005_early262_20261013`, `rear_cpu_balanced582_continue_20261013`, `rear_cpu_balanced582_early262_20261013`, `rear_cpu_clip005_win585_rear12_continue_20261013`, `rear_cpu_curriculum025_20261012`, `rear_cpu_curriculum025_early393_20261012`, `rear_cpu_damage0125_20261012`, `rear_cpu_damage0125_continue_20261012`, `rear_cpu_damage0125_lr3e5_20261012`, `rear_cpu_damage050_win585_continue_20261013`, `rear_cpu_entropy001_early393_20261013`, `rear_cpu_entropy003_20261012`, `rear_cpu_entropy005_20261012`, `rear_cpu_entropy006_20261012`, `rear_cpu_entropy01_20261012`, `rear_cpu_joint585_continue_early393_20261013`, `rear_cpu_joint585_epochs2_20261013`, `rear_cpu_joint585_epochs2_early393_20261013`, `rear_cpu_joint585_lr3e5_20261013`, `rear_cpu_joint585_lr3e5_early393_20261013`, `rear_cpu_joint585_potential010_20261013`, `rear_cpu_joint585_potential010_early262_20261013`, `rear_cpu_lr3e5_20261012`, `rear_cpu_lr3e5_continue_20261012`, `rear_cpu_main_20261012`, `rear_cpu_position0005_20261013`, `rear_cpu_position002_20261012`, `rear_cpu_rearwin025_20261012`, `rear_cpu_rearwin075_20261012`, `rear_cpu_weakrear010_20261013`, `rear_cpu_weakrear010_continue_20261013`, `rear_cpu_weakrear010_early262_20261013`, `rear_cpu_weakrear010_entropy001_20261013`, `rear_cpu_weakrear010_lr3e5_20261013`, `rear_cpu_weakrear_curriculum025_20261013`, `rear_cpu_weakrear_damage050_20261013`, `rear_cpu_weakrear_received025_20261013`, `rear_cpu_win585_continue_early655_20261013`, `rear_cpu_win586_continue_early131_20261013`, `rear_cpu_win586_epochs1_20261013`, `rear_cpu_win586_epochs1_early393_20261013`, `rear_cpu_win586_epochs2_continue_20261013`.

원본: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_actor_head_only_20261014/plan.json)

| 상대 ID | 추출 확률 |
| --- | ---: |
| `cem_original` | 0.0204966354 |
| `ace` | 0.0216657365 |
| `evader` | 0.0329670471 |
| `fixed_left` | 0.0153307802 |
| `fixed_right` | 0.0152874802 |
| `intercept_s2500_g3_c6` | 0.0153307802 |
| `extend_cem_s6800_g3_c9` | 0.0237410380 |
| `control_s7400_t2621440_a4900` | 0.0212729357 |
| `control_s7601_t1310720_a4900` | 0.0240008382 |
| `temporal_extend_left` | 0.0243905386 |
| `temporal_delayed_left` | 0.0252132393 |
| `temporal_extend_right` | 0.0207966352 |
| `novel_opening_right` | 0.0250732893 |
| `league_contextual_speed_pilot_20261007_c0` | 0.0207865853 |
| `league_contextual_speed_pilot_20261007_c1` | 0.0211762857 |
| `circler` | 0.0282906427 |
| `lead` | 0.0310185452 |
| `pursuit` | 0.0224451372 |
| `league_residual_continuation_20261007_s3301_t2097152` | 0.0174957823 |
| `league_pursuit_context_train_20261007_346fd4060999` | 0.0153307802 |
| `episode_mix_sampled_4901` | 0.0248567891 |
| `ddqn_s0` | 0.0160235809 |
| `symgate_s2400_g0_c1` | 0.0208863358 |
| `league_temporal_repair_20261007_2ca01b55b31e` | 0.0160235809 |
| `history_s6000_t1048576_a4900` | 0.0204966354 |
| `reactive_brake_pursuit_left` | 0.0255928898 |
| `gate_s2302_g1_c9` | 0.0208863358 |
| `portfolio_s6400_g0_c2_a1` | 0.0152874802 |
| `risk_break_cem_s7200_g1_c9` | 0.0298494442 |
| `hold5_s6500_t262144_a4901` | 0.0160235809 |
| `sampled_s3301_t2097152_a4900` | 0.0189679836 |
| `switching_threat_close` | 0.0224420367 |
| `untrained_parameter_02` | 0.0216626360 |
| `reactive_range_escape_right` | 0.0241206884 |
| `switching_turn_strong` | 0.0228317371 |
| `novel_lag_fast` | 0.0226484871 |
| `episode_mix_sampled_4900` | 0.0235144879 |
| `league_maneuver_parallel_20261006_545956914cb9` | 0.0167596816 |
| `league_residual_continuation_20261007_s3300_t4194304` | 0.0160235809 |
| `league_temporal_repair_20261007_6607864eceb4` | 0.0160235809 |
| `sampled_s3300_t2097152_a4900` | 0.0153307802 |
| `hold5_s6501_t262144_a4900` | 0.0152874802 |
| `league_pursuit_context_train_20261007_5a5ecfbeabd6` | 0.0152874802 |
| `portfolio_s6401_g2_c2_a0` | 0.0152874802 |
| `ddqn_s1` | 0.0149410799 |
| `untrained_parameter_04` | 0.0232214375 |
| `control_s7600_t1310720_a4900` | 0.0240008382 |
| `switching_range_wide` | 0.0236111378 |

### P2 — 54개 상대

상대 ID·비중 쌍의 canonical JSON SHA-256: `31555e692ffd6cf4b376a882de4e9bd33ade436d16b3d246a4b6ab6a32fe9c29`.

사용 실행: `aim_cpu_contact_anchor_20261015`, `aim_cpu_encounter_actor_20261015`, `aim_cpu_encounter_actor_v2_20261015`, `aim_cpu_encounter_speed_20261015`, `aim_cpu_encounter_speed_20261015_unlaunched_band_collision`, `aim_cpu_known_mutual_focus_20261015`, `aim_cpu_long_horizon_20261015`, `aim_cpu_long_horizon_v2_20261015`, `aim_cpu_mutual_continue_20261015`, `aim_cpu_mutual_head_20261015`, `aim_cpu_mutual_replica_20261015`, `aim_cpu_opening_coverage_20261015`, `aim_cpu_opening_rehearsal_20261015`, `aim_cpu_post_encounter_20261015`, `aim_cpu_post_encounter_v2_20261015`, `aim_cpu_prefix_diversity_20261015`, `aim_cpu_reengagement_20261015`, `aim_cpu_reengagement_focus_20261015`, `aim_cpu_separate_clip_20261015`.

원본: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_contact_anchor_20261015/plan.json)

| 상대 ID | 추출 확률 |
| --- | ---: |
| `cem_original` | 0.0091096157 |
| `ace` | 0.0113960007 |
| `evader` | 0.0146520209 |
| `fixed_left` | 0.0068136801 |
| `fixed_right` | 0.0067944356 |
| `intercept_s2500_g3_c6` | 0.0068136801 |
| `extend_cem_s6800_g3_c9` | 0.0229190636 |
| `control_s7400_t2621440_a4900` | 0.0359564049 |
| `control_s7601_t1310720_a4900` | 0.0354020215 |
| `temporal_extend_left` | 0.0550098507 |
| `temporal_delayed_left` | 0.0606758488 |
| `temporal_extend_right` | 0.0233772246 |
| `novel_opening_right` | 0.0111436841 |
| `league_contextual_speed_pilot_20261007_c0` | 0.0180724046 |
| `league_contextual_speed_pilot_20261007_c1` | 0.0182456048 |
| `circler` | 0.0125736190 |
| `lead` | 0.0349874335 |
| `pursuit` | 0.0099756166 |
| `league_residual_continuation_20261007_s3301_t2097152` | 0.0077759032 |
| `league_pursuit_context_train_20261007_346fd4060999` | 0.0068136801 |
| `episode_mix_sampled_4901` | 0.0251817375 |
| `ddqn_s0` | 0.0106551604 |
| `symgate_s2400_g0_c1` | 0.0092828159 |
| `league_temporal_repair_20261007_2ca01b55b31e` | 0.0071215915 |
| `history_s6000_t1048576_a4900` | 0.0091096157 |
| `reactive_brake_pursuit_left` | 0.0396431689 |
| `gate_s2302_g1_c9` | 0.0092828159 |
| `portfolio_s6400_g0_c2_a1` | 0.0067944356 |
| `risk_break_cem_s7200_g1_c9` | 0.0256339108 |
| `hold5_s6500_t262144_a4901` | 0.0071215915 |
| `sampled_s3301_t2097152_a4900` | 0.0084302149 |
| `switching_threat_close` | 0.0205749453 |
| `untrained_parameter_02` | 0.0308292517 |
| `reactive_range_escape_right` | 0.0336885039 |
| `switching_turn_strong` | 0.0260484988 |
| `novel_lag_fast` | 0.0348009766 |
| `episode_mix_sampled_4900` | 0.0228183747 |
| `league_maneuver_parallel_20261006_545956914cb9` | 0.0092155318 |
| `league_residual_continuation_20261007_s3300_t4194304` | 0.0071215915 |
| `league_temporal_repair_20261007_6607864eceb4` | 0.0071215915 |
| `sampled_s3300_t2097152_a4900` | 0.0068136801 |
| `hold5_s6501_t262144_a4900` | 0.0067944356 |
| `league_pursuit_context_train_20261007_5a5ecfbeabd6` | 0.0067944356 |
| `portfolio_s6401_g2_c2_a0` | 0.0067944356 |
| `ddqn_s1` | 0.0066404799 |
| `untrained_parameter_04` | 0.0403559746 |
| `control_s7600_t1310720_a4900` | 0.0407023749 |
| `switching_range_wide` | 0.0440627436 |
| `untrained_parameter_01` | 0.0127928282 |
| `hold5_s6500_t262144_a4900` | 0.0092592593 |
| `league_residual_continuation_20261007_s3300_t2097152` | 0.0092592593 |
| `portfolio_s6400_g0_c2_a0` | 0.0092592593 |
| `reactive_brake_pursuit_right` | 0.0322274571 |
| `round_00` | 0.0092592593 |

### P3 — 54개 상대

상대 ID·비중 쌍의 canonical JSON SHA-256: `1f630188d3e8d5e2e9df2a015e843b732dabb32cabe24b7d37a678c86fffdb53`.

사용 실행: `aim_cpu_encounter_loss_focus_20261015`.

원본: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_encounter_loss_focus_20261015/plan.json)

| 상대 ID | 추출 확률 |
| --- | ---: |
| `cem_original` | 0.0089945500 |
| `ace` | 0.0112520551 |
| `evader` | 0.0144669477 |
| `fixed_left` | 0.0067276149 |
| `fixed_right` | 0.0067086135 |
| `intercept_s2500_g3_c6` | 0.0067276149 |
| `extend_cem_s6800_g3_c9` | 0.0226295674 |
| `control_s7400_t2621440_a4900` | 0.0355022309 |
| `control_s7601_t1310720_a4900` | 0.0349548501 |
| `temporal_extend_left` | 0.0543150081 |
| `temporal_delayed_left` | 0.0599094377 |
| `temporal_extend_right` | 0.0230819413 |
| `novel_opening_right` | 0.0110029256 |
| `league_contextual_speed_pilot_20261007_c0` | 0.0178441278 |
| `league_contextual_speed_pilot_20261007_c1` | 0.0180151402 |
| `circler` | 0.0124147986 |
| `lead` | 0.0345454989 |
| `pursuit` | 0.0098496122 |
| `league_residual_continuation_20261007_s3301_t2097152` | 0.0076776839 |
| `league_pursuit_context_train_20261007_346fd4060999` | 0.0067276149 |
| `episode_mix_sampled_4901` | 0.0248636609 |
| `ddqn_s0` | 0.0105205725 |
| `symgate_s2400_g0_c1` | 0.0091655624 |
| `league_temporal_repair_20261007_2ca01b55b31e` | 0.0070316370 |
| `history_s6000_t1048576_a4900` | 0.0089945500 |
| `reactive_brake_pursuit_left` | 0.0391424266 |
| `gate_s2302_g1_c9` | 0.0091655624 |
| `portfolio_s6400_g0_c2_a1` | 0.0067086135 |
| `risk_break_cem_s7200_g1_c9` | 0.0253101227 |
| `hold5_s6500_t262144_a4901` | 0.0070316370 |
| `sampled_s3301_t2097152_a4900` | 0.0083237309 |
| `switching_threat_close` | 0.0203150582 |
| `untrained_parameter_02` | 0.0304398400 |
| `reactive_range_escape_right` | 0.0332629763 |
| `switching_turn_strong` | 0.0257194740 |
| `novel_lag_fast` | 0.0343613971 |
| `episode_mix_sampled_4900` | 0.0225301503 |
| `league_maneuver_parallel_20261006_545956914cb9` | 0.0090991282 |
| `league_residual_continuation_20261007_s3300_t4194304` | 0.0070316370 |
| `league_temporal_repair_20261007_6607864eceb4` | 0.0070316370 |
| `sampled_s3300_t2097152_a4900` | 0.0067276149 |
| `hold5_s6501_t262144_a4900` | 0.0067086135 |
| `league_pursuit_context_train_20261007_5a5ecfbeabd6` | 0.0067086135 |
| `portfolio_s6401_g2_c2_a0` | 0.0067086135 |
| `ddqn_s1` | 0.0065566025 |
| `untrained_parameter_04` | 0.0398462286 |
| `control_s7600_t1310720_a4900` | 0.0401882535 |
| `switching_range_wide` | 0.0435061766 |
| `untrained_parameter_01` | 0.0252624778 |
| `hold5_s6500_t262144_a4900` | 0.0091423033 |
| `league_residual_continuation_20261007_s3300_t2097152` | 0.0091423033 |
| `portfolio_s6400_g0_c2_a0` | 0.0091423033 |
| `reactive_brake_pursuit_right` | 0.0318203844 |
| `round_00` | 0.0091423033 |

### P4 — 54개 상대

상대 ID·비중 쌍의 canonical JSON SHA-256: `101a4785c11a7cea8cc8011417b92c1ed8609770a303734eed8e68b60bbce73d`.

사용 실행: `aim_cpu_known_focus_20261014`.

원본: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_focus_20261014/plan.json)

| 상대 ID | 추출 확률 |
| --- | ---: |
| `cem_original` | 0.0091096157 |
| `ace` | 0.0101840184 |
| `evader` | 0.0629814501 |
| `fixed_left` | 0.0068136801 |
| `fixed_right` | 0.0067944356 |
| `intercept_s2500_g3_c6` | 0.0068136801 |
| `extend_cem_s6800_g3_c9` | 0.0135721618 |
| `control_s7400_t2621440_a4900` | 0.0294275144 |
| `control_s7601_t1310720_a4900` | 0.0329207687 |
| `temporal_extend_left` | 0.0591696685 |
| `temporal_delayed_left` | 0.0704464217 |
| `temporal_extend_right` | 0.0154074170 |
| `novel_opening_right` | 0.0333974136 |
| `league_contextual_speed_pilot_20261007_c0` | 0.0114576908 |
| `league_contextual_speed_pilot_20261007_c1` | 0.0116308910 |
| `circler` | 0.0125736190 |
| `lead` | 0.0226628540 |
| `pursuit` | 0.0102221953 |
| `league_residual_continuation_20261007_s3301_t2097152` | 0.0077759032 |
| `league_pursuit_context_train_20261007_346fd4060999` | 0.0068136801 |
| `episode_mix_sampled_4901` | 0.0172119298 |
| `ddqn_s0` | 0.0081079064 |
| `symgate_s2400_g0_c1` | 0.0092828159 |
| `league_temporal_repair_20261007_2ca01b55b31e` | 0.0071215915 |
| `history_s6000_t1048576_a4900` | 0.0091096157 |
| `reactive_brake_pursuit_left` | 0.0313474940 |
| `gate_s2302_g1_c9` | 0.0092828159 |
| `portfolio_s6400_g0_c2_a1` | 0.0067944356 |
| `risk_break_cem_s7200_g1_c9` | 0.0162870089 |
| `hold5_s6500_t262144_a4901` | 0.0071215915 |
| `sampled_s3301_t2097152_a4900` | 0.0086767937 |
| `switching_threat_close` | 0.0129948279 |
| `untrained_parameter_02` | 0.0234978912 |
| `reactive_range_escape_right` | 0.0211382569 |
| `switching_turn_strong` | 0.0205653896 |
| `novel_lag_fast` | 0.0258470323 |
| `episode_mix_sampled_4900` | 0.0154441026 |
| `league_maneuver_parallel_20261006_545956914cb9` | 0.0075103920 |
| `league_residual_continuation_20261007_s3300_t4194304` | 0.0071215915 |
| `league_temporal_repair_20261007_6607864eceb4` | 0.0071215915 |
| `sampled_s3300_t2097152_a4900` | 0.0068136801 |
| `hold5_s6501_t262144_a4900` | 0.0067944356 |
| `league_pursuit_context_train_20261007_5a5ecfbeabd6` | 0.0067944356 |
| `portfolio_s6401_g2_c2_a0` | 0.0067944356 |
| `ddqn_s1` | 0.0066404799 |
| `untrained_parameter_04` | 0.0552596106 |
| `control_s7600_t1310720_a4900` | 0.0353249112 |
| `switching_range_wide` | 0.0376791429 |
| `untrained_parameter_01` | 0.0575886884 |
| `hold5_s6500_t262144_a4900` | 0.0092592593 |
| `league_residual_continuation_20261007_s3300_t2097152` | 0.0092592593 |
| `portfolio_s6400_g0_c2_a0` | 0.0092592593 |
| `reactive_brake_pursuit_right` | 0.0315129888 |
| `round_00` | 0.0092592593 |

### P5 — 54개 상대

상대 ID·비중 쌍의 canonical JSON SHA-256: `7ae11d3192ffc1c041f9b1af527554db468fd2faf5c154b1f2731bd2e5a0de8e`.

사용 실행: `aim_cpu_known_long2_20261014`, `aim_cpu_known_long_20261014`.

원본: [plan.json](/home/hoseon/Desktop/박찬서/RL_KAU/.worktrees/rear-async/aircombat-rl/runs/aim_cpu_known_long2_20261014/plan.json)

| 상대 ID | 추출 확률 |
| --- | ---: |
| `cem_original` | 0.0182192315 |
| `ace` | 0.0192584325 |
| `evader` | 0.0293040418 |
| `fixed_left` | 0.0136273602 |
| `fixed_right` | 0.0135888713 |
| `intercept_s2500_g3_c6` | 0.0136273602 |
| `extend_cem_s6800_g3_c9` | 0.0211031449 |
| `control_s7400_t2621440_a4900` | 0.0189092761 |
| `control_s7601_t1310720_a4900` | 0.0213340784 |
| `temporal_extend_left` | 0.0216804787 |
| `temporal_delayed_left` | 0.0224117683 |
| `temporal_extend_right` | 0.0184858980 |
| `novel_opening_right` | 0.0222873683 |
| `league_contextual_speed_pilot_20261007_c0` | 0.0184769647 |
| `league_contextual_speed_pilot_20261007_c1` | 0.0188233651 |
| `circler` | 0.0251472380 |
| `lead` | 0.0275720402 |
| `pursuit` | 0.0199512331 |
| `league_residual_continuation_20261007_s3301_t2097152` | 0.0155518064 |
| `league_pursuit_context_train_20261007_346fd4060999` | 0.0136273602 |
| `episode_mix_sampled_4901` | 0.0220949237 |
| `ddqn_s0` | 0.0142431830 |
| `symgate_s2400_g0_c1` | 0.0185656318 |
| `league_temporal_repair_20261007_2ca01b55b31e` | 0.0142431830 |
| `history_s6000_t1048576_a4900` | 0.0182192315 |
| `reactive_brake_pursuit_left` | 0.0227492354 |
| `gate_s2302_g1_c9` | 0.0185656318 |
| `portfolio_s6400_g0_c2_a1` | 0.0135888713 |
| `risk_break_cem_s7200_g1_c9` | 0.0265328392 |
| `hold5_s6500_t262144_a4901` | 0.0142431830 |
| `sampled_s3301_t2097152_a4900` | 0.0168604299 |
| `switching_threat_close` | 0.0199484771 |
| `untrained_parameter_02` | 0.0192556765 |
| `reactive_range_escape_right` | 0.0214406119 |
| `switching_turn_strong` | 0.0202948774 |
| `novel_lag_fast` | 0.0201319885 |
| `episode_mix_sampled_4900` | 0.0209017670 |
| `league_maneuver_parallel_20261006_545956914cb9` | 0.0148974947 |
| `league_residual_continuation_20261007_s3300_t4194304` | 0.0142431830 |
| `league_temporal_repair_20261007_6607864eceb4` | 0.0142431830 |
| `sampled_s3300_t2097152_a4900` | 0.0136273602 |
| `hold5_s6501_t262144_a4900` | 0.0135888713 |
| `league_pursuit_context_train_20261007_5a5ecfbeabd6` | 0.0135888713 |
| `portfolio_s6401_g2_c2_a0` | 0.0135888713 |
| `ddqn_s1` | 0.0132809599 |
| `untrained_parameter_04` | 0.0206412778 |
| `control_s7600_t1310720_a4900` | 0.0213340784 |
| `switching_range_wide` | 0.0209876781 |
| `untrained_parameter_01` | 0.0185185185 |
| `hold5_s6500_t262144_a4900` | 0.0185185185 |
| `league_residual_continuation_20261007_s3300_t2097152` | 0.0185185185 |
| `portfolio_s6400_g0_c2_a0` | 0.0185185185 |
| `reactive_brake_pursuit_right` | 0.0185185185 |
| `round_00` | 0.0185185185 |

## 검증 기록

- 계획서 122개 JSON 읽기 성공; 작성 전후 SHA-256 일치.
- 선정 모델 7개 × 파일 4종 = 28개 파일 존재·SHA-256 확인.
- 상대 추출 비중 5개 고유 벡터의 길이·확률 합 확인.
- 학습 완료 여부·현재 프로세스 상태는 이번 파일 작성 범위에서 새로 판정하지 않았다.
- 코드·가중치·계획서 변경 없음. 추가 학습·평가 및 GitHub 게시 없음.

