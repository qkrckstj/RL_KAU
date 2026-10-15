# 기존 모델 승률 순위 — 계열별 대표 1개

생성 시각: 2026-10-15T22:07:38+09:00.

**학습 실행 88개에서 평가된 체크포인트 674개 → 7개 계열 대표로 통합.** 완료 평가 기록 1128건을 확인했으며, 반복 인용된 기준 모델은 중복 집계하지 않았다.

## 선정·통합 기준

- 앞서 요청한 전체 모델 재평가·기준별 상위 10개 수집 대신, 최신 요청에 따라 기존 기록을 계열별로 통합한 순위다. 새 경기는 실행하지 않았다.
- 정렬 기준은 **기존 경기 승률 = 승 / 전체 경기 수**다. 무승부도 분모에 포함한다. 계열마다 경기 수와 무관하게 가장 높은 기록 하나를 고른다. 서로 다른 평가를 합산하거나 가중치를 주지 않는다. 동률이면 경기 수가 많은 기록, 이후 체크포인트·근거 경로 사전순으로 선택한다.
- 계열은 체크포인트의 `greedy/policy_net.json`에 기록된 정책 종류·관측 구조로 넓게 묶었다. 교전 이후 속도 탐색은 행동 분포가 달라 별도로 둔다. 같은 계열의 학습 난수·중간 체크포인트·설정 변형은 통합한다. **실제 행동 유사도를 측정해 같다고 판정한 것은 아니다.**
- 범위는 현재 작업 공간 `runs/`의 완료된 원본 CPU PPO 중간·개발·전체 54상대 평가다. 경기 수는 128·672·768·1,728경기다. 작동 확인용 2·4·96경기 기록 10건은 제외했다. 이전 CEM/DQN 및 별도 GPU 기록까지 포함한 전체 저장소 순위는 아니다.
- 같은 128경기도 두 종류가 있으며 상대·초기조건이 다르다. 따라서 **기록된 승률의 순서이며, 동일 조건에서 측정한 실력 순위는 아니다.** 여러 시도 중 최고값을 고르므로 시도가 많았던 계열에 유리할 수 있다.
- 요청했던 상대전적 기준(한 번이라도 패하면 상대 패배, 패배 없이 승리가 있으면 상대 승리, 전부 무승부면 미결)은 아래에 참고용으로 함께 적었다. 이번 표의 정렬에는 쓰지 않으며, 이 보고서로 기준 모델을 교체하지 않는다.
- 기존 체크포인트·경기 원본·학습 설정·실행 중 프로세스를 보존했다.

## 대표 모델 순위

| 순위 | 계열 | 대표 모델 | 기록 승률 | 승 / 무 / 패 | 경기 수 | 평가 종류 | 상대 수 | 통합 모델 수 |
| ---: | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 후방·잔차 PPO | [s44701_step_002621440](../artifacts/20261015/models/s44701_step_002621440) | 86.72% | 111 / 16 / 1 | 128 | monitor | 8 | 325 |
| 2 | 예측 조준 PPO | [s45800_step_002621440](../artifacts/20261015/models/s45800_step_002621440) | 85.94% | 110 / 16 / 2 | 128 | monitor | 8 | 313 |
| 3 | 교전 이후 분리 PPO | [s47236_step_002621440](../artifacts/20261015/models/s47236_step_002621440) | 76.33% | 1319 / 309 / 100 | 1,728 | all54 | 54 | 8 |
| 4 | 교전 이후 속도 탐색 PPO | [s47238_step_010485760](../artifacts/20261015/models/s47238_step_010485760) | 75.45% | 507 / 119 / 46 | 672 | development | 21 | 4 |
| 5 | 상태별 사전 정책 조정형 | [s46201_step_005242880](../artifacts/20261015/models/s46201_step_005242880) | 73.66% | 495 / 130 / 47 | 672 | development | 21 | 8 |
| 6 | 선회율 입력 보강형 | [s46401_step_005242880](../artifacts/20261015/models/s46401_step_005242880) | 73.51% | 494 / 128 / 50 | 672 | development | 21 | 8 |
| 7 | 기수 방향 이력 보강형 | [s46500_step_005242880](../artifacts/20261015/models/s46500_step_005242880) | 71.73% | 482 / 135 / 55 | 672 | development | 21 | 8 |

## 대표 모델 근거 및 통합된 학습 실행

### 1. 후방·잔차 PPO

40차원 잔차 정책. 후방·피해량 보상, 학습률, 엔트로피, 이어 학습 변형을 통합했다.

- 메타데이터 종류: `residual_ppo_actor`.
- 대표 체크포인트: [s44701_step_002621440](../artifacts/20261015/models/s44701_step_002621440).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s44701_step_002621440/completion.json).
- 근거 SHA-256: `5c76d28746ed0040dffc4cfefeb6c075ff3b6ab18f777998efcf9f0a3442d13f`.
- 평가 구성: 8 opponents x 8 initial conditions x 2 seats x 1 action seeds = 128 games.
- 초기조건 난수: `[1702500000, 1702500001, 1702500002, 1702500003, 1702500004, 1702500005, 1702500006, 1702500007]`; seats: `['blue', 'red']`; action seeds: `[4900]`.
- 상대 목록: `ace`, `circler`, `extend_cem_s6800_g3_c9`, `fixed_right`, `lead`, `league_residual_continuation_20261007_s3301_t2097152`, `pursuit`, `switching_turn_strong`.
- 이 평가의 상대전적 승 / 패 / 미결: **7 / 1 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (32):

  - `rear_cpu_actualrear010_20261013`
  - `rear_cpu_balanced582_clip005_20261013`
  - `rear_cpu_balanced582_continue_20261013`
  - `rear_cpu_clip005_win585_rear12_continue_20261013`
  - `rear_cpu_curriculum025_20261012`
  - `rear_cpu_damage0125_20261012`
  - `rear_cpu_damage0125_continue_20261012`
  - `rear_cpu_damage0125_lr3e5_20261012`
  - `rear_cpu_damage050_win585_continue_20261013`
  - `rear_cpu_entropy003_20261012`
  - `rear_cpu_entropy005_20261012`
  - `rear_cpu_entropy006_20261012`
  - `rear_cpu_entropy01_20261012`
  - `rear_cpu_joint585_epochs2_20261013`
  - `rear_cpu_joint585_lr3e5_20261013`
  - `rear_cpu_joint585_potential010_20261013`
  - `rear_cpu_lr3e5_20261012`
  - `rear_cpu_lr3e5_continue_20261012`
  - `rear_cpu_main_20261012`
  - `rear_cpu_position0005_20261013`
  - `rear_cpu_position002_20261012`
  - `rear_cpu_rearwin025_20261012`
  - `rear_cpu_rearwin075_20261012`
  - `rear_cpu_weakrear010_20261013`
  - `rear_cpu_weakrear010_continue_20261013`
  - `rear_cpu_weakrear010_entropy001_20261013`
  - `rear_cpu_weakrear010_lr3e5_20261013`
  - `rear_cpu_weakrear_curriculum025_20261013`
  - `rear_cpu_weakrear_damage050_20261013`
  - `rear_cpu_weakrear_received025_20261013`
  - `rear_cpu_win586_epochs1_20261013`
  - `rear_cpu_win586_epochs2_continue_20261013`

### 2. 예측 조준 PPO

43차원 예측 조준 정책. 보상·커리큘럼·상대 가중치·행동망 동결·PPO 설정 변형을 통합했다.

- 메타데이터 종류: `predictive_aim_ppo_actor`.
- 대표 체크포인트: [s45800_step_002621440](../artifacts/20261015/models/s45800_step_002621440).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s45800_step_002621440/completion.json).
- 근거 SHA-256: `a66ef6624009c788976f8f7a9418f981841e0e7437e60a9eb8daf78dd6ad3ccd`.
- 평가 구성: 8 opponents x 8 initial conditions x 2 seats x 1 action seeds = 128 games.
- 초기조건 난수: `[1702500000, 1702500001, 1702500002, 1702500003, 1702500004, 1702500005, 1702500006, 1702500007]`; seats: `['blue', 'red']`; action seeds: `[4900]`.
- 상대 목록: `ace`, `circler`, `extend_cem_s6800_g3_c9`, `fixed_right`, `lead`, `league_residual_continuation_20261007_s3301_t2097152`, `pursuit`, `switching_turn_strong`.
- 이 평가의 상대전적 승 / 패 / 미결: **6 / 2 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (50):

  - `aim_cpu_actor_head_only_20261014`
  - `aim_cpu_best498_continue_20261013`
  - `aim_cpu_contact_anchor_20261015`
  - `aim_cpu_control_guidance_20261013`
  - `aim_cpu_current_anchor504_20261014`
  - `aim_cpu_current_anchor508_20261014`
  - `aim_cpu_dealt0625_20261014`
  - `aim_cpu_draw_strength050_20261014`
  - `aim_cpu_draw_utility025_resume_20261014`
  - `aim_cpu_enemy_los_control_20261014`
  - `aim_cpu_entropy001_20261014`
  - `aim_cpu_entropy003_20261013`
  - `aim_cpu_entropy005_resume_v2_20261014`
  - `aim_cpu_feature_train1e6_20261014`
  - `aim_cpu_guidance_range12k_20261013`
  - `aim_cpu_head_rate1e6_20261014`
  - `aim_cpu_head_rate5e6_20261014`
  - `aim_cpu_heading_rate_20261013`
  - `aim_cpu_known_focus_20261014`
  - `aim_cpu_known_long2_20261014`
  - `aim_cpu_known_long_20261014`
  - `aim_cpu_known_mutual_focus_20261015`
  - `aim_cpu_long_credit0999_20261014`
  - `aim_cpu_long_horizon_v2_20261015`
  - `aim_cpu_mixed_start_20261013`
  - `aim_cpu_mutual_continue_20261015`
  - `aim_cpu_mutual_head_20261015`
  - `aim_cpu_mutual_replica_20261015`
  - `aim_cpu_opening_coverage_20261015`
  - `aim_cpu_opening_rehearsal_20261015`
  - `aim_cpu_post_encounter_v2_20261015`
  - `aim_cpu_predictive_20261013`
  - `aim_cpu_predictive_enemy_20261014`
  - `aim_cpu_predictive_enemy_lr1e5_resume_20261014`
  - `aim_cpu_prefix_diversity_20261015`
  - `aim_cpu_prior3_20261013`
  - `aim_cpu_received075_resume_20261014`
  - `aim_cpu_received0875_20261014`
  - `aim_cpu_received1_20261013`
  - `aim_cpu_reengagement_20261015`
  - `aim_cpu_reengagement_focus_20261015`
  - `aim_cpu_reference510_v2_20261014`
  - `aim_cpu_reference_anchor01_20261014`
  - `aim_cpu_relative_reward_20261013`
  - `aim_cpu_selected504_resume_20261014`
  - `aim_cpu_selected510_resume_20261014`
  - `aim_cpu_separate_clip_20261015`
  - `aim_cpu_short500_resume_20261014`
  - `aim_cpu_short508_resume_20261014`
  - `aim_cpu_start025_20261014`

### 3. 교전 이후 분리 PPO

초기 정책을 보존하고 공개 관측의 사거리 진입·이탈 이후 별도 정책을 학습하는 계열. 상대 가중치 변형을 통합했다.

- 메타데이터 종류: `encounter_isolated_ppo_actor`.
- 대표 체크포인트: [s47236_step_002621440](../artifacts/20261015/models/s47236_step_002621440).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s47236_step_002621440/completion.json); role `candidate`.
- 근거 SHA-256: `1f44a1344e42fea27a1fdd68a65327334ac937335d1a5f5060e6be79937acf64`.
- 평가 구성: 54 opponents x 8 initial conditions x 2 seats x 2 action seeds = 1728 games.
- 초기조건 난수: `[1702600000, 1702600001, 1702600002, 1702600003, 1702600004, 1702600005, 1702600006, 1702600007]`; seats: `['blue', 'red']`; action seeds: `[4900, 4901]`.
- 상대 목록: `ace`, `cem_original`, `circler`, `control_s7400_t2621440_a4900`, `control_s7600_t1310720_a4900`, `control_s7601_t1310720_a4900`, `ddqn_s0`, `ddqn_s1`, `episode_mix_sampled_4900`, `episode_mix_sampled_4901`, `evader`, `extend_cem_s6800_g3_c9`, `fixed_left`, `fixed_right`, `gate_s2302_g1_c9`, `history_s6000_t1048576_a4900`, `hold5_s6500_t262144_a4900`, `hold5_s6500_t262144_a4901`, `hold5_s6501_t262144_a4900`, `intercept_s2500_g3_c6`, `lead`, `league_contextual_speed_pilot_20261007_c0`, `league_contextual_speed_pilot_20261007_c1`, `league_maneuver_parallel_20261006_545956914cb9`, `league_pursuit_context_train_20261007_346fd4060999`, `league_pursuit_context_train_20261007_5a5ecfbeabd6`, `league_residual_continuation_20261007_s3300_t2097152`, `league_residual_continuation_20261007_s3300_t4194304`, `league_residual_continuation_20261007_s3301_t2097152`, `league_temporal_repair_20261007_2ca01b55b31e`, `league_temporal_repair_20261007_6607864eceb4`, `novel_lag_fast`, `novel_opening_right`, `portfolio_s6400_g0_c2_a0`, `portfolio_s6400_g0_c2_a1`, `portfolio_s6401_g2_c2_a0`, `pursuit`, `reactive_brake_pursuit_left`, `reactive_brake_pursuit_right`, `reactive_range_escape_right`, `risk_break_cem_s7200_g1_c9`, `round_00`, `sampled_s3300_t2097152_a4900`, `sampled_s3301_t2097152_a4900`, `switching_range_wide`, `switching_threat_close`, `switching_turn_strong`, `symgate_s2400_g0_c1`, `temporal_delayed_left`, `temporal_extend_left`, `temporal_extend_right`, `untrained_parameter_01`, `untrained_parameter_02`, `untrained_parameter_04`.
- 이 평가의 상대전적 승 / 패 / 미결: **31 / 23 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (2):

  - `aim_cpu_encounter_actor_v2_20261015`
  - `aim_cpu_encounter_loss_focus_20261015`

### 4. 교전 이후 속도 탐색 PPO

교전 분리 구조에 별도의 조건부 속도 탐색 분포를 적용하므로 기본 교전 분리형과 구분했다.

- 메타데이터 종류: `encounter_isolated_ppo_actor__conditional_speed`.
- 대표 체크포인트: [s47238_step_010485760](../artifacts/20261015/models/s47238_step_010485760).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s47238_step_010485760/completion.json).
- 근거 SHA-256: `1806920538e6e225251ff045f40cae97206af1035423a717603345b48ff7ceb1`.
- 평가 구성: 21 opponents x 8 initial conditions x 2 seats x 2 action seeds = 672 games.
- 초기조건 난수: `[1702600000, 1702600001, 1702600002, 1702600003, 1702600004, 1702600005, 1702600006, 1702600007]`; seats: `['blue', 'red']`; action seeds: `[4900, 4901]`.
- 상대 목록: `ace`, `cem_original`, `circler`, `control_s7600_t1310720_a4900`, `ddqn_s1`, `episode_mix_sampled_4900`, `evader`, `gate_s2302_g1_c9`, `history_s6000_t1048576_a4900`, `hold5_s6500_t262144_a4900`, `lead`, `league_contextual_speed_pilot_20261007_c0`, `league_pursuit_context_train_20261007_346fd4060999`, `league_residual_continuation_20261007_s3300_t2097152`, `portfolio_s6400_g0_c2_a0`, `pursuit`, `reactive_brake_pursuit_right`, `risk_break_cem_s7200_g1_c9`, `round_00`, `temporal_extend_left`, `untrained_parameter_01`.
- 이 평가의 상대전적 승 / 패 / 미결: **11 / 10 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (1):

  - `aim_cpu_encounter_speed_20261015`

### 5. 상태별 사전 정책 조정형

상태에 따라 기존 사전 정책과의 충돌을 조정하는 계열.

- 메타데이터 종류: `state_conflict_prior_aim_ppo_actor`.
- 대표 체크포인트: [s46201_step_005242880](../artifacts/20261015/models/s46201_step_005242880).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s46201_step_005242880/completion.json).
- 근거 SHA-256: `4aa8901dbc6d9d44b4e288e491374c04e91167c8f9d39b439f21c58f8775b5ec`.
- 평가 구성: 21 opponents x 8 initial conditions x 2 seats x 2 action seeds = 672 games.
- 초기조건 난수: `[1702600000, 1702600001, 1702600002, 1702600003, 1702600004, 1702600005, 1702600006, 1702600007]`; seats: `['blue', 'red']`; action seeds: `[4900, 4901]`.
- 상대 목록: `ace`, `cem_original`, `circler`, `control_s7600_t1310720_a4900`, `ddqn_s1`, `episode_mix_sampled_4900`, `evader`, `gate_s2302_g1_c9`, `history_s6000_t1048576_a4900`, `hold5_s6500_t262144_a4900`, `lead`, `league_contextual_speed_pilot_20261007_c0`, `league_pursuit_context_train_20261007_346fd4060999`, `league_residual_continuation_20261007_s3300_t2097152`, `portfolio_s6400_g0_c2_a0`, `pursuit`, `reactive_brake_pursuit_right`, `risk_break_cem_s7200_g1_c9`, `round_00`, `temporal_extend_left`, `untrained_parameter_01`.
- 이 평가의 상대전적 승 / 패 / 미결: **12 / 9 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (1):

  - `aim_cpu_state_prior_20261013`

### 6. 선회율 입력 보강형

선회율 입력을 추가한 관측 구조.

- 메타데이터 종류: `predictive_aim_rate_input_ppo_actor`.
- 대표 체크포인트: [s46401_step_005242880](../artifacts/20261015/models/s46401_step_005242880).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s46401_step_005242880/completion.json).
- 근거 SHA-256: `c24216718ef2b7b502147c65c9fe177e6206209aeb13d8966545259b0ea898c9`.
- 평가 구성: 21 opponents x 8 initial conditions x 2 seats x 2 action seeds = 672 games.
- 초기조건 난수: `[1702600000, 1702600001, 1702600002, 1702600003, 1702600004, 1702600005, 1702600006, 1702600007]`; seats: `['blue', 'red']`; action seeds: `[4900, 4901]`.
- 상대 목록: `ace`, `cem_original`, `circler`, `control_s7600_t1310720_a4900`, `ddqn_s1`, `episode_mix_sampled_4900`, `evader`, `gate_s2302_g1_c9`, `history_s6000_t1048576_a4900`, `hold5_s6500_t262144_a4900`, `lead`, `league_contextual_speed_pilot_20261007_c0`, `league_pursuit_context_train_20261007_346fd4060999`, `league_residual_continuation_20261007_s3300_t2097152`, `portfolio_s6400_g0_c2_a0`, `pursuit`, `reactive_brake_pursuit_right`, `risk_break_cem_s7200_g1_c9`, `round_00`, `temporal_extend_left`, `untrained_parameter_01`.
- 이 평가의 상대전적 승 / 패 / 미결: **11 / 10 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (1):

  - `aim_cpu_rate_input_20261014`

### 7. 기수 방향 이력 보강형

기수 방향 이력을 추가한 관측 구조.

- 메타데이터 종류: `predictive_aim_heading_memory_ppo_actor`.
- 대표 체크포인트: [s46500_step_005242880](../artifacts/20261015/models/s46500_step_005242880).
- 완료 평가: [원본 JSON](../artifacts/20261015/evidence/s46500_step_005242880/completion.json).
- 근거 SHA-256: `9dd04513a3fcbf9ebf84172a451de5b0d19e8cb034612ce60ff93e87ce62554f`.
- 평가 구성: 21 opponents x 8 initial conditions x 2 seats x 2 action seeds = 672 games.
- 초기조건 난수: `[1702600000, 1702600001, 1702600002, 1702600003, 1702600004, 1702600005, 1702600006, 1702600007]`; seats: `['blue', 'red']`; action seeds: `[4900, 4901]`.
- 상대 목록: `ace`, `cem_original`, `circler`, `control_s7600_t1310720_a4900`, `ddqn_s1`, `episode_mix_sampled_4900`, `evader`, `gate_s2302_g1_c9`, `history_s6000_t1048576_a4900`, `hold5_s6500_t262144_a4900`, `lead`, `league_contextual_speed_pilot_20261007_c0`, `league_pursuit_context_train_20261007_346fd4060999`, `league_residual_continuation_20261007_s3300_t2097152`, `portfolio_s6400_g0_c2_a0`, `pursuit`, `reactive_brake_pursuit_right`, `risk_break_cem_s7200_g1_c9`, `round_00`, `temporal_extend_left`, `untrained_parameter_01`.
- 이 평가의 상대전적 승 / 패 / 미결: **13 / 8 / 0**. 평가 조건에 따른 값이며 위 승률 순위 정렬에는 사용하지 않았다.
- 통합한 학습 실행 (1):

  - `aim_cpu_heading_memory_20261014`

## 검증

대표 7개의 기존 원본 경기 4,672건을 확인했다. 경기 수·승수, 상대×초기조건×진영×행동 난수 조합의 중복 및 누락 여부를 검사하고, 종료 체력을 소수점 네 자리로 반올림하여 모든 승패를 다시 계산했다. 집계 대상 전체 완료 요약의 승수/경기 수와 승률도 대조했다. 나머지 모델은 완료 요약으로 선별했으며 모든 원본 경기를 다시 읽지는 않았다.

추가 시뮬레이션: **0경기**. 모델 삭제: **0개**. 학습·평가 코드 또는 설정 변경: **0건**.
