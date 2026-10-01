# A3 고정 potential 보상 비교 — 사전 프로토콜

작성일: 2026-10-01. 새 시험 결과를 보기 전에 고정한다.

## 질문과 대조군

FairFight의 31개 관측 DQN에서 종료 보상에 potential shaping을 추가하면
동일한 예산의 공식 승률과 마지막 모델의 성능 유지가 개선되는지 확인한다.
새 PC는 Ryzen 7 9800X3D, RAM 32 GB, RTX 5080이다. CPU PyTorch를 사용하고,
기존 Intel PC의 결과는 참고 자료로 보존한다. 주 비교 대조군은 이 PC에서 다시 학습한다.

terminal / potential_v1 각각 seed 0·1·2, 1,024,000스텝을 새로 학습한다.
총 6,144,000스텝이다. 두 조건의 네트워크·관측·행동·상대·물리·초기 seed·
최적화·학습 좌석 red·검증 규칙은 같다. CPU는 실행별 1스레드다.
기존 버퍼와 가중치로 초기화하지 않는다.

- DQN MLP 64·64 ReLU, gamma 0.999, lr 0.001
- replay 50000, batch 32, learning starts 1000, train_freq 4
- gradient_steps 1, target update 1000
- epsilon은 처음 20480스텝에 1 → 0.05, 이후 고정
- 원래 보상은 승리 +1, 그 외 경기 종료 -0.2, 진행 중 0

## potential 정의

기존 31개 정규화 관측만 사용한다. 새 정책 입력은 추가하지 않는다.

```text
proximity = 10000 / (range + 10000)
nose_advantage = (cos(own ATA) - cos(opponent ATA)) / 2
Phi = 0.25 * proximity * nose_advantage
    + 0.125 * (own_health - opponent_health)
    + 0.125 * (own_track_fraction - opponent_track_fraction)
r_shaped = r_terminal + 0.999 * Phi(next) - Phi(current)
```

Phi는 [-0.5, 0.5] 범위다. 거리 자체를 매 스텝 더하는 보상이 아니라
현재/다음 potential 차이를 사용한다. 계수는 이번 실험 도중 변경하지 않는다.
양 기수 방향 차이를 포함하므로 단순히 적에게 접근하는 것만 유리하게 보지 않는다.

reset 직후 Phi를 저장한다. kill/died/mutual 및 실제 timeout에서 다음 Phi를 0으로 둔다.
데이터 수집 때문에 외부에서 끊긴 transition은 다음 Phi를 유지하고 bootstrap한다.
할인된 shaping의 합은 실제 종료 에피소드에서 -Phi(initial)로 상쇄된다.
이는 학습 효율 향상을 보장하지 않으며 공식 평가 결과로 검증할 가설이다.

## 검증, 선택, 시험

- 기존 검증 band 900000, 20경기, 매 51200스텝.
- 204800 / 512000 / 1024000 예산 이내의 검증 선택 모델과 각 예산 끝 모델을 모두 보고한다.
- 예산 내 선택은 검증 승리 수 → 평균 승리 시간 → 이른 스텝 순이다.
- 조건 선택은 100만 예산의 seed별 선택 모델들의 검증 승리 수 합 → 승리 경기 가중 평균 시간 순이다.
  완전 동점이면 단순한 terminal 조건을 유지한다.
- 선택 조건 안에서 대표 모델은 검증 승리 수 → 평균 승리 시간 → 이른 스텝 → 낮은 seed 순으로 고른다.
- 학습 6회와 모든 선택을 마친 뒤 새 시험 band **1300000**, 양 좌석 총 40경기를 연다.
  두 좌석은 같은 초기 seed 20개를 사용한다.
- 선택 기록에 가중치 SHA-256을 저장하고 시험 전에 고정한다.
- 시험 결과로 대표 모델을 다시 고르거나 potential 계수를 수정하지 않는다.
- 40경기를 독립 학습 40회처럼 취급하지 않는다. 3개 학습 seed의 평균·표본 표준편차와
  seed별 대응 차이를 보고하며 초기 선별로 해석한다.

## 결과에 따른 다음 단계

potential 조건의 선택 모델 평균 시험 승률이 대조군보다 10%p 이상 높고,
3개 중 적어도 2개 seed가 개선되며, 마지막 모델의 평균 승률도 나빠지지 않으면
새 독립 학습 seed를 이용한 확인 실험으로 진행한다. 조건을 만족하지 않으면
검증 이력과 학습 실패 양상을 분석하고, 보상·탐험을 동시에 바꾸지 않는 다음 실험을 정한다.
이 기준은 자동 채택 규칙이 아니다. 최종 시험을 반복 최적화에 사용하지 않는다.

## 실행과 보관

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tools.plan_a_shaping smoke --out runs/a3_smoke_20261001 --workers 2
.\.venv\Scripts\python.exe -X utf8 -m tools.plan_a_shaping run --out runs/plan_a_20261001/shaping --workers 3
```

smoke는 seed 0에서 두 조건을 2048스텝 학습하고 이미 사용한 검증 band만 평가한다.
본 실행은 plan.json·패키지 버전·코드 해시·소스 복사본을 학습 전에 저장한다.
모든 검증 가중치와 세 예산 시점의 replay/RNG를 남긴다.
중단된 학습을 같은 폴더에서 몰래 재시작하지 않는다. 새 출력 폴더를 사용한다.
실험 폴더는 기본적으로 Git ignored다. 원본 duration 자료를 덮어쓰지 않는다.
