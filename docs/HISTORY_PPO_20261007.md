# 시간 관측을 추가한 PPO 비교 학습

## 설정 감사로 앞선 관측 단독 효과 해석 철회

`league_history_matched_20261007`의 학습은 종료 코드0으로 완료됐지만, 시간 관측 모델의 prototype에서 PPO 설정 일부가 기본값으로 남아 있음을 확인했다. 가중치/Adam/초기 동작 복제는 맞았으나 할인율 등 학습 알고리즘이 같지 않았다. 아래 첫 결과를 관측 방식만의 효과로 해석한 설명은 철회한다. 원본 소스·동결 계획·모델·원시 결과를 덮어쓰지 않고 `treatment_config_audit.json`에 실제 차이와 해석 제한을 기록했다.

|설정|현재 관측 실제값|첫 시간 관측 실제값|
|---|---:|---:|
|gamma|0.999|0.99|
|GAE lambda|0.99|0.95|
|학습 epochs|4|10|
|entropy 계수|0.005|0|
|clip 범위|0.1|0.2|
|target KL|0.015|없음|

학습 wrapper의 shaping gamma는0.999였으므로 첫 시간 관측의 learner gamma와도 달랐다. 앞선 후보는 공식 규칙에서 작동하는 모델로 보존하지만, 관측만 바꾼 공정한 ablation 결과라고 부르지 않는다. 통합 점검은 정책/optimizer 복원과 초기 경기 동등성을 확인했지만 전체 PPO 설정 보존은 검사하지 못했다.

수정 `league_history_clone_matched_checks_20261007`은 원본의 PPO 설정 전체를 복사하고 학습률·clip schedule과 save/load 이후 설정까지 일치함을 검증했다.2,048개 입력의 초기 actor/critic 및 기존 Adam 상태 보존도 다시 통과했다. 신규 controller `tools/league_history_config_matched_train.py`는 각 learner 로드 직후 실제 설정과 topology를 검사하고 `loaded_ppo_configuration.json`에 남긴다. 다르면 학습 시작 전에 차단한다.

수정 비교는 새로운 RNG6000/6001, 새 학습 대역276M/277M, 새85M 개발 조건으로 진행한다. 이전 예약을 우회하거나 재사용하지 않는다. 두 방식에 각각 같은1,048,576단계씩 네 pilot, 합계4,194,304단계를 배정한다. 수정된 smoke 출력은 `league_history_config_smoke_20261007`, 본 출력은 `league_history_config_matched_20261007`이다. 앞선 네 pilot의 비용/결과는 별도 실험으로 남긴다. 최종 시험·보류 상대·GitHub 업로드는 여전히 제외한다.

---

첫 RNG5900의두 방식이 각각1,048,576단계를 완료했다.84M 개발 패널 승률은 같은 초기 기준161/256(62.89%)에서 control181/256(70.70%), history182/256(71.09%)다. 상대 그룹 균등 승률은 control74.77%, history76.94%, 더 낮은 복제의하위 사분위 점수는31.25%와34.38%였다. 우측 이탈 상대는둘 다0/8승이며 우측 weave에는 control4/8, history1/8승으로 상대별 변화가 고르지 않다. 첫 시드만으로 시간 관측이 우월하다고 확정하지 않는다. 학습 시간은199.31초와240.55초로 이번 기록에서 시간 관측이 더 오래 걸렸다. 이는 실제 서로 다른 학습 궤적의 처리 시간이며 동일 고정 workload의 하드웨어 benchmark가 아니다. `first_rng_analysis.json`에 원시 평가 해시와 재계산 결과를 보존했다. 두 번째 RNG5901을 history → control 순으로 자동 진행한다.

새 조건83M의4,800경기 진단에서 유지한 후보의 승률은64.81%, 이전 기준은61.00%였다. 코드 그룹 균등 승률은 후보67.43%, 기준70.00%로 오히려 낮았다. 후보와 기준의 차이에 대한 조건/행동 복제 bootstrap 구간은0을 포함했다. 우측 이탈 상대는 후보도0승16패, evader는16무승부였다. Ace는 기준16승에서 후보9승7무로 내려갔다. 따라서 새 최강 정책으로 승격하지 않고 관측 전략을 비교한다. 알려진 상대의 새로운 초기조건 진단이지 미사용 상대 또는 최종 시험은 아니다.

새 전략은 현재 상대 상태31개, 최근1초·5초의 상태 변화 각31개, 현재 teacher 행동9개를 사용한다. 입력40개가102개로 늘어난다. 변화량은절반으로 정규화한다. 관측이 부족한 초기에는 가장 오래된 해당 경기 관측을 사용한다. 새 경기에서는 이력을 초기화하며, 미래 관측·상대ID·비공개 시드는 사용하지 않는다. 공식 FairFight·승패·기존 보상·원래 CEM teacher·prior bias6·상대100개와 선택 분포는 유지한다.

새 입력 열의 가중치와 Adam moment를0으로 만들고 기존 열·다른 가중치·Adam moment와 step을 그대로 옮겼다.2,048개 합성 입력에서 초기 행동 확률과 가치 출력이 기존 모델과 같았다. NumPy actor의 최대 확률 오차는2.38e-7이었다. learner/optimizer 저장 복원도 통과했다. 이것만으로 비행 성능 향상을 증명하지 않는다.

통합 실행 `league_history_smoke_20261007`은 종료 코드0, 두 방식에서 각각32,768단계, 합계65,536단계를 실제 학습했다. 가중치 업데이트·저장 복원·평가 중 학습 이력 보존을 확인했다. 학습 전 네 상대·양 좌석·두 행동 복제의16개 짝지은 경기에서 초기 원시 비행 기록과 승패가 두 방식에 일치했다. 새로운 전체 모델에 대해 NumPy 전용 실행의 경기/궤적 동등성을 검증했다고 주장하지는 않는다.

본 실행은 `league_history_matched_20261007`이다. 현재 관측(control)과 시간 관측(history)을 각각 학습 RNG5900/5901에서1,048,576단계씩, 합계4,194,304단계를 추가한다. 두 번째 RNG에서는 순서를 뒤집는다. 작업자8개·환경32개·PPO rollout4,096개다. 같은 pretrained 시작 정책과 대응되는 optimizer를 사용하고 새로운 RNG/초기조건274M·275M로 시작한다. 같은 RNG의 두 방식은 초기조건 대역을 공유한다. 별도 초기화부터의 독립 학습이나 원래 JSBSim 궤적의 복원은 아니다.

84M 조건2개·양 좌석·32상대·행동 복제2개의 소규모 개발 평가를 사용한다. 두 반복이 끝난 뒤 현재 관측 대비 시간 관측의 상대별 성적과 개선 일관성을 비교한다.83M 결과는 이번 전략 선택에 사용했으므로 이후 미사용 검증이라고 부르지 않는다. 최종 시험과 미사용 상대는 열지 않았고 승격하지 않았다. GitHub는 명시적 요청 때만 올린다.

```powershell
# aircombat-rl에서 실행. 현재 살아 있는 실행을 중복 시작하지 말 것.
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& '.venv-cpu\Scripts\python.exe' -X utf8 -u -m tools.league_history_train --assessment runs\league_control_transfer_assessment_20261007 --out runs\league_history_matched_20261007
```

핵심 근거는 transfer assessment의 `analysis.json`/`completion.json`, `history_features_checks_20261007.xml`, `league_history_clone_checks_20261007/completion.json`, history smoke의 `completion.json`과 본 실행의 동결 `plan.json`이다. 실제 상태는 runtime PID와 프로세스, append-only telemetry와 completion/failure로 확인한다. 원자 교체되는 progress.json을 반복해서 열지 않는다. 이미 사용된 출력 폴더나 예약 초기조건을 우회·덮어쓰지 않는다.

신규 controller는 기존 grouped trainer에 환경·checkpoint·평가 adapter를 명시적으로 연결한다. 기존에 동결한 소스 파일을 수정하지 않는다. 이전 정책·실패 결과·원시 기록은 모두 보존한다. `accounting_note.json`은 조상 공통 경로와 이번 시작 정책의 pilot 단계를 구분해 중복 합산을 방지한다.
