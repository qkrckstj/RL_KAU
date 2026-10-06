# 새 작업 세션은 여기서 시작

이 문서와 저장소의 파일을 기준으로, 이전 대화 없이 프로젝트를 파악하고 이어갈 수 있도록 정리했다.
코드 실행과 저장소 접근이 가능한 환경이 필요하다. 실행 중인 프로세스나 Python 환경 자체가 GitHub로 이동하는 것은 아니다.

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
