# 리그 정책과 재실행 자료

현재 묶음은 **개발 단계 자료**다. `manifest.json`의 `stage`를 확인한다.
최종 평가 전에는 이 묶음을 대회 성능이 확정된 결과로 해석하지 않는다.

- `models/cem_original`: 기존 Ace 상대 CEM 정책, 원본 코드와 계수를 복사해 보존했다.
- `models/round_00`부터 `round_05`: 6라운드에서 선택한 모든 후보. 채택되지 않은 후보도 포함한다.
- `models/ddqn_s1`, `models/ddqn_s0`: 대결 재실행에 필요한 과거 신경망 정책과 관측 변환 코드.
  seed 0은 학습에서 제외한 상대다. 파일을 보관하는 것과 학습에 사용하는 것은 구분한다.
- `source_snapshot`: 본 학습에 사용한 Python 소스. JSBSim 자체와 그 배포 데이터는 설치된 패키지를 사용한다.
- `replay_tools`: 이 묶음을 만든 시점의 재평가·복원 도구.
- `evidence`: 개발 평가, 설정, 소스 해시, 실제 패키지 버전 기록.

정책은 각 모델 폴더의 `policy.py`, `wrappers.py`, `policy_net.zip`으로 공식 로더에서 실행할 수 있다.
CEM 후보의 ZIP은 17개 제어 계수이며 신경망 가중치가 아니다. DDQN 폴더의 ZIP은 신경망 모델이다.

프로젝트 루트 `aircombat-rl`에서 기존 Python 환경을 사용한다:

```powershell
# 공식 Ace 상대 시청
python -m tools.watch templates/project_04_fair --design experiments/league/bundle/models/round_05 --seed 32000500

# 같은 상대 목록으로 새로운 개발 대결 실행. 출력 폴더는 아직 없는 경로를 지정한다.
python -m tools.league_replay --bundle experiments/league/bundle --model round_05 --out runs/my_league_replay --band 33010000 --n 40

# 다른 컴퓨터에서 과거 runs/ 입력이 없을 때만 복원한다. 다른 내용의 기존 파일은 덮어쓰지 않는다.
python -m tools.league_replay --bundle experiments/league/bundle --restore-inputs

# 다른 사람이 제공한 정책 폴더와 직접 대결
python -m tools.league_duel --a experiments/league/bundle/models/round_05 --b path/to/other_policy --out runs/duel_with_other --band 33020000 --n 40
```

다른 컴퓨터에서는 현재 프로젝트의 코드와 이 묶음을 함께 복사한다. 원본 상위 저장소만 새로 클론한 상태라면
현재 추가한 `tools/league_*.py`, `tools/autolab_cem.py`, `experiments/plan_a/tactical.py`, `experiments/league/controller.py`도 필요하다.
소스 사본은 이 묶음의 `source_snapshot`과 `replay_tools`에 있다. 기존 작업을 덮어쓰지 않도록 별도 체크아웃에서 복원한다.
Python 3.12를 사용했으며 정확한 실행 버전은 `evidence/runtime.json`에 있다. 모든 대결은 CPU로 실행했다.

`--transfer`는 학습에서 제외한 로컬 상대도 평가한다. 이를 실행해 결과를 보고 다음 모델을 바꾸면 그 평가는 더 이상 손대지 않은 최종 시험이 아니다.
기존 시험 band 40000000은 자동 최종 검증용으로 예약돼 있으므로 수동 개발 평가에 쓰지 않는다.
