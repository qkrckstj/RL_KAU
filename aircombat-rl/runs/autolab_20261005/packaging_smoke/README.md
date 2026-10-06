# FairFight CEM 정책

공개39관측으로 선회·목표 속도·추적 전환을 결정하는7개 파라미터 정책이다.
경기 결과로 값을 학습했다. DQN/PPO 신경망 정책과 구분한다.
원본 JSBSim 물리, Ace 상대, 무장,9행동,20Hz 주기와 공식 승패 판정을 사용한다.
선택 기준과 출처는 `selection.json`, 파라미터는 `policy_net.json`에 있다.
`policy_net.zip`은 JSON 파라미터를 담은 압축 파일로 공식 정책 로더가 바로 발견한다.

`aircombat-rl` 폴더에서 기존 Python 환경으로 실행:

```powershell
python -m tools.watch templates/project_04_fair --design experiments/plan_a/cem_policy --seed 904000
```

개발 조건 점검(최종 시험을 반복 튜닝에 사용하지 않음):

```powershell
python -m tools.grade templates/project_04_fair --design experiments/plan_a/cem_policy --band 906000 --n 40 --out experiments/plan_a/cem_policy/development_check.json
```

최종 검증과 한계는 상위 폴더의 `autolab.md`와 완료 보고서를 참고한다.
미세조정 반복들은 같은 학습된 부모를 공유하며, 독립 최초 학습6회로 세지 않는다.
