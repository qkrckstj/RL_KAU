# 후방 전술 개발 모델 묶음

사용자가 선택한 후방 공격 방향의 개발 후보다. 종합 승률 우승 모델로 교체한 것이 아니다. retained_ppo는 영상 첫 경기의 모델, rear_candidate는 두 번째 경기의 geometry_s10101 모델이다. archived_opponent는 첫 경기의 상대다.

원래 정책 실행 코드와 가중치는 바이트 그대로 복사했다. policy_net.json의 source는 출처 기록이며 실행 시 외부 경로가 필요하지 않다. learner.zip은 두 후보의 PPO actor/critic/optimizer 스냅샷이다. rng.pt가 포함된 모델은 난수 기록도 보관한다. 이것만으로 과거48상대 실험 전체나 중단한 JSBSim 상태를 완전히 복원한다고 주장하지 않는다. 다음 학습은 새 환경을 시작하고 별도 계획·상대 명세를 고정해야 한다.

저장소 설치 후 aircombat-rl에서:

```powershell
python -m tools.verify_rear_branch_bundle
```

두 공개 경기의 승패·종료 체력·경기 시간·실제 후방 피해와 파일 SHA256을 검증한다. 학습은 하지 않는다. 모델 및 상대 경로는 manifest.json에 저장소 상대 경로로 기록되어 있다. 설치 환경은 REPRODUCE.md 참고. 다른 컴퓨터의 실행 속도는 달라질 수 있다.

[최신 계획](../../../../docs/REAR_BRANCH_PLAN_20261012.md), [영상](../../../../docs/media/rear_wins_20261012/rear_wins.mp4).
