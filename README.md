# RL_KAU — JSBSim 강화학습 실험

`inmo-jang/aircombat-rl` 기반 FairFight 프로젝트의 코드, 훈련 계획, 평가 기록,
학습된 모델과 재생 버퍼를 함께 보관한다. 정리일: 2026-10-01.

**다른 컴퓨터에서는 [환경 설치·학습 재개 안내](REPRODUCE.md)부터 읽는다.**

```powershell
git clone https://github.com/qkrckstj/RL_KAU.git
cd RL_KAU\aircombat-rl
```

Python 환경은 새 컴퓨터에서 만든다. 이전 PC의 모델·버퍼와 새 A3 모델은 일반 Git 파일로 포함했다.
Git LFS나 submodule 초기화는 필요하지 않다. 새 A3의 대용량 재생 버퍼 18개는 로컬 보관이며,
A3 학습 상태를 그대로 이어가려면 해당 버퍼를 별도로 복사해야 한다.

## 새 PC 후속 결과 — 2026-10-01

**A3 614만 스텝 보상 비교와 후속 진단을 완료했다.** 같은 PC에서 새로 학습한
종료 보상 대조군의 평균 시험 승률은 19.17%, shaping은 29.17%였다.
seed별 변화는 -12.5 / +42.5 / 0%p였고 마지막 모델은 두 조건 모두 0승이다.
일관된 개선 확인 기준을 충족하지 않아 A4는 실행하지 않았다.
다음 우선순위는 과도하게 커진 Q값과 가치 학습의 안정성이다.

실제 실행은 **CPU**였다. CUDA용 PyTorch는 별도 설치했지만 Windows 정책에서
DLL 로딩이 차단돼 GPU 학습·속도 비교는 수행하지 못했다. 사용자 결정에 따라
CUDA 해결은 보류하고 CPU 환경을 유지한다. Windows 보안 설정은 변경하지 않았다.

[A3 상세 결과·그래프](aircombat-rl/experiments/plan_a/shaping_results.md) ·
[최신 인수인계](HANDOFF.md) · [새 PC 검증](docs/verification/new_pc_a3.json)

아래 학습량 비교는 이전 PC에서 완료한 이관 기록이다.
새 A3 결과·그래프·로그·모든 검증 가중치·대표 정책과 실행 소스를 이 저장소에 포함했다.
업로드한 318개 산출물과 로컬에 보관한 재생 버퍼 18개의 해시는
[GitHub 보관 목록](docs/verification/a3_github_manifest.json)에 구분해 기록했다.

## 이전 PC의 학습량 비교 결론

원본 재현, 개선 관측 DQN/PPO 비교, DQN 100만 스텝 추가 실험까지 완료했다.
추가 학습에서는 동일한 새 시험 40경기에서 다음 결과를 얻었다.

| 학습 예산 | 검증으로 고른 모델의 시드별 승리 | 평균 승률 |
|---|---|---:|
| 204,800 | 0/40 · 0/40 · 10/40 | 8.33% |
| 512,000 | 0/40 · 0/40 · 10/40 | 8.33% |
| 1,024,000 | 10/40 · 6/40 · 10/40 | 21.67% |

**100만 스텝의 마지막 모델들은 모두 0/40이었다.** 중간 모델의 검증 선택이 필요하다.
대표 모델은 여전히 seed 2의 204,800스텝 모델이며, 추가 학습의 이득은
다른 시드에서도 좋은 중간 모델을 얻은 데 있다. 반복 3회로 안정적인 수렴을 확보한 것은 아니다.
이후 새 PC에서 A3 보상 비교를 완료했다. 최신 결과는 위 링크를 따른다.

## 문서와 저장 위치

- [전체 훈련 계획](JSBSim_비행기_AI_훈련_계획.md)
- [계획 A: 원본 재현·관측·DQN/PPO 비교](aircombat-rl/experiments/plan_a/README.md)
- [100만 스텝 추가 실험과 그래프](aircombat-rl/experiments/plan_a/duration.md)
- [다른 컴퓨터에서 설치·재개](REPRODUCE.md)
- [다음 작업 인수인계](HANDOFF.md)
- [논문과 교범 원문 링크](REFERENCES.md)
- [원본 저장소와 복사 범위](PROVENANCE.json)

```text
aircombat-rl/
  aircombat_gym/       JSBSim 환경
  templates/          원본 과제와 장치 선택을 추가한 학습 명령
  experiments/        관측·학습 설정 및 실험 기록
  tools/              실험 실행, 평가, 장치 벤치마크, DQN 재개
  tests/              환경·관측·탐험 일정·재개 검증
  runs/
    device_benchmark/ CPU/XPU 속도 비교 원본
    xpu_smoke/        Intel XPU 작동 검증
    plan_a_20261001/
      a0_original/    원본 DQN 실행
      comparison/     DQN/PPO 3조건 × 3시드 비교
      duration/       100만 스텝 × 3시드, 가중치·버퍼·시험 결과
```

`runs/`는 앞으로 생길 실행을 기본적으로 Git에서 제외한다. 이번 이관 시점의
실험 파일은 명시적으로 추적하므로 clone하면 함께 내려온다.
새 실험은 반드시 새로운 출력 폴더를 사용하고, 보관할 결과를 골라 커밋한다.

기반 코드: [inmo-jang/aircombat-rl](https://github.com/inmo-jang/aircombat-rl),
커밋 `4141a2996507e7f0b2e7a8af26815832b5e13d52`.
교수자용 비공개 `solutions` submodule과 개인 Python 환경은 복사하지 않았다.

## 이관 검증

이관한 코드에서 전체 테스트 **62개**가 통과했다.
GitHub에 올린 뒤 별도 폴더에 다시 clone하여 실험 파일 **361개**의 해시를 확인했고,
대표 체크포인트에서 **128스텝 재개**도 성공했다. 기존 50,000개 버퍼를 유지하면서
학습 갱신이 50,950회에서 50,982회로 늘었다.
[검증 기록](docs/verification/clone_smoke.json)을 함께 보관한다.
이 검증은 기존 Windows Python 환경에서 수행했으며 새 컴퓨터의 CUDA 하드웨어 검증은 별도다.
