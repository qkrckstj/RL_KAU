# RL_KAU — JSBSim 강화학습 실험

`inmo-jang/aircombat-rl` 기반 FairFight 프로젝트의 코드, 훈련 계획, 평가 기록,
학습된 모델과 재생 버퍼를 함께 보관한다. 정리일: 2026-10-06.

## 첫 리그 최종 시험: 체력 기준 재판정 — 2026-10-09

**16종 상대의 과거 최종 시험을 종료 체력 기준으로 재판정하면 기존 CEM 승률35.8% → 사전 선택 정책 `refine_2200` 85.8%**다.
같은 40초기조건×양 좌석×16상대=정책당1,280경기이며 선택 정책은1,098승126무56패, 기존은458승305무517패다.
기존 정책과 직접 대결은 새 정책80전 전승이다. 실제 다른 참가자 상대 성능은 아직 미확인이다.
양쪽 종료 체력을 소수점 네 자리로 반올림하고 높으면 승리·같으면 무승부로 처리했다(`user_terminal_health_round4_v2`).
이전26.6%→77.6%는 구판정 수치다. 이번 변경은 같은 경기의 재집계이며 추가 학습 성과나 새로운 미사용 시험이 아니다.
[재판정 근거·상대별 집계·재현 방법](docs/first_league_health_rejudge_20261009/README.md)을 참고한다.

후보 학습→기존 정책들과 대결을6회 실행하고 우수 모델을3번 교체했다.
부모를 공유한 추가 학습3회 후 개발 기준으로 `refine_2200`을 미리 선택했으며 최종 시험으로 재선택하지 않았다.
당시 사전 평가 관문은 구판정으로 통과했고 이동용 모델 실행 검사도 완료했다. 구판정 관문·신뢰구간을 새 기준 검증으로 재사용하지 않는다.

- **[새 GPT/새 컴퓨터는 여기서 시작](START_HERE.md)** · [현재 상태와 다음 작업](docs/LEAGUE_STATE.json)
- [최종 결과·상대별 표·그래프](aircombat-rl/experiments/league/result.md)
- [지금까지 한 일·첫 업로드 범위](docs/PROGRESS_20261006.md)
- [리그 자동 학습 절차](aircombat-rl/experiments/league/README.md)
- [최종 정책 파일](aircombat-rl/experiments/league/bundle/models/final/)
- [이동용 정책·실행 자료](aircombat-rl/experiments/league/bundle/README.md)

이번 업데이트에는 이전 신경망 비교·실패 진단·Ace 상대 정책도 함께 담았다.
약1.3GB의 새 과거 신경망 재생 버퍼는 로컬에 유지하며, 정확한 범위는
[보관 목록](docs/verification/20261006_github_snapshot.json)에 구분한다.

## 이전 완료: Ace 상대 CEM 제어 파라미터 학습 — 2026-10-06

별도 시험100초기조건×양 좌석에서 개발 기준으로 미리 선택한 정책이 **200승/200경기**,
같은 조건의 고정 좌선회·감속은69승이었다. 부모 공유 미세조정3회의 선택 모델 평균99.3%,
마지막 모델 평균99.0%로 이 Ace 상대 단계의 사전 성공 기준을 충족했다. 이후 위의 다중 상대 학습으로 확장했다.
공개 관측/원본 행동을 쓰는7개 제어 파라미터 탐색이며 신경망 DQN/PPO와 구분한다.
적용 범위는 현재FairFight와 Ace 상대다.

- [완료 보고서·그래프·한계](aircombat-rl/experiments/plan_a/autolab_result.md)
- [저장한 정책과 실행 방법](aircombat-rl/experiments/plan_a/cem_policy/README.md)
- [자동 실험 기록](aircombat-rl/experiments/plan_a/autolab.md)

아래는 기존 이전/실험 기록이다. 최신 GitHub 보관 범위와 실행 상태는 위 업데이트를 기준으로 읽는다.

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
