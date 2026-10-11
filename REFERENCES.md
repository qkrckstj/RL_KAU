# 검토한 논문·교범

훈련 계획에 사용한 원문 위치다. 2026-10-12 사용자 요청으로 현재 전략에 필요한
[핵심 PDF 6편](docs/papers/README.md)을 추가했다. 나머지는 공식·저자 원문 링크를 유지한다.
현재 적용 판단은 [자동 학습 전략](docs/AUTOMATED_TACTICS_STRATEGY_20261012.md), 과거 계획은
[훈련 계획](JSBSim_비행기_AI_훈련_계획.md)에 있다.

| 자료 | 당시 파일 | 원문 |
|---|---|---|
| PPO | `1707.06347v2.pdf` | [arXiv v2](https://arxiv.org/abs/1707.06347v2) |
| Bansal 등의 경쟁 학습 | `1710.03748v3.pdf` | [arXiv v3](https://arxiv.org/abs/1710.03748v3) |
| Pope 등의 AlphaDogfight | `2105.00990v3.pdf` | [arXiv v3](https://arxiv.org/abs/2105.00990v3) |
| Agarwal 등의 통계적 평가 | `2108.13264v4.pdf` | [arXiv v4](https://arxiv.org/abs/2108.13264v4) |
| Chai 등의 계층적 6DOF 공중전 | `2212.03830v1.pdf` | [arXiv v1](https://arxiv.org/abs/2212.03830v1) |
| 2D HHMARL | `2309.11247v1.pdf` | [arXiv v1](https://arxiv.org/abs/2309.11247v1) |
| 3D HHMARL | `2510.11474v2.pdf` | [arXiv v2](https://arxiv.org/abs/2510.11474v2) |
| SAC | `haarnoja18b.pdf` | [PMLR](https://proceedings.mlr.press/v80/haarnoja18b.html) |
| Potential-based reward shaping | `shaping-icml99.pdf` | [저자 PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) |
| FAA Airplane Flying Handbook, Chapter 4 | `05_afh_ch4.pdf` | [FAA 공식 교범](https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/airplane_handbook) |

계획서의 페이지 번호는 당시 검토한 PDF를 기준으로 한다.
`runs/**/source_snapshot/`의 문서는 당시 원본 기록이므로 이전 로컬 PDF 경로가 남아 있을 수 있다.

## 시범 상태 분포 보완의 추가 근거

Ross, Gordon, Bagnell (2011), **A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning**. [공식 논문 페이지](https://proceedings.mlr.press/v15/ross11a.html), [공식 PDF](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

후보가 직접 방문한 상태에서 전문가 행동 라벨을 수집하는 설계의 근거다. 프로젝트에서는 CEM을 전문가로 사용한 한 차례 데이터 추가 실험이며 논문 전체 알고리즘이나 이론 보장의 재현은 아니다. 특정 상대 개선이 다른 상대 손실을 동반하여 현재 기본 정책으로 채택하지 않았다.

## 전문가 선택과 평가의 추가 근거

Sutton, Precup, Singh (1999), **Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning**. [출판사 원문](https://www.sciencedirect.com/science/article/pii/S0004370299000521), [저자 연구 페이지](https://web.eecs.umich.edu/~baveja/Options.html). 일정 시간 전문가 선택을 유지하는 실험의 개념적 근거다. 고정 1초 선택기의 성공이나 논문 알고리즘 전체 재현을 의미하지 않는다.

Agarwal 등 (2021), **Deep Reinforcement Learning at the Edge of the Statistical Precipice**. [보관한 PDF](docs/papers/2108.13264v4.pdf), [공식 원문](https://arxiv.org/abs/2108.13264v4). 적은 반복의 결과를 과대해석하지 않기 위한 평가 참고다. 현재 두 반복을 통계적 우위의 증명으로 부르지 않는다.

## 최신 적용 상태

정렬 잠재 보상 실험은 할인 누적 등식을 검증했지만 두 반복의 승률 개선이 일치하지 않아 채택하지 않았다. Reward Shaping 논문의 이론적 성질과 실제 학습 성능을 구분한다. 상대 난이도 단계화는 다음 비교 후보이며, 경쟁 학습·단계적 훈련 문헌을 배경으로 삼되 아직 효과가 확인된 방법으로 소개하지 않는다. 최신 수치와 판단은 [현재 전략](docs/AUTOMATED_TACTICS_STRATEGY_20261012.md)에 모았다.
