# 현재 자동 학습에 필요한 논문

사용자가 로컬에 제공한 PDF를 2026-10-12 명시적 업로드 요청에 따라 보관한다. 원본 파일명과 내용을 유지했다. 각 문서의 저작권과 이용 조건은 원 저자·출판자에게 있다. 저장소 코드의 라이선스가 논문에 새로 적용되는 것은 아니다.

| PDF | 공식·저자 출처 | 현재 역할 |
|---|---|---|
| [AlphaDogfight](2105.00990v3.pdf) | [arXiv v3](https://arxiv.org/abs/2105.00990v3) | 상대 위치 보상과 후방 확보/사격 정책의 구분 |
| [Coordinated Strategies](2510.11474v2.pdf) | [arXiv v2](https://arxiv.org/abs/2510.11474v2) | 위치 확보·공격·방어 및 단계적 훈련 |
| [PPO](1707.06347v2.pdf) | [arXiv v2](https://arxiv.org/abs/1707.06347v2) | 현재 사용하는 강화학습 알고리즘 |
| [Emergent Complexity](1710.03748v3.pdf) | [arXiv v3](https://arxiv.org/abs/1710.03748v3) | 상대 경쟁 학습 배경 |
| [Policy Invariance under Reward Transformations](shaping-icml99.pdf) | [저자 PDF](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf) | 잠재함수 보상과 직접 보상의 차이 |

핵심은 [전략 문서의 적용 표](../AUTOMATED_TACTICS_STRATEGY_20261012.md)다. 논문 수록은 해당 논문의 전체 구조를 구현했거나 그 성능을 재현했다는 의미가 아니다. 특히 새 직접 위치 보상은 개발 비교 중이며 성능 개선은 미확정이다.

원본 파일 크기와 SHA-256: [manifest.json](manifest.json). 추가 평가·계층형 공중전·비행 교범 참고자료는 [전체 목록](../../REFERENCES.md)에 있다.

## 추가 평가 논문과 시간 추상화 근거

- [통계적 평가 PDF](2108.13264v4.pdf): 사용자 제공 원본을 추가 보관했다. [arXiv](https://arxiv.org/abs/2108.13264v4). 적은 반복과 개발 패널 재사용의 한계를 해석하는 참고자료다.
- [DAgger 공식 원문](https://proceedings.mlr.press/v15/ross11a.html): 후보 방문 상태와 전문가 시범의 분포 차이를 다루는 근거다. 공식 PDF는 해당 페이지에서 제공한다.
- [Options 저자 페이지](https://web.eecs.umich.edu/~baveja/Options.html): 1초 동안 전문가 선택을 유지하는 설계의 개념적 근거다. 저자 페이지에 논문 링크가 있다.

후자의 두 자료는 공식/저자 링크로 제공한다. 현재 핵심 로컬 PDF는 총 6편이다. 최신 선택기와 보상 비교는 종합 개선으로 채택되지 않았으며 다음 단계는 더 낮은 피격의 후방 교사 확보다.
