# 다른 컴퓨터에서 이어가기 — CPU 인계, 2026-10-16

현재 학습 Goal은 **paused**다. 이번 요청은 GitHub 게시와 이식 준비이며 자동 학습 재개 요청으로 해석하지 않는다. 다음 작업자는 이 문서 → [전체 학습·평가 경과](SEVEN_OPPONENT_PROGRESS_20261016.md) → [현재 상태](LEAGUE_STATE.json)의 `publication_handoff_20261016` 순으로 읽는다. 옛 문서의 `running`, 절대 경로, PID는 과거 기록이다.

## GPU 필요 여부

**최근 직접 조종, 관측 기반 controller 개발, JSBSim 경기 평가는 모두 CPU로 했다. GPU가 필요 없다.** 기존 PPO 학습 코드도 `device='cpu'`로 학습한다. NumPy 추론 모델 역시 CPU로 실행한다. 기존 PC의 CUDA PyTorch 설치 여부와 GPU 사용은 다른 문제다. 이번 인계는 CUDA가 없는 별도 PyTorch CPU 가상환경으로 검증한다.

Linux x86_64/Python 3.12를 기준으로 설치·검증했다. Windows에서는 WSL2 Ubuntu를 사용한다. 새 runner와 기존 controller는 `fcntl` 잠금을 사용하므로 native Windows 실행은 검증하지 않았다. 다른 CPU·라이브러리에서 모든 수치가 bitwise 동일하다는 보장은 없다. GPU 시뮬레이터로 바꿔 검증하는 경로가 아니다.

## 받기와 설치

아직 PR을 merge하지 않았으므로 **현재 게시 브랜치**를 받는다. 기존 작업 폴더가 있다면 새 경로에 clone하여 미게시 작업을 보존한다.

```bash
git clone --branch feat/seven-model-league-20261015 https://github.com/qkrckstj/RL_KAU.git RL_KAU-cpu
cd RL_KAU-cpu
git rev-parse HEAD
cd aircombat-rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-portable.txt psutil==7.2.2
python -m pip install --no-deps -e .
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m tools.cpu_handoff verify
```

검증 환경 출력은 `2.14.1+cpu None False`다. 가상환경 자체는 옮기지 않는다. `tools`/`experiments`는 저장소 내부 코드이므로 위처럼 `aircombat-rl`에서 실행한다. 지정 wheel이 목적지 플랫폼에서 설치되지 않으면 실제 오류와 환경을 기록하고 해결한 뒤 검증한다. 설치 실패를 재현 성공으로 처리하지 않는다.

## 무엇을 가져왔는가

|내용|위치|
|---|---|
|원본 7개, 원래 learner 및 greedy/행동난수 4900·4901 exports|[기존 manifest](../artifacts/20261015/manifest.json)|
|현재 보존 controller: feedback002, 203승4무17패/strict1상대 통과|[candidate](../artifacts/20261016/direct/feedback_002/candidate/)|
|feedback000–003 전체 후보·224경기별 결과·원본 plan/completion/audit|[직접 제어 증거](../artifacts/20261016/direct/)|
|신경망 최선 strict 대표 s61500(3/7) 및 마지막 완료 s75501(0/7)|[선택 모델](../artifacts/20261016/models/)|
|92개 완료 후보별 설정·학습량·시간·상대별 결과|[전체 경과 MD](SEVEN_OPPONENT_PROGRESS_20261016.md), [원본 result/settings](../artifacts/20261016/evidence/production/)|
|학습 분기·직접 조종·회피 실험의 기존 MD 전체|[reports](../artifacts/20261016/reports/)|
|실제 수동 경기 영상|[전체 영상](../artifacts/20261016/video/manual003_full_match.mp4), [느린 측면 공격](../artifacts/20261016/video/manual003_side_attack_slow.mp4)|
|수동 명령/관측 테이프와 필요한 회귀 테스트 fixture|`aircombat-rl/runs/direct_pilot_20261016/`|
|CPU controller/PPO/평가 구현, 새 이식 진입점|`aircombat-rl/experiments/aim_async`, `experiments/direct_pilot`, `tools/cpu_handoff.py`|
|논문 조사와 mutual-zero 진단|[논문 조사](LITERATURE_2D_RL_MUTUAL_ZERO_20261016.md), [진단](../artifacts/20261016/reports/seven_mutual_zero_diagnosis_20261016/SUMMARY.md)|

새 파일의 SHA-256은 [신규 manifest](../artifacts/20261016/manifest.json), 선택 이유·제외 범위는 [selection](../artifacts/20261016/selection.json)에 있다. 원본 7개 및 [122개 설정 변경 목록](HISTORICAL_WEIGHT_AND_SETTING_CHANGES_20261015.md)은 이전 게시 바이트를 유지한다.

중간 신경망 체크포인트 전체, 과거 넓은 탐색의 대량 raw cells, 무관한 GPU 미게시 소스, 가상환경·캐시는 제외했다. 해당 로컬 원본은 삭제하지 않았다. 선택한 모델과 진행 내역을 이용한 **새 실험 재개**를 지원한다. 과거 모든 프로세스/RNG/파일 경로를 그대로 복구하는 exact resume나 모든 중간 모델의 제공을 약속하지 않는다.

## 확인과 동일 조건 재평가

먼저 코드/증거 검사만 실행한다. 다음 테스트는 추가 native 경기를 실행하지 않는다.

```bash
python -m pytest -q tests/test_cpu_handoff.py tests/test_agent_direct*.py tests/test_agent_manual_replay.py tests/test_manual_feedback*.py tests/test_range_evaluate.py tests/test_early_range_evaluate.py
```

재개 요청을 받은 작업자는 실제 프로세스·완료/실패 파일·RAM 여유를 먼저 확인한다. 다음 명령들은 **실행 시에만** 경기를 시작하며 문서를 읽거나 `verify`를 실행하는 것으로 시작되지 않는다.

```bash
# 설치 확인용: 원본 7개 × 양 좌석 × 기존 초기조건 1개 × 행동난수 4900 = 14경기
# 별도의 재현 smoke이며 strict224를 대체하거나 통과를 주장하지 않는다.
python -m tools.cpu_handoff smoke --workers 1 --output runs/cpu_smoke_새로운이름

# 보존002 후보의 기존 consumed screen 전체: 정확히224경기, 학습0
python -m tools.cpu_handoff screen --workers 3 --output runs/cpu_screen_새로운이름
```

출력 경로는 새 폴더여야 한다. 기존 폴더·실행 중 잠금·파일 해시 변경은 거부한다. 실패한 실행을 덮어쓰거나 자동 재시도하지 않는다. `failure.json`과 실제 프로세스 종료를 확인하고 원인을 해결한 새 실행으로 기록한다. 실행 중에는 같은 폴더를 다른 프로세스에서 재개하지 않는다.

`--workers`는 1–3이다. 기본은 1이며 3은 최근 고정 screen의 실행량이다. 현재 runner는 실행 전 사용 가능 RAM 4/6/8 GiB 이상을 각각 요구한다. 메모리가 부족하면 worker 수를 줄이고 기록한다. 스레드 수는 위 환경 변수로 Python 시작 전에 1로 제한한다. 학습용 8process/32env 설정은 별개이며 다른 컴퓨터에 그대로 권하지 않는다.

새 출력에는 `plan.json`(실제 경로·모델·풀·조건·동결 소스 해시), `runtime.json`, 경기별 `cells`, 진행 기록, `completion.json`(경기 수·소요 시간·피해/위치 지표·strict 상대표)이 생긴다. 학습 결정 0, pool admission 없음, version 없음으로 기록한다. 224 screen이 모두 통과해도 이 CLI는 confirmation/admission을 자동 실행하지 않는다.

기본 controller의 역사적 224 결과는 **203승4무17패**, strict **1/7(47238만 통과)**다. 61500의 3/7은 신경망 모델의 다른 결과다. 둘을 혼동하지 않는다. 동일 조건 재실행은 기존 screen의 재현 확인이며 새 미사용 증거가 아니다.

## 계속할 작업 — Goal 재개를 명시적으로 요청받은 뒤

1. `feedback002`를 원본으로 보존하고 새 후보 폴더로 복사한다. [다음 가설 설명](../artifacts/20261016/reports/direct_pilot_20261016/feedback_003/NEXT_MERGE_BRIEF.md)을 먼저 읽는다. `merge_range`만 240→600으로 바꾸는 미시험 가설이다. `brake_speed`는 **482**다. 기각된 feedback003의 520을 상속하지 않는다.
2. 새 `policy_net.zip`의 `parameters.json`과 `policy_net.json`을 함께 변경하고 부모002/변경 항목/새 후보 ID를 명시한다. `policy.py`는002와 byte-identical이어야 한다. 기존 export의 `entrant.json`은 과거 경로 provenance이며 새 runner가 실제 위치에서 entrant를 구성한다.
3. 실제 새 패키지를 공식 loader로 읽어 변경 파라미터가 적용됐는지, 소비된 테이프에서 행동/phase·반복 호출/reset이 일치하는지 검사한다. 기존 기본 파라미터 Policy와만 비교해서 새 파일 검증으로 오인하지 않는다. 준비/독립 검토 후 하나의 고정 후보만 평가한다.
4. `python -m tools.cpu_handoff screen --candidate artifacts/새후보폴더 --output runs/새실험 --workers 3`로 동일224를 실행한다. 후보 폴더 전체의 비-cache 파일과 실행 소스를 동결한다. 보조 파일을 저장소 밖의 변경 가능한 경로에서 읽는 모델은 먼저 self-contained export로 만든다. 학습량0/평가224/시간/상대별WDL·strict/부모 대비 회귀를 기록한다.
5. 전승 실패 시 원본002를 보존하고 원인을 진단한다. 성공 시에만 전체 seed 사용·reservation 기록을 대조하여 **새 미사용**16조건 confirmation을 별도 준비한다. 448 결과까지 모두 통과해야 v1·풀 추가가 가능하다. 임의의 옛 confirmation band를 unseen으로 재사용하지 않는다.

후방·측면 모두 괜찮으며 일방 공격을 만들고 유지하는 것이 목적이다. 단순 도주 무승부는 실패다. 전승 모델을 보존한 후에만 초반 피격 감소 개선을 시도한다. 이후 실제 개선 모델을 v2,v3…로 기록하고 부모·가중치·정책·설정·평가·해시를 보존한다. 현재는 v1이 없고 neural imitation/curriculum을 시작할 단계가 아니다.

## 이전 PPO를 다시 연구할 경우

선택한 신경망 폴더에는 `learner.zip`(actor/critic/Adam), `READY.json`, worker/RNG 스냅샷과 greedy/sampled inference export가 있다. source checkpoint와 설정은 [selection](../artifacts/20261016/selection.json) 및 각 production settings/result에서 찾는다. CPU 모델을 읽는 예:

```python
from pathlib import Path
from stable_baselines3 import PPO
path = Path.cwd().parent / 'artifacts/20261016/models/neural_latest_s75501/learner.zip'
model = PPO.load(path, device='cpu')
print(model.device, model.observation_space, model.action_space)
```

이 명령은 모델 읽기이며 학습 재개가 아니다. 분기별 custom policy/참조 archive/prior semantics와 새 worker 환경·학습 분포·gamma/GAE·optimizer override·seed reservation을 다시 검증한 새 config가 있어야 실제 continuation이 가능하다. READY의 옛 절대 경로와 config hash를 치환하거나 frozen check를 건너뛰어 실행하지 않는다. 과거 `seven_model_league_v* prepare`는 원본 로컬 archive 경로에 의존하므로 **그대로 새 컴퓨터에서 실행하는 재개 명령이 아니다**. 최근 직접 제어 작업의 다음 단계는 위 controller 가설이며 PPO 자동 재개는 이번 인계 범위가 아니다.

## 게시 및 검증 기록

[게시 검증 기록](verification/cpu_handoff_20261016.json)에서 새 CPU 환경, 테스트·모델 로딩·재현 smoke와 실제 exit를 확인한다. 게시 검증 경기는 기존 production/직접 제어 누계와 별도 집계한다. [PR #4](https://github.com/qkrckstj/RL_KAU/pull/4)는 소유자가 merge한다. 이후 GitHub 업로드는 사용자가 다시 요청할 때만 한다.
