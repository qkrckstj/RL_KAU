# 다른 컴퓨터에서 설치하고 이어 작업하기

## 1. 코드와 실험 파일 받기

```powershell
git clone https://github.com/qkrckstj/RL_KAU.git
cd RL_KAU\aircombat-rl
```

이후 명령은 `aircombat-rl` 폴더에서 실행한다. Python **3.12**를 사용한다.
이 저장소의 실험 파일들은 Git LFS 없이 일반 Git에 포함돼 있다.

## 2. 새 Python 환경 만들기

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

아래 Windows 명령의 `.\.venv\Scripts\python.exe`를 Linux에서는
환경 활성화 후 `python`으로 바꾸고, 경로 구분자를 `/`로 바꾼다.

### CPU

```powershell
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-portable.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
```

### NVIDIA CUDA

[PyTorch 공식 설치 선택기](https://pytorch.org/get-started/locally/)에서 목적 컴퓨터의
OS, Pip, Python, GPU/드라이버에 맞는 CUDA 빌드를 선택한다.
표시된 명령의 `pip` 또는 `pip3` 부분을 `.\.venv\Scripts\python.exe -m pip`로 바꿔
**이 가상환경 안에** PyTorch를 설치한다. 이어서:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-portable.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

마지막 결과가 `True`여야 재개 명령에 `--device cuda`를 사용한다.
이 절은 이전 PC 이관 당시의 일반 설치 안내다. 새 PC의 RTX 5080과 드라이버는 확인했지만
Windows DLL 차단으로 실제 CUDA 학습은 검증하지 못했다. 현재는 CUDA 사용을 보류한다.

### Intel XPU — 기존 Windows 환경과 같은 구성

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-windows-xpu.lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.xpu.is_available())"
```

위 잠금 파일은 Windows·Intel 환경용이다. NVIDIA 컴퓨터에는 CPU/CUDA 설치 절차를 사용한다.
기존 환경은 torch 2.14.1+xpu, SB3 2.9.0, JSBSim 1.3.0이었다.
장치·라이브러리·OS가 달라지면 수치 결과가 비트 단위로 같다고 보장할 수 없다.
이 노트북의 작은 DQN 실험에서는 CPU 1스레드가 XPU보다 빨랐다.

## 3. 설치 점검

```powershell
.\.venv\Scripts\python.exe -m tools.verify_transfer
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest tests/test_plan_a.py tests/test_plan_a_long.py tests/test_resume_dqn.py -q
```

공식 평가 경로로 저장된 대표 모델을 확인한다. 아래 band는 이미 사용한 회귀 점검용이다.

```powershell
.\.venv\Scripts\python.exe -m tools.grade templates/project_04_fair --design runs/plan_a_20261001/duration/selected_policy --n 40 --band 1200000 --out runs/transfer_check/score.json
```

그래픽 창으로 보려면:

```powershell
.\.venv\Scripts\python.exe -m tools.watch templates/project_04_fair --design runs/plan_a_20261001/duration/selected_policy
```

## 4. 저장된 DQN을 실제로 이어 학습하기

현재 대표 모델인 seed 2, 204,800스텝 체크포인트에는 재생 버퍼도 있다.
다음 명령은 여기에 51,200스텝을 추가하고, **새 출력 폴더**에 저장한다.

```powershell
.\.venv\Scripts\python.exe -m tools.resume_dqn --run runs/plan_a_20261001/duration/s2 --checkpoint 204800 --additional-steps 51200 --device cpu --out runs/resumed_s2_256000
```

NVIDIA 환경이면 `--device cuda`, 지원 Intel 환경이면 `--device xpu`로 바꾼다.
먼저 아주 짧게 확인하려면 `--additional-steps 128 --val-every 128 --val-n 2`를 사용하고
`--out runs/resume_smoke_new_pc`처럼 별도 폴더를 준다. 2경기 검증은 작동 점검용이다.

100만 스텝 마지막 모델부터 이어가려면 `--checkpoint 1024000`을 지정한다.
다만 마지막 모델의 성적은 좋지 않았으므로 대표 모델과 혼동하지 않는다.
seed 0·1도 `--run`의 `s2`를 `s0`·`s1`로 바꾸면 재개할 수 있다.
기존 버퍼는 각 시드의 **204800 / 512000 / 1024000** 시점에 있다.
819200·972800 등의 중간 가중치는 추론·평가용이며 당시 버퍼가 없어 이 명령으로 완전한 learner 재개는 할 수 없다.

재개 명령의 동작:

- 모델·target network·optimizer·재생 버퍼와 저장한 CPU RNG를 복원한다.
- 탐험 확률을 0.05로 유지하고, 원래 스텝 수에 추가 학습량을 더한다.
- 기존 모델도 검증 후보에 포함해, 이후 성능이 나빠져도 좋은 모델을 보존한다.
- 검증마다 가중치와 버퍼를 함께 저장하므로 새 출력 폴더에서도 다시 재개할 수 있다.
- 자동으로 최종 시험 집합을 열지 않는다. 검증 band 900000만 사용한다.

**비행 중 JSBSim 상태는 저장하지 않았으므로 새 에피소드에서 재개한다.**
중단 없는 학습과 동일한 경로를 재현한다는 뜻은 아니다. 새 경기 seed 기본값은 10002다.
GPU RNG는 새 seed로 초기화하며, 저장한 Python/NumPy/PyTorch CPU RNG는 복원한다.
관측·보상 코드를 바꾼 실험에 기존 버퍼를 그대로 섞는 명령이 아니다.

## 5. 다음 실험을 새로 시작하기

권장 다음 단계는 [인수인계 문서](HANDOFF.md)의 **A3 보상 비교**다.
보상 변경 실험은 기존 learner의 단순 재개와 구분하고, 대조군과 같은 조건에서 새로 학습한다.

기존 실험을 다시 재현하려면:

```powershell
.\.venv\Scripts\python.exe -m tools.plan_a matrix --out runs/plan_a_repeat --workers 3
.\.venv\Scripts\python.exe -m tools.plan_a_long run --out runs/duration_repeat
```

이 두 기존 실험 실행기는 **CPU 1스레드 × 최대 3개 프로세스**로 고정돼 있다.
CUDA로 바뀌는 명령은 위 `tools.resume_dqn --device cuda` 또는 각 과제의 `train.py --device cuda`다.
기존 행렬 실행기를 CUDA로 자동 전환한다고 오해하지 않는다.
`plan_a_long run`은 원래 시험 band 1200000까지 재평가하므로 재현 점검용이다.
새 방법의 성능 판단에는 이미 확인한 시험 집합을 새 집합처럼 사용하지 않는다.

## 보관과 재현 범위

`runs/`의 기존 스냅샷은 저장소에 포함돼 있지만, 새 실행 폴더는 기본적으로 ignored 상태다.
새 작업을 보관할 때는 코드와 결과를 검토한 뒤 필요한 실행 폴더만 `git add -f`로 추가한다.
개인 `.venv`, 패키지 캐시, 인증 파일은 올리지 않는다.
이관 당시에는 실제 체크포인트에서 128스텝을 재개해 카운터 증가, 50,000개 버퍼 유지,
가중치 저장과 검증을 확인했다. 다른 컴퓨터의 장치 검증은 위 설치 점검으로 수행한다.
GitHub에서 다시 clone한 복사본에서도 같은 재개 점검을 통과했다.
[전체 검증 기록](docs/verification/clone_smoke.json)에 당시 커밋과 결과를 남겼다.

## 2026-10-01 새 PC 검증 추가

Ryzen 7 9800X3D / Python 3.12.10의 CPU 환경은
`aircombat-rl/requirements-windows-cpu.lock.txt`에 고정했다.
Windows에서 한글 경로를 사용할 때는 Python 명령에 `-X utf8`을 지정한다.
패키지 메타데이터 기록은 `pip freeze`의 한글 Git 경로 오류를 피하도록 표준 라이브러리로 수집한다.

`resume_dqn`은 실행 폴더의 source_sha256을 검증한 뒤 저장 당시 utils.py를 불러온다.
현재 소스에 A3 보상이 추가돼도 이전 sparse 모델은 이전 보상으로 재개된다.
보상 조건을 바꾸면서 과거 버퍼를 재사용하는 명령은 아니다.

RTX 5080을 인식했지만 별도 CUDA 환경은 Windows 애플리케이션 제어에서 DLL 로딩이 차단됐다.
`.venv-cuda` 설치 완료만으로 CUDA 사용이 검증된 것은 아니다.
자세한 결과는 [새 PC 검증 기록](docs/verification/new_pc_a3.json)에 있다.


## A3 결과의 GitHub 보관 범위

[보관 목록](docs/verification/a3_github_manifest.json)의 `files`에는 업로드한 A3 산출물
318개의 상대 경로·크기·SHA-256이 있다. 평가 JSON, 학습/검증 로그, 그래프,
고정 소스, 120개 검증 체크포인트, 대표·최종 가중치와 RNG 기록을 포함한다.
`local_only_files`의 재생 버퍼 18개는 약 230 MiB이며 원래 PC에 보존한다.
[전체 로컬 목록](docs/verification/a3_artifact_manifest.json)은 두 범위를 합친 336개 파일의 기록이다.

clone만으로 A3 모델 평가와 결과 열람이 가능하다. A3 체크포인트에서
`tools.resume_dqn`으로 학습 상태를 이어가거나 `tools.diagnose_shaping`으로
버퍼 진단을 다시 수행하려면 필요한 `replay_buffer.pkl`을 원래 상대 경로에 복사하고
목록의 해시와 일치하는지 확인해야 한다. 기존 duration 실험의 버퍼는 계속 저장소에 포함한다.

CUDA DLL 차단 해결과 GPU 사용은 사용자 결정으로 보류했다.
Smart App Control을 포함한 Windows 보안 설정은 변경하지 않았다.
