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
목적 컴퓨터의 GPU와 드라이버는 아직 확인하지 않았으므로 CUDA 빌드를 임의로 고정하지 않았다.
실제 CUDA 학습은 이 노트북에서 검증하지 않았다.

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
