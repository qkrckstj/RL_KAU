"""Package completed temporal search and contextual pilot, excluding live work."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
from copy import deepcopy
import ast
import json
import math
import shutil
import subprocess
import numpy as np
from tools.autolab_cem import ROOT,read,sha
from tools.league_train import verify
from tools.league_matches import Actor
from tools.league_completed_replay import stage_inputs

SOURCES=('runs/league_temporal_repair_20261007','runs/league_contextual_speed_pilot_20261007')
TESTS=('tests/test_league_temporal_repair_train.py','tests/test_league_contextual_speed.py',
       'tests/test_league_contextual_speed_pilot.py','tests/test_league_completed_replay.py')
OMIT={'runtime.json','launcher_runtime.json','queue_runtime.json','initialized_workers.json','progress.json'}


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


VERIFY_CODE='''from pathlib import Path
import argparse,hashlib,json,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path);p.add_argument('--check-runtime',action='store_true');a=p.parse_args()
snapshot=Path(__file__).resolve().parent;m=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'));root=(a.workspace or snapshot/'workspace_overlay').resolve()
for name,entry in m['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError(name)
for name,digest in m['artifact_files'].items():
    path=(snapshot/name).resolve()
    if not path.is_relative_to(snapshot) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError(name)
result={'files_verified':len(m['files'])}
if a.check_runtime:
    if a.workspace is None:raise ValueError('Restore base checkout and overlay first')
    code="""from pathlib import Path
import sys,json,numpy as np
import aircombat_gym.wvr.envs.fair as fair
from tools.autolab_cem import ROOT,read
from tools.league_matches import Actor
from tools.league_completed_replay import stage_inputs
from tools.league_assessment_replay import stage_inputs as old_inputs
assert ROOT.resolve()==Path.cwd().resolve()
assert Path(fair.__file__).resolve().is_relative_to(ROOT.resolve())
cases=read(Path(sys.argv[1]));manifest=read(Path(sys.argv[2]));count=0
for evidence in manifest['evidence_sources']:
    parser=old_inputs if evidence['reader']=='assessment' else stage_inputs
    for stage in evidence['stages']:parser(ROOT/evidence['source'],stage)
for case in cases['models']:
    actor=Actor(case['spec']);actions=[]
    for sequence in cases['observations']:
        actor.reset()
        for obs in sequence:actions.append(list(map(float,actor.act(obs=np.asarray(obs,dtype=np.float32)))))
    assert actions==case['expected_actions'],case['spec']['id']
    count+=len(actions)
print(json.dumps(dict(models=len(cases['models']),actions_checked=count,relocated_root=True,official_environment_from_checkout=True,flights_run=0)))
"""
    r=subprocess.run([sys.executable,'-X','utf8','-c',code,str(snapshot/'verification_cases.json'),str(snapshot/'manifest.json')],cwd=root/'aircombat-rl',capture_output=True,text=True,encoding='utf-8')
    if r.returncode:raise RuntimeError(r.stdout+r.stderr)
    result['runtime']=json.loads(r.stdout.strip().splitlines()[-1])
print(json.dumps(result,indent=2))
'''

README='''# 시간 상대 보완 학습과 속도 정책 실험 재현 묶음

기준 커밋 `461284b1b046e890dab8436570ef34b71904e36b`에 적용하는 로컬 묶음이다. GitHub에는 올리지 않았다. 범위는 완료된 `league_temporal_repair_20261007`과 `league_contextual_speed_pilot_20261007`까지다. 뒤에 시작한5계수 이동방향 조건 학습은 포함하지 않는다.

## 완료 결과

두 독립 CEM 검색은6세대/10세대·학습114,400경기, 별도 개발 검증7,700경기를 실행했다. 동일 정책의 평가 사본은 실행 횟수에서 제외했다. 선택 비교4,400경기에서 기준1464승·393무·343패, 후보1469승·374무·357패여서 채택 기준에 미달했다. 개발 개선은 있었으나 선택 단계에서 자격을 얻지 못했다. 동결 제어기의 `no_development_improvement` 상태명이 이를 정확히 표현하지 못하므로 `completed_experiment_analysis.json`의 해석을 읽는다. 최종60000000과 매개변수 홀드아웃61000000은 미사용이다.

후속8개 속도 설정은 선별4,392경기·확인4,392경기와 별도의 동작 검사100회를 실행했다. 확인에서 기준895승·339무·230패, 적극적c0는827승·400무·237패, 보수적c6는899승·342무·223패였다. c0는 고속 회피 우측0승→9승/24경기로 개선했으나 Ace24승→8승으로 악화됐다. 두 후보 모두 자격 기준을 통과하지 못했고 기존 분리 조합을 유지했다. 이 실험은 개발용이며 최종 성능 증거가 아니다.

## 포함 범위

완료된 두 실험의 모든 후보 모델, 집계 파일에 담긴 원시 에피소드, 세대별 분포/RNG 상태와 결과, 평가 재사용 기록, 소스/입력 해시, 분석 보고서와 재계산 코드, 16작업자 실행 방식의 완료 벤치마크를 포함한다. 이전 분리 조합 최종/시간 상대 자료와 이전 묶음의 보존 자료도 포함한다. 중복된 상대별 개별 파일은 제외했으며 같은 원시 에피소드가 후보 집계 파일에 남아 있다.

가상환경, 실행 프로세스, 런타임 PID 파일, 진행 중인5계수 학습, 중간 실행 재개는 포함하지 않는다. 계획/완료 기록 안의 역사적 PID·Windows 경로는 실행 상태가 아니다. 원래 대기용 제어기의 `--after-pid`를 다른 컴퓨터에서 다시 실행하지 않는다. 아래 재현 실행기는 이전 PC의 프로세스를 요구하지 않는다.

검증 범위와 수치는 `verification_report.json`을 읽는다. 별도 경로에서 파일/입력/모델 행동과 관련 검사를 확인하며, 전체 학습·전체 대결을 다시 실행했다는 뜻은 아니다.

## 복원과 검증

기준 커밋의 새 저장소를 준비하고 `workspace_overlay` 내용을 저장소 루트에 적용한다. 루트 아래 `aircombat-rl` 폴더가 있어야 한다. `REPRODUCE.md`와 `manifest.json`의 환경 기록에 따라 의존성을 준비한다. 복원한 `aircombat-rl`에서:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$snapshotPath = 'C:\\path\\to\\replay_snapshot_contextual_20261007'
python "$snapshotPath\\verify.py" --workspace .. --check-runtime
python runs/league_temporal_repair_20261007/analysis_completed_searches.py
```

## 완료 경기 다시 실행

```powershell
python -m tools.league_completed_replay --source runs/league_temporal_repair_20261007 --stage selection --out runs/replayed_temporal_selection --workers 3
python -m tools.league_completed_replay --source runs/league_contextual_speed_pilot_20261007 --stage confirmation --out runs/replayed_context_confirmation --workers 3
```

완료된 학습 세대도 `--stage s3001/g7/screen`처럼 지정할 수 있다. 출력 폴더는 새 경로여야 한다. 초기조건·상대·좌석은 기존 기록을 그대로 재사용하고 처리 시간을 제외한 기록을 비교해 `replay_matched`/`replay_differed`를 남긴다. 하드웨어·설치 버전 차이로 결과가 달라질 수 있으며 이를 숨기지 않는다. 재현은 새 독립 평가가 아니다. 3작업자는 시작 예시이며 다른 컴퓨터의 처리 속도와 메모리 할당 여유를 측정해 정한다.

## 다음 학습과 데이터 구분

여기까지의 조건57000000·57010000·62000000·62010000은 개발에 사용됐다. 이전 최종58000000·시간 상대52000000도 이미 소비됐다. 이들을 새 미학습 증거로 쓰지 않는다. 원래 작업 PC에서는 이후 공개 이동방향 조건의5계수 CEM이 시작됐으나 그 소스·결과는 이 묶음의 범위 밖이다. 새 에이전트는 이 묶음을 완료된 근거로 읽고 새로운 실행 폴더·독립 RNG·조건을 정해야 한다. 실제 상대 학생 정책 전체에 대한 우월성이나 전 상황 학습을 입증하지 않았다.

공식 물리/판정과 기존 정책을 보존한다. GitHub 업로드는 사용자의 새 명시적 요청이 있을 때만 한다.
'''


def build(old,out):
    if out.exists():raise FileExistsError('Use a new snapshot folder')
    prior=read(old/'manifest.json')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if commit!=prior['base_commit']:raise ValueError('Changed base commit')
    for source in SOURCES:
        if not (ROOT/source/'completion.json').exists():raise ValueError('Incomplete source')
    stages={SOURCES[0]:['selection'],SOURCES[1]:['screen','confirmation']}
    for source,rows in stages.items():
        for stage in rows:stage_inputs(ROOT/source,stage)
    overlay=out/'workspace_overlay';overlay.mkdir(parents=True)
    for name,entry in prior['files'].items():
        original=old/'workspace_overlay'/name;target=(overlay/name).resolve()
        if not target.is_relative_to(overlay.resolve()) or sha(original)!=entry['sha256']:raise ValueError(name)
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,target)
    names=set(TESTS)|{'tools/league_completed_replay.py','tools/league_contextual_snapshot.py','tools/league_split_benchmark.py'}
    source_roots=[ROOT/s for s in SOURCES]+[ROOT/'runs/league_split_benchmark_20261007']
    for source in source_roots:
        for p in source.rglob('*'):
            if not p.is_file() or p.suffix not in ('.json','.py','.zip','.pth','.md'):continue
            if p.name in OMIT or any(n=='__pycache__' or n.endswith('_matches') for n in p.parts):continue
            names.add(p.relative_to(ROOT).as_posix())
    pending=[p/'plan.json' for p in source_roots];seen=set()
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        seen.add(p);data=read(p);names.add(p.relative_to(ROOT).as_posix())
        if not all(isinstance(data.get(k),dict) for k in ('source_sha256','input_sha256')):continue
        verify(data)
        for table in ('source_sha256','input_sha256'):
            names.update(data[table])
            pending.extend(ROOT/n for n in data[table] if n.endswith('/plan.json'))
    pending=list(TESTS);seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        seen.add(name)
        for node in ast.walk(ast.parse((ROOT/name).read_text(encoding='utf-8'))):
            modules=[]
            if isinstance(node,ast.Import):modules=[x.name for x in node.names]
            elif isinstance(node,ast.ImportFrom):modules=[f'tests.{x.name}' for x in node.names] if node.module=='tests' else [node.module or '']
            for module in modules:
                if module.startswith('tests.'):
                    dependency=module.replace('.','/')+'.py';names.add(dependency);pending.append(dependency)
    for name in sorted(names):
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT):raise ValueError('Escaping source path')
        target=overlay/'aircombat-rl'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    # Scoped handoff, not the live workspace's process/status documents.
    (overlay/'START_HERE.md').write_text(README,encoding='utf-8')
    (overlay/'AGENTS.md').write_text('Read START_HERE.md and docs/LEAGUE_STATE.json. This snapshot ends at the completed contextual-speed pilot. No running process or current pursuit-context training is included. Preserve official rules, policies and frozen hashes. Use tools.league_completed_replay for completed stages; never interpret historical PIDs as live. No GitHub publication without a new explicit user request.\n',encoding='utf-8')
    write(overlay/'docs/LEAGUE_STATE.json',dict(objective_status='active_not_achieved',snapshot_scope=list(SOURCES),
        current_experiment=None,completed_sources={s:read(ROOT/s/'completion.json') for s in SOURCES},
        retained_policy=read(ROOT/SOURCES[1]/'completion.json')['retained_policy'],publication_policy='No automatic GitHub publication',
        pending='Current original-PC pursuit-context learning is outside this snapshot; use completed evidence to plan a new experiment.'))
    cases=deepcopy(read(old/'verification_cases.json'));old_actions={json.dumps(c['spec'],sort_keys=True):c['expected_actions'] for c in cases['models']}
    for turn in (0.,.1,-.1):
        sequence=[]
        for step in range(201):
            x=list(cases['observations'][0][0]);elapsed=step*.1
            x[0:6]=[0.,0.,3000.,0.,450*.514444,0.];x[15:21]=[0.,4000.,3000.,600*.514444*math.sin(turn*elapsed),600*.514444*math.cos(turn*elapsed),0.]
            x[11]=0.;x[26]=turn*elapsed;x[29]=turn;x[34]=0.;x[38]=120.-elapsed;sequence.append(x)
        cases['observations'].append(sequence)
    specs=[c['spec'] for c in cases['models']];known={json.dumps(s,sort_keys=True) for s in specs}
    for source in source_roots[:2]:
        for p in sorted((source/'models').glob('*/entrant.json')):
            spec=read(p);key=json.dumps(spec,sort_keys=True)
            if key not in known:known.add(key);specs.append(spec)
    models=[]
    for spec in specs:
        actor=Actor(spec);actions=[]
        for sequence in cases['observations']:
            actor.reset()
            for obs in sequence:actions.append(list(map(float,actor.act(obs=np.asarray(obs,dtype=np.float32)))))
        key=json.dumps(spec,sort_keys=True)
        if key in old_actions and actions[:len(old_actions[key])]!=old_actions[key]:raise ValueError('Old model behavior changed')
        models.append(dict(spec=spec,expected_actions=actions))
    cases['models']=models;write(out/'verification_cases.json',cases)
    (out/'README.md').write_text(README,encoding='utf-8');(out/'verify.py').write_text(VERIFY_CODE,encoding='utf-8')
    files={p.relative_to(overlay).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(overlay.rglob('*')) if p.is_file()}
    evidence=[dict(source=prior['source'],stages=prior['stages'],reader='assessment')]+[dict(source=s,stages=rows,reader='completed') for s,rows in stages.items()]
    manifest=dict(created_at=datetime.now(timezone.utc).isoformat(),base_commit=commit,evidence_sources=evidence,
        environment=read(ROOT/SOURCES[1]/'plan.json')['environment'],files=files,
        artifact_files={n:sha(out/n) for n in ('README.md','verify.py','verification_cases.json')},
        tests=list(TESTS),models=len(models),scope='Completed evidence through contextual-speed pilot. No current pursuit-context learning, virtualenv, processes, mid-run resume or fresh performance claim.')
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(folder=str(out),files=len(files),models=len(models),actions_per_model=sum(map(len,cases['observations'])),uncompressed_overlay_bytes=sum(x['bytes'] for x in files.values()))))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__);p.add_argument('--old',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    build(a.old.resolve(),a.out.resolve())
