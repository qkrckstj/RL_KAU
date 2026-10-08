"""Build a local overlay from completed assessment evidence; never publish it."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import ast
import json
import shutil
import subprocess
import sys
import zipfile
import numpy as np

from tools.autolab_cem import ROOT, read, sha
from tools.league_reliable_io import write
from tools.league_train import verify
from tools.league_matches import Actor
from tools.league_assessment_replay import stage_inputs


TESTS = ['tests/test_league_tournament_assess.py', 'tests/test_league_tournament_metrics.py',
    'tests/test_league_thread_benchmark_v2.py', 'tests/test_league_assessment_replay.py',
    'tests/test_league_separated_probe.py', 'tests/test_league_separated_validate.py',
    'tests/test_league_validation_report.py']


def build(source, old, out, check, probe=None):
    if out.exists() or check.exists():
        raise FileExistsError('Use new snapshot and check folders')
    completed = read(source / 'completion.json')
    stages = (['selection'] if (source/'selection/schedule.json').exists() else [])
    stages += (['final'] if completed['final_opened'] else []) + (['temporal'] if completed['temporal_opened'] else [])
    if not stages: raise ValueError('No completed assessment stage')
    plan = read(source / 'plan.json'); verify(plan)
    for stage in stages: stage_inputs(source, stage)
    prior = read(old / 'manifest.json')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if commit != prior['base_commit']:
        raise ValueError('Base checkout differs from preceding overlay')
    overlay = out / 'workspace_overlay'; overlay.mkdir(parents=True)
    check.mkdir(parents=True)
    for name, entry in prior['files'].items():
        path = old / 'workspace_overlay' / name
        if sha(path) != entry['sha256']: raise ValueError(f'Prior overlay changed: {name}')
        target = overlay / name
        if not target.resolve().is_relative_to(overlay.resolve()): raise ValueError('Escaping overlay path')
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    names = set(TESTS) | {'tools/league_assessment_replay.py', 'tools/league_assessment_snapshot.py',
        'tools/league_validation_report.py', 'experiments/league/adaptive_progress.md'}
    if 'final_models' in plan:
        analysis=source/'analysis_auto'
        if read(analysis/'completion.json')['status']!='analysis_complete':
            raise ValueError('Raw-result analysis must complete before packaging final validation')
        names.update(p.relative_to(ROOT).as_posix() for p in analysis.iterdir()
            if p.is_file() and p.suffix in ('.json','.jsonl','.md'))
    pending = [source / 'plan.json']; seen = set()
    if probe is not None:
        diagnostic=read(probe/'plan.json')
        if diagnostic['input_sha256'].get((source/'plan.json').relative_to(ROOT).as_posix()) != sha(source/'plan.json'):
            raise ValueError('Diagnostic belongs to another assessment')
        if read(probe/'completion.json')['status'] != 'diagnostic_complete': raise ValueError('Incomplete diagnostic')
        pending.append(probe/'plan.json')
        names.update(p.relative_to(ROOT).as_posix() for p in probe.glob('*.json'))
    while pending:
        path = pending.pop().resolve()
        if path in seen: continue
        seen.add(path); data = read(path)
        names.add(path.relative_to(ROOT).as_posix())
        # Old diagnostic plans used a scalar source digest. Their complete
        # bytes are already protected by the referencing plan's input hash;
        # they are evidence files, not modern recursive dependency manifests.
        if not all(isinstance(data.get(k), dict) for k in ('source_sha256', 'input_sha256')):
            continue
        verify(data)
        for table in ('source_sha256', 'input_sha256'):
            names.update(data[table])
            for name in data[table]:
                if name.endswith('/plan.json'):
                    pending.append(ROOT / name)
    for name in ('completion.json', 'frozen_selection.json', 'final_decision.json', 'selection_analysis.json',
                 'physics_preservation_audit.json'):
        if (source / name).exists(): names.add((source / name).relative_to(ROOT).as_posix())
    for stage in stages:
        for path in (source / stage).glob('*.json'):
            if path.name != 'reuse.json': names.add(path.relative_to(ROOT).as_posix())
    repair = ROOT / plan.get('training_source',plan['previous'])
    for path in [repair / 'completed_searches_analysis.json'] + list(repair.glob('s*/result.json')):
        names.add(path.relative_to(ROOT).as_posix())
    if 'benchmark' in plan:
        benchmark = ROOT / plan['benchmark']
        for path in benchmark.glob('r*/result.json'): names.add(path.relative_to(ROOT).as_posix())
    for band in (plan['final_band'], plan['temporal']['band']):
        path = ROOT / f'runs/holdout_claim_{band}.json'
        if path.exists(): names.add(path.relative_to(ROOT).as_posix())
    pending_tests = list(TESTS); seen_tests = set()
    while pending_tests:
        name = pending_tests.pop()
        if name in seen_tests: continue
        seen_tests.add(name)
        for node in ast.walk(ast.parse((ROOT / name).read_text(encoding='utf-8'))):
            modules = []
            if isinstance(node, ast.Import): modules = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [f'tests.{x.name}' for x in node.names] if node.module == 'tests' else [node.module or '']
            for module in modules:
                if module.startswith('tests.'):
                    dependency = module.replace('.', '/') + '.py'
                    names.add(dependency); pending_tests.append(dependency)
    specs = plan['opponents'] + [plan['anchor'], plan['parent']] + plan['candidates'] + plan['temporal']['opponents']
    for spec in specs:
        if spec['kind'] == 'submission':
            names.update(path.relative_to(ROOT).as_posix() for path in (ROOT / spec['design']).iterdir()
                if path.is_file() and path.suffix in ('.py', '.json', '.zip', '.pth'))
    for name in names:
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT): raise ValueError('Escaping project path')
        target = overlay / 'aircombat-rl' / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    for name in ('AGENTS.md', 'START_HERE.md', 'REPRODUCE.md', 'docs/LEAGUE_STATE.json'):
        target = overlay / name
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT.parent / name, target)
    cases = read(old / 'verification_cases.json'); models = list(cases['models'])
    known = {json.dumps(m['spec'], sort_keys=True) for m in models}
    for spec in specs:
        key = json.dumps(spec, sort_keys=True)
        if key in known or spec['kind'] == 'bot': continue
        known.add(key); actor = Actor(spec); actions = []
        for sequence in cases['observations']:
            actor.reset()
            for obs in sequence: actions.append(list(map(float, actor.act(obs=np.asarray(obs, dtype=np.float32)))))
        models.append(dict(spec=spec, expected_actions=actions))
    cases['models'] = models; write(out / 'verification_cases.json', cases)
    files = {p.relative_to(overlay).as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size)
        for p in sorted(overlay.rglob('*')) if p.is_file()}
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), base_commit=commit,
        source=source.relative_to(ROOT).as_posix(), status=completed['status'], stages=stages,
        selected=completed.get('selected'), preserved_validated_policy=plan['anchor'],
        environment=plan['environment'], files=files,
        scope='Completed search summaries and assessment raw aggregate episode records; source/input/model overlay on the named base commit. Excludes active processes, venv, full historical training episode logs and exact mid-run resumption. Replaying consumed conditions is not independent validation.')
    write(out / 'manifest.json', manifest)
    verifier = '''from pathlib import Path
import argparse, hashlib, json, subprocess, sys
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path);p.add_argument('--check-runtime',action='store_true');a=p.parse_args()
snapshot=Path(__file__).resolve().parent;m=json.loads((snapshot/'manifest.json').read_text(encoding='utf-8'));root=(a.workspace or snapshot/'workspace_overlay').resolve()
for name,entry in m['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']: raise ValueError(name)
result={'files_verified':len(m['files'])}
if a.check_runtime:
    if a.workspace is None: raise ValueError('Supply restored base checkout plus overlay')
    code="""from pathlib import Path
import json,sys,numpy as np
import aircombat_gym.wvr.envs.fair as fair
from tools.autolab_cem import ROOT,read
from tools.league_matches import Actor
from tools.league_assessment_replay import stage_inputs
assert ROOT.resolve()==Path.cwd().resolve()
assert Path(fair.__file__).resolve().is_relative_to(ROOT.resolve())
cases=read(Path(sys.argv[1]));manifest=read(Path(sys.argv[2]));count=0
for stage in manifest['stages']: stage_inputs(ROOT/manifest['source'],stage)
for case in cases['models']:
    actor=Actor(case['spec']);actions=[]
    for sequence in cases['observations']:
        actor.reset()
        for obs in sequence: actions.append(list(map(float,actor.act(obs=np.asarray(obs,dtype=np.float32)))))
    assert actions==case['expected_actions'],case['spec']['id']
    count+=len(actions)
print(json.dumps(dict(models=len(cases['models']),actions_checked=count,relocated_root=True,official_environment_from_checkout=True,completed_stage_inputs_verified=manifest['stages'],flights_run=0)))
"""
    r=subprocess.run([sys.executable,'-X','utf8','-c',code,str(snapshot/'verification_cases.json'),str(snapshot/'manifest.json')],cwd=root/'aircombat-rl',capture_output=True,text=True,encoding='utf-8')
    if r.returncode: raise RuntimeError(r.stdout+r.stderr)
    result['runtime']=json.loads(r.stdout.strip().splitlines()[-1])
print(json.dumps(result,indent=2))
'''
    (out / 'verify.py').write_text(verifier, encoding='utf-8')
    archive = check / 'base.zip'
    subprocess.run(['git', 'archive', commit, '--format=zip', '--output', str(archive)], cwd=ROOT, check=True)
    checkout = check / 'checkout'; project = checkout / 'aircombat-rl'; project.mkdir(parents=True)
    with zipfile.ZipFile(archive) as z:
        if any(not (project / name).resolve().is_relative_to(project.resolve()) for name in z.namelist()):
            raise ValueError('Escaping base archive path')
        z.extractall(project)
    for name in files:
        target=checkout/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(overlay/name,target)
    verification=subprocess.run([sys.executable,'-X','utf8',str(out/'verify.py'),'--workspace',str(checkout),'--check-runtime'],
        capture_output=True,text=True,encoding='utf-8',check=True)
    runtime=json.loads(verification.stdout)
    tests=subprocess.run([sys.executable,'-X','utf8','-m','pytest','-q',*TESTS],cwd=project,
        capture_output=True,text=True,encoding='utf-8',check=True)
    write(out/'verification_report.json',dict(runtime=runtime,tests_stdout=tests.stdout,tests_stderr=tests.stderr,
        scope='Same installed environment, different checkout path. Synthetic actions and stage-input verification; no additional flights or full repeated training.'))
    print(json.dumps(dict(snapshot=str(out),files=len(files),models=len(models),stages=stages,verified=True)))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--old',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--check',type=Path,required=True);p.add_argument('--probe',type=Path)
    a=p.parse_args();build(a.source.resolve(),a.old.resolve(),a.out.resolve(),a.check.resolve(),a.probe.resolve() if a.probe else None)
