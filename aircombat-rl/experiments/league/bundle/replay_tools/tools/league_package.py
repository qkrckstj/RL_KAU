"""Immutable portable policies, opponent inputs and evaluation evidence."""
from argparse import ArgumentParser
from pathlib import Path
import gzip
import json
import shutil
from tools.autolab_cem import ROOT,read,write,sha
from tools.league_train import export,verify


def copy_checked(source,target):
    if target.exists():
        if sha(source)!=sha(target):
            raise FileExistsError(f'Will not overwrite a different artifact: {target}')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)


def package(main,followup,out,final=False):
    plan=read(main/'plan.json')
    verify(plan)
    complete=read(main/'completion.json')
    history=read(main/'history.json')
    if complete['status']!='development_batch_complete' or len(history)!=plan['rounds']:
        raise ValueError('Development batch is not complete')
    if (out/'manifest.json').exists():
        manifest=read(out/'manifest.json')
        if manifest['development_plan_sha256']!=sha(main/'plan.json'):
            raise ValueError('Bundle belongs to another experiment')
        for name,digest in manifest['files'].items():
            if sha(out/name)!=digest:
                raise ValueError(f'Bundled artifact changed: {name}')
    else:
        out.mkdir(parents=True,exist_ok=True)
        manifest=dict(format_version=1,stage='development_only',models={},opponents=[],restore_inputs={},
              development_plan_sha256=sha(main/'plan.json'),development_champion=complete['champion']['id'],files={})
    models=manifest['models']
    original=ROOT/'experiments/plan_a/cem_policy'
    for name in ('policy.py','wrappers.py','policy_net.zip','policy_net.json','selection.json'):
        copy_checked(original/name,out/'models/cem_original'/name)
    def portable(identity,folder):
        return dict(id=identity,kind='submission',design=f'models/{folder}',weights=f'models/{folder}/policy_net.zip')
    models['cem_original']=portable('cem_original','cem_original')
    for record in history:
        spec=record['challenger']
        folder=out/'models'/spec['id']
        if not folder.exists():
            export(folder,spec)
        elif read(folder/'entrant.json')!=spec:
            raise ValueError('Archived model differs')
        models[spec['id']]=portable(spec['id'],spec['id'])
        copy_checked(main/f'round_{record["round"]:02d}/round.json',out/'evidence'/f'round_{record["round"]:02d}.json')
    for identity,design,weights in (
        ('ddqn_s1',plan['roster'][1]['design'],plan['roster'][1]['weights']),
        ('ddqn_s0','runs/plan_a_20261005/double/ddqn/s0','runs/plan_a_20261005/double/ddqn/s0/policy_net.zip')):
        for name in ('policy.py','wrappers.py','utils.py'):
            relative=f'{design}/{name}'
            target=out/'models'/identity/name
            copy_checked(ROOT/relative,target)
            manifest['restore_inputs'][relative]=target.relative_to(out).as_posix()
        copy_checked(ROOT/weights,out/'models'/identity/'policy_net.zip')
        manifest['restore_inputs'][weights]=f'models/{identity}/policy_net.zip'
        models[identity]=portable(identity,identity)
    manifest['restore_inputs']['experiments/plan_a/cem_policy/policy_net.json']='models/cem_original/policy_net.json'
    manifest['opponents']=[dict(models[p['id']]) if p['id'] in models else p for p in plan['roster']]
    manifest['opponents'] += [dict(models[r['challenger']['id']]) for r in history]
    manifest['transfer_opponents']=[dict(id='lead',kind='bot',name='lead'),dict(id='circler',kind='bot',name='circler'),models['ddqn_s0']]
    copy_checked(main/'plan.json',out/'evidence/development_plan.json')
    copy_checked(main/'history.json',out/'evidence/history.json')
    copy_checked(main.parent/'runtime.json',out/'evidence/runtime.json')
    for source in (main/'source_snapshot').rglob('*.py'):
        copy_checked(source,out/'source_snapshot'/source.relative_to(main/'source_snapshot'))
    for name in ('tools/league_replay.py','tools/league_package.py','tools/league_validate.py',
                 'tools/league_runner_probe.py','tools/league_report.py'):
        copy_checked(ROOT/name,out/'replay_tools'/name)
    if final:
        terminal=read(followup/'completion.json')
        if terminal['status'] not in ('evaluation_passed','analysis_required') or terminal['stage']!='heldout':
            raise ValueError('Final evaluation, including refinement cross-play, has not finished')
        frozen=read(followup/'frozen_selection.json')
        for key,spec in [('final',frozen['chosen'])]+[(s['id'],s) for s in frozen['selected']]:
            folder=out/'models'/key
            if not folder.exists():
                export(folder,spec)
            elif read(folder/'entrant.json')!=spec:
                raise ValueError('Final policy changed')
            models[key]=portable(spec['id'],key)
        for name in ('frozen_selection.json','verdict.json','completion.json','refinement_crossplay.json',
                     'plan.json','pool_selection.json','screen_gate.json','league_validate.snapshot.py'):
            copy_checked(followup/name,out/'evidence/final'/name)
        results=[read(p)['result'] for p in sorted((followup/'heldout').glob('match_*.json'))]
        compressed=gzip.compress(json.dumps(results,ensure_ascii=False).encode('utf-8'),mtime=0)
        destination=out/'evidence/final/episodes.json.gz'
        if destination.exists() and destination.read_bytes()!=compressed:
            raise ValueError('Final episode evidence changed')
        destination.write_bytes(compressed)
        manifest['stage']=terminal['status']
        manifest['final_choice']=frozen['chosen']['id']
        manifest['scope']=read(followup/'verdict.json')['scope']
    (out/'.gitignore').write_text('!**/policy_net.zip\n',encoding='utf-8')
    manifest['files']={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*'))
                       if p.is_file() and p.name not in ('manifest.json','README.md') and '__pycache__' not in p.parts}
    write(out/'manifest.json',manifest)
    print(json.dumps(dict(stage=manifest['stage'],models=list(models),files=len(manifest['files'])),ensure_ascii=False))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--main',type=Path,required=True)
    p.add_argument('--followup',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--final',action='store_true')
    a=p.parse_args()
    package(a.main.resolve(),a.followup.resolve(),a.out.resolve(),a.final)
