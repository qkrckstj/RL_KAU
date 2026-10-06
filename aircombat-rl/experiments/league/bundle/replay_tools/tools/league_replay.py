"""Re-evaluate a portable league policy, or restore only its historical inputs."""
from argparse import ArgumentParser
from pathlib import Path
from tools.autolab_cem import ROOT,read,write,sha
from tools.league_matches import evaluate_jobs
from tools.league_package import copy_checked


def checked_path(root,relative):
    path=(root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes its expected directory')
    return path


def load_bundle(bundle):
    manifest=read(bundle/'manifest.json')
    for path,digest in manifest['files'].items():
        if sha(checked_path(bundle,path))!=digest:
            raise ValueError(f'Artifact hash mismatch: {path}')
    return manifest


def resolve_spec(bundle,spec):
    result=dict(spec)
    if result['kind']=='submission':
        result['design']=str(checked_path(bundle,result['design']))
        result['weights']=str(checked_path(bundle,result['weights']))
    return result


def run(bundle,model,out,band,n,restore=False,transfer=False,smoke=False):
    manifest=load_bundle(bundle)
    if restore:
        for target,source in manifest['restore_inputs'].items():
            copy_checked(checked_path(bundle,source),checked_path(ROOT,target))
        print('Historical opponent inputs verified/restored; different existing files were not overwritten.')
        return
    # Runtime code is part of the benchmark. A fresh future implementation must
    # not silently claim the same old experiment without the same source.
    plan=read(bundle/'evidence/development_plan.json')
    for path,digest in plan['source_sha256'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError(f'Benchmark source changed: {path}; use a matching checkout/snapshot')
    if out is None or out.exists():
        raise FileExistsError('Provide a new --out directory')
    own=resolve_spec(bundle,manifest['models'][model])
    foes=manifest['opponents']+(manifest['transfer_opponents'] if transfer else [])
    if smoke:
        foes=[p for p in foes if p['id']=='ace']
    jobs=[dict(own=own,foe=resolve_spec(bundle,p),band=band,n=n) for p in foes]
    write(out/'plan.json',dict(model=model,band=band,n=n,transfer=transfer,smoke=smoke,
          purpose='Open replay/development evaluation, not a new sealed final test',bundle_manifest_sha256=sha(bundle/'manifest.json')))
    results=evaluate_jobs(jobs,out/'matches')
    write(out/'results.json',results)
    write(out/'completion.json',dict(status='complete'))
    print([(r['foe'],r['summary']['wins'],r['summary']['draws'],r['summary']['losses']) for r in results])


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--model',default='final')
    p.add_argument('--out',type=Path)
    p.add_argument('--band',type=int,default=33010000)
    p.add_argument('--n',type=int,default=40)
    p.add_argument('--restore-inputs',action='store_true')
    p.add_argument('--transfer',action='store_true')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    run(a.bundle.resolve(),a.model,a.out.resolve() if a.out else None,a.band,a.n,a.restore_inputs,a.transfer,a.smoke)
