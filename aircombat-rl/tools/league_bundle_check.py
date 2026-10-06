"""Check original preservation and the packaged final policy in the official grader.

Uses an already-open development seed, never the final test band. Run after
the simulation workers finish so it does not add another concurrent workload.
"""
from argparse import ArgumentParser
from pathlib import Path
from tools.autolab_cem import ROOT,read,write,sha
from tools.league_replay import load_bundle,resolve_spec
from tools.league_matches import duel
from tools.grade import score
from tools.project import load as load_project


def run(bundle,out):
    manifest=load_bundle(bundle)
    if manifest['stage'] not in ('evaluation_passed','analysis_required') or 'final' not in manifest['models']:
        raise ValueError('Final policy has not been packaged')
    spec=resolve_spec(bundle,manifest['models']['final'])
    expected=read(Path(spec['design'])/'entrant.json')
    if expected['kind']!='reactive':
        raise ValueError('This check expects the frozen 17-parameter league policy')
    official=score(Path(spec['design']),Path(spec['weights']),load_project('templates/project_04_fair'),32000500,2)
    reference=duel(expected,dict(id='ace',kind='bot',name='ace'),32000500,2)
    if official['episodes']!=reference['episodes']:
        raise AssertionError('Official loader/grader changed match behavior')
    original=ROOT/'experiments/plan_a/cem_policy'
    selection=read(original/'selection.json')
    if sha(original/'policy.py')!=selection['policy_sha256']:
        raise AssertionError('Original policy source changed')
    if sha(original/'policy_net.zip')!=selection['archive_sha256']:
        raise AssertionError('Original weights archive changed')
    plan=read(bundle/'evidence/development_plan.json')
    original_json='experiments/plan_a/cem_policy/policy_net.json'
    if sha(ROOT/original_json)!=plan['input_sha256'][original_json]:
        raise AssertionError('Original coefficients changed')
    result=dict(status='passed',bundle_manifest_sha256=sha(bundle/'manifest.json'),
        original_policy_preserved=True,original_weights_preserved=True,
        official_and_search_episodes_identical=True,band=32000500,n=2,
        episodes=official['episodes'],scope='Loader/physics compatibility check, not performance estimation')
    write(out,result)
    print('Original preserved; packaged policy matches the official grader in both seats.')


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    try:
        run(a.bundle.resolve(),a.out.resolve())
    except Exception as e:
        write(a.out,dict(status='failed',error=repr(e)))
        raise
