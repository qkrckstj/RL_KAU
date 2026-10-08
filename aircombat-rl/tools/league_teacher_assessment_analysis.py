"""Verify the wider existing-controller comparison before selecting a teacher."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_sampled_assess import analyze
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_residual_train import validate_records


def verify(folder):
    plan=io.read(folder/'plan.json');io.verify(plan);done=io.read(folder/'completion.json')
    assert done['status']=='sampled_fresh_assessment_complete'
    paths=[folder/'plan.json',folder/'completion.json',folder/'analysis.json',Path(__file__).resolve()];rows=[]
    for i,j in enumerate(plan['jobs']):
        p=folder/f'matches/match_{i:04d}.json';d=io.read(p);paths.append(p)
        assert d['job']==j;validate_job_result(j,d['result']);rows.append(d['result'])
    for specs in plan['roles'].values():
        for s in specs:validate_records([r for r in rows if r['own']==s['id']],plan['opponents'],plan['band'],plan['n'])
    computed=analyze(plan,rows);assert computed==io.read(folder/'analysis.json')
    games=sum(len(r['episodes']) for r in rows);assert games==done['games']==4800
    result=dict(status='teacher_assessment_raw_verified',games=games,input_sha256={io.relative(p):io.sha(p) for p in paths},roles=computed['roles'],comparisons=computed['comparisons'],role_meanings=plan['role_meanings'],comparison_sign='current PPO minus comparator',policy_promoted=False)
    io.write(folder/'raw_teacher_analysis.json',result)
    print(result['status'],games)
    for k,v in computed['roles'].items():print(k,plan['role_meanings'][k],{x:v[x] for x in ['mean_win_rate','group_balanced_win_rate','conservative_lower_quarter','worst_score']})
    print(computed['comparisons'])


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();verify(a.folder.resolve())
