"""Verify the recent-policy cross-play matrix from all immutable raw games."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io
from tools.league_recent_round_robin import analyze
from tools.league_sampled_numpy_qualify import validate_job_result
from tools.league_matches import summary


def verify(folder):
    plan=io.read(folder/'plan.json');io.verify(plan);done=io.read(folder/'completion.json')
    assert done['status']=='recent_round_robin_complete'
    paths=[folder/'plan.json',folder/'completion.json',folder/'analysis.json',Path(__file__).resolve(),io.ROOT/'tests/test_recent_robin_metrics.py'];rows=[]
    for i,j in enumerate(plan['jobs']):
        p=folder/f'matches/match_{i:04d}.json';d=io.read(p);paths.append(p)
        assert d['job']==j;validate_job_result(j,d['result']);assert d['result']['summary']==summary(d['result']['episodes']);rows.append(d['result'])
    analysis=analyze(plan,rows);assert analysis==io.read(folder/'analysis.json')
    assert sum(len(r['episodes']) for r in rows)==done['games']
    out=dict(status='recent_round_robin_raw_verified',games=done['games'],expanded_archive_size=done['expanded_archive_size'],input_sha256={io.relative(p):io.sha(p) for p in paths},policy_promoted=False)
    io.write(folder/'raw_verification.json',out)
    print(out['status'],out['games'])
    for name,r in sorted(analysis['aggregates'].items(),key=lambda kv:kv[1]['equal_role_score'],reverse=True):print(name,r)
    for role in ['baseline','control_s6500','control_s6501','hold5_s6500','hold5_s6501']:
        print('weaknesses',role,{name:analysis['matrix'][role][name] for name in ['temporal_extend_right','temporal_weave_right','evader']})


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();verify(a.folder.resolve())
