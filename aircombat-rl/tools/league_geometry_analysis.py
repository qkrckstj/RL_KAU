"""Recompute diagnostic outcomes and descriptive geometry from immutable games."""
from argparse import ArgumentParser
from pathlib import Path
from statistics import median
import math
from tools import league_thread_benchmark as io
from tools.league_sampled_numpy_qualify import validate_job_result


def analyze(folder):
    plan=io.read(folder/'plan.json');io.verify(plan)
    assert io.read(folder/'completion.json')['status']=='weakness_geometry_complete'
    paths=[folder/'plan.json',folder/'completion.json',Path(__file__).resolve()]
    for i,j in enumerate(plan['qualification_jobs']):
        p=folder/f'qualification/check_{i:03d}.json';r=io.read(p);paths.append(p)
        assert r['job']==j and r['exact_episode_summary_equality'] and r['games']==2*j['n']
    rows=[]
    for i,j in enumerate(plan['jobs']):
        p=folder/f'matches/match_{i:03d}.json';d=io.read(p);paths.append(p)
        assert d['job']==j
        # Observer schema omits band; validate raw seed/seat pairs against plan.
        validate_job_result(dict(j,traced=False),dict(d['result'],band=j['band']))
        r=d['result'];expected={(e['seed'],e['seat']) for e in r['episodes']}
        assert len(r['traces'])==j['n'] and {(t['seed'],t['seat']) for t in r['traces']}==expected
        for t in r['traces']:
            g=t['geometry_1s'];assert g and g[0]['t']==0
            assert all(b['t']>a['t'] for a,b in zip(g,g[1:]))
            for sample in g:
                assert all(math.isfinite(v) for v in sample.values() if v is not None)
                assert sample['range_m']>=0 and abs(sample['own_nose_error_deg'])<=180 and abs(sample['enemy_nose_error_deg'])<=180
            e=next(e for e in r['episodes'] if (e['seed'],e['seat'])==(t['seed'],t['seat']))
            assert abs(t['damage']['terminal']['t']-e['t'])<.11
        rows.append(r)
    result={}
    med=lambda v:median(v) if v else None
    for role,specs in plan['roles'].items():
        ids={s['id'] for s in specs};result[role]={}
        for foe in plan['opponents']:
            selected=[r for r in rows if r['own'] in ids and r['foe']==foe['id']]
            traces=[t for r in selected for t in r['traces']]
            hit=[t['damage']['first_hit'] for t in traces if t['damage']['first_hit'] is not None]
            attack=[t['damage']['first_enemy_hit'] for t in traces if t['damage']['first_enemy_hit'] is not None]
            end=[t['geometry_1s'][-1] for t in traces]
            result[role][foe['id']]=dict(games=len(traces),**{k:sum(r['summary'][k] for r in selected) for k in ('wins','draws','losses')},first_hit_median_s=med(hit),first_attack_median_s=med(attack),no_attack_games=len(traces)-len(attack),terminal_range_median=med([g['range_m'] for g in end]),terminal_speed_median_knots=med([g['own_speed_knots'] for g in end]))
    out=dict(status='geometry_raw_verified',games=sum(len(r['episodes']) for r in rows),roles=result,input_sha256={io.relative(p):io.sha(p) for p in paths},scope='Descriptive diagnosis on four shared ICs, both seats, known opponents; no causal attribution or generalization claim. Clock medians condition on events occurring. Geometry logged at approximately1Hz, damage events at physics steps.')
    io.write(folder/'raw_geometry_analysis.json',out)
    return out


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['games'])
