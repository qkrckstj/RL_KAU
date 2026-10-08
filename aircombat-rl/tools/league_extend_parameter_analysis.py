"""Recompute tactical CEM selection, cached games and all raw profiles."""
from argparse import ArgumentParser
from pathlib import Path
import json,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_residual_train import validate_records
from tools.league_tournament_metrics import profile,code_groups
from tools.league_extend_parameter_train import rank


def analyze(out):
    plan=io.read(out/'plan.json');io.verify(plan);done=io.read(out/'completion.json')
    assert done['status']=='extend_parameter_cem_complete'
    paths=[out/'plan.json',out/'completion.json',Path(__file__).resolve()];games=0;evaluations={}
    for p in sorted(out.rglob('result.json')):
        r=io.read(p);paths.append(p);rows=[]
        for f in sorted((p.parent/'matches').glob('match_*.json')):
            d=io.read(f);paths.append(f);j=d['job'];x=d['result']
            assert j['own']==r['spec'] and x['own']==r['spec']['id'] and j['foe']['id']==x['foe'] and j['band']==r['band'] and j['n']==r['n']
            rows.append(x)
        training=p.parent.name=='training'
        assert r['opponents']==plan['training_opponents' if training else 'development_opponents']
        assert r['n']==plan['training_n' if training else 'development_n']
        assert r['band'] in plan['training_bands'] if training else r['band']==plan['development_band']
        validate_records(rows,r['opponents'],r['band'],r['n'])
        computed=profile(rows,code_groups(r['opponents']));assert r['profile']==computed and r['rank']==rank(computed)
        assert r['games']==sum(len(x['episodes']) for x in rows);games+=r['games'];evaluations[io.relative(p)]=r
    qual=io.read(out/'qualification.json');assert qual['status']=='exact_baseline_teacher_reproduction' and qual['games']==16
    checks=list((out/'qualification').glob('check_*.json'));assert len(checks)==4
    for p in checks:
        q=io.read(p);paths.append(p);assert q['exact_episode_summary_equality'] and q['original']['episodes']==q['baseline']['episodes'] and q['original']['summary']==q['baseline']['summary']
    summaries=[];lo,hi=np.array(plan['low']),np.array(plan['high']);base=np.array(plan['baseline_parameters']);initial=(base-lo)/(hi-lo)
    for ordinal,seed in enumerate(plan['seeds']):
        rng=np.random.default_rng(seed);mean=initial.copy();std=np.full(len(mean),plan['initial_std']);incumbent=initial.copy();best=None;cache={}
        for generation in range(plan['generations']):
            population=np.clip(rng.normal(mean,std,(plan['population'],len(mean))),0.,1.);population[0]=incumbent;population[1]=initial;records=[]
            for i,unit in enumerate(population):
                parameters=lo+unit*(hi-lo)
                if np.array_equal(unit,initial):parameters=base.copy()
                p=out/f's{seed}/g{generation:02d}/c{i:02d}/candidate.json';c=io.read(p);paths.append(p);r=c['record'];key=tuple(parameters)
                assert np.array_equal(parameters,r['parameters'])
                assert c['reused']==(key in cache)
                if key in cache:assert cache[key]==r
                else:
                    cache[key]=r;ev=evaluations[r['evaluation']]
                    assert ev['spec']==r['spec'] and ev['rank']==r['rank'] and ev['band']==plan['training_bands'][ordinal]
                    z=io.ROOT/r['spec']['weights'];paths.append(z)
                    with zipfile.ZipFile(z) as archive:config=json.loads(archive.read('parameters.json'))
                    assert config==dict(zip(plan['parameter_names'],r['parameters']))
                    paths+=list((io.ROOT/r['spec']['design']).glob('*.py'))
                records.append(r)
                if best is None or tuple(r['rank'])>tuple(best['rank']):best=r
            elite=population[sorted(range(len(records)),key=lambda i:tuple(records[i]['rank']),reverse=True)[:plan['elites']]]
            mean=.5*mean+.5*elite.mean(0);std=np.maximum(plan['min_std'],.5*std+.5*elite.std(0));incumbent=np.array(best['unit_parameters'])
            p=out/f's{seed}/generation_{generation:02d}.json';g=io.read(p);paths.append(p)
            assert g['records']==records and g['selected']==best and np.array_equal(g['next_mean'],mean) and np.array_equal(g['next_std'],std)
        p=out/f's{seed}/frozen_selection.json';paths.append(p);assert io.read(p)==best
        r=next(r for r in done['searches'] if r['seed']==seed)
        assert r['selected']==best and r['unique_training_candidates']==len(cache)
        dev=evaluations[io.relative(out/f's{seed}/development/result.json')];assert r['development']==dev and dev['spec']==best['spec']
        summaries.append(dict(seed=seed,win_rate=dev['profile']['mean_win_rate'],group_win_rate=dev['profile']['group_balanced_win_rate'],rank=dev['rank'],parameters=best['parameters'],unique_training_candidates=len(cache),spec=best['spec']))
    assert done['baseline']==evaluations[io.relative(out/'baseline_development/result.json')]
    result=dict(status='extend_parameter_raw_verified',games=games+16,searches=summaries,baseline=done['baseline'],policy_promoted=False,input_sha256={io.relative(p):io.sha(p) for p in paths},scope=plan['scope'])
    io.write(out/'raw_parameter_analysis.json',result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();r=analyze(a.folder.resolve());print(r['status'],r['games'],r['searches'])
