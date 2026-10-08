"""Two independent CEM searches over seven public-state tactical parameters."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
import json,os,shutil,traceback,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_matches import duel,initialize_worker
from tools.league_residual_train import validate_records,development_panel
from tools.league_tournament_metrics import code_groups,profile
from experiments.league.extend_parameter_policy import Policy,NAMES,LOW,HIGH,BASELINE


def rank(p):
    average=.5*(p['mean_win_rate']+p['group_balanced_win_rate'])
    return [.75*average+.25*p['lower_quarter_score'],p['lower_quarter_score'],p['worst_score'],-p['losing_matchups']]


def export(folder,parameters,identity):
    config=dict(zip(NAMES,map(float,parameters)));Policy(configuration=config)
    folder.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(io.ROOT/'experiments/league/extend_parameter_policy.py',folder/'policy.py')
    (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
    with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('parameters.json',json.dumps(config,sort_keys=True))
    spec=dict(id=identity,kind='submission',design=io.relative(folder),weights=io.relative(folder/'policy_net.zip'))
    io.write(folder/'entrant.json',spec);io.write(folder/'parameters.json',config)
    return spec


def evaluate(pool,spec,foes,band,n,folder):
    rows=[];jobs=[dict(own=spec,foe=f,band=band,n=n) for f in foes]
    futures={pool.submit(duel,**j):(i,j) for i,j in enumerate(jobs)}
    for future in as_completed(futures):
        i,j=futures[future];r=future.result();rows.append(r)
        io.write(folder/f'matches/match_{i:03d}.json',dict(job=j,result=r))
        if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<1.8:raise MemoryError('Resource floor')
    validate_records(rows,foes,band,n);p=profile(rows,code_groups(foes))
    r=dict(spec=spec,opponents=foes,band=band,n=n,profile=p,rank=rank(p),games=len(foes)*n)
    io.write(folder/'result.json',r);return r


def qualify(job):
    a=duel(job['original'],job['foe'],job['band'],2)
    b=duel(job['baseline'],job['foe'],job['band'],2)
    assert a['episodes']==b['episodes'] and a['summary']==b['summary']
    return dict(job=job,original=a,baseline=b,exact_episode_summary_equality=True)


def freeze(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Assessment still live')
    old=io.read(source/'plan.json');io.verify(old);raw=io.read(source/'raw_transfer_analysis.json')
    if raw['status']!='extend_transfer_raw_verified':raise ValueError('Verified transfer required')
    for p,h in raw['input_sha256'].items():
        if io.sha(io.ROOT/p)!=h:raise ValueError('Changed transfer evidence')
    parts=raw['analysis']['partitions']
    gains={role:{part:parts[part]['roles'][role]['mean_win_rate']-parts[part]['roles']['early']['mean_win_rate'] for part in ['archive','reactive_untrained']} for role in ['candidate','warm']}
    if all(v>=.005 for r in gains.values() for v in r.values()):raise ValueError('Consistent neural transfer gains; reconsider tactical switch')
    teacher=old['roles']['teacher'][0];foes=old['opponents'];groups=code_groups(foes)
    scores=raw['analysis']['roles']['teacher']['opponents']
    weak=sorted(scores,key=lambda k:((scores[k]['wins']+.5*scores[k]['draws'])/scores[k]['games'],k))[:8]
    training=[next(s for s in foes if s['id']==k) for k in weak]
    seen={groups[s['id']] for s in training}
    for s in foes:
        if len(training)==24:break
        if groups[s['id']] not in seen:training.append(s);seen.add(groups[s['id']])
    for s in foes:
        if len(training)==24:break
        if s not in training:training.append(s)
    dev=development_panel(foes,groups,weak,teacher['id'],size=32)
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_extend_parameter_analysis.py',io.ROOT/'experiments/league/extend_parameter_policy.py',io.ROOT/'tests/test_extend_parameters.py']:sources[io.relative(p)]=io.sha(p)
    for p in [source/'plan.json',source/'completion.json',source/'raw_transfer_analysis.json',io.ROOT/'runs/extend_parameter_checks_20261007.xml']:inputs[io.relative(p)]=io.sha(p)
    import xml.etree.ElementTree as ET
    for s in ET.parse(io.ROOT/'runs/extend_parameter_checks_20261007.xml').iter('testsuite'):
        assert int(s.get('failures','0'))==int(s.get('errors','0'))==0 and int(s.get('tests','0'))>=2
    plan=dict(source=io.relative(source),teacher=teacher,opponents=foes,groups=groups,training_opponents=training,development_opponents=dev,
        neural_gain_evidence=gains,weakness_ids=weak,seeds=[6800,6801],training_bands=[290000000,291000000],development_band=99000000,
        training_n=4,development_n=8,workers=8,population=10,generations=4,elites=3,
        parameter_names=list(NAMES),low=list(LOW),high=list(HIGH),baseline_parameters=list(BASELINE),initial_std=.23,min_std=.04,
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),
        scope='CEM tactical parameter learning, NOT PPO updates. Two independent search RNGs/training IC sets; common baseline. All132 prior opponents preserved;24training/32development panel. Reactive scripts are now consumed and no longer heldout. Official physics/actions/verdicts unchanged. No promotion/final/GitHub claim.',
        selection_rule='Training objective=.75*(uniform+codegroup win)/2+.25 lower-quarter score; then tail/worst/loss count tie-breaks. Each generation retains incumbent and baseline. Cache exact same parameter vectors on identical training conditions; no double-counted executions. One nominee per search frozen on training results before shared fresh99M development. Independent search repeats are not independent architectures.')
    for band,n in [(b,4) for b in plan['training_bands']]+[(plan['development_band'],8)]:
        claim=io.ROOT/f'runs/extend_parameter_reservation_{band}.json'
        if claim.exists():raise FileExistsError('Conditions reserved')
        io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2))
    io.verify(plan);io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


def run(source,out):
    plan=freeze(source,out)
    if min(io.resources()[k] for k in ['commit_headroom_gib','available_memory_gib'])<4.5:raise MemoryError('Startup headroom')
    io.write(out/'runtime.json',dict(pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat(),resources=io.resources()))
    try:
        baseline=export(out/'baseline',BASELINE,'extend_parameter_baseline')
        with ProcessPoolExecutor(max_workers=8,initializer=initialize_worker) as pool:
            ids=['ace','evader','temporal_extend_left',plan['opponents'][-12]['id']]
            jobs=[dict(original=plan['teacher'],baseline=baseline,foe=next(s for s in plan['opponents'] if s['id']==k),band=170410000) for k in ids]
            futures={pool.submit(qualify,j):i for i,j in enumerate(jobs)}
            for future in as_completed(futures):io.write(out/f'qualification/check_{futures[future]:02d}.json',future.result())
            io.write(out/'qualification.json',dict(status='exact_baseline_teacher_reproduction',games=16))
            print('tactical baseline reproduction passed16 games',flush=True)
            completed=[];lo,hi=np.array(LOW),np.array(HIGH);initial=(np.array(BASELINE)-lo)/(hi-lo)
            for ordinal,seed in enumerate(plan['seeds']):
                rng=np.random.default_rng(seed);mean=initial.copy();std=np.full(len(mean),plan['initial_std']);incumbent=initial.copy();best=None;cache={}
                for generation in range(plan['generations']):
                    io.verify(plan);population=np.clip(rng.normal(mean,std,(plan['population'],len(mean))),0.,1.)
                    population[0]=incumbent;population[1]=initial;records=[]
                    for i,unit in enumerate(population):
                        parameters=lo+unit*(hi-lo)
                        if np.array_equal(unit,initial):parameters=np.array(BASELINE)
                        key=tuple(parameters);folder=out/f's{seed}/g{generation:02d}/c{i:02d}'
                        if key in cache:r=cache[key];reused=True
                        else:
                            spec=baseline if np.array_equal(parameters,BASELINE) else export(folder/'policy',parameters,f'extend_cem_s{seed}_g{generation}_c{i}')
                            ev=evaluate(pool,spec,plan['training_opponents'],plan['training_bands'][ordinal],plan['training_n'],folder/'training')
                            r=dict(parameters=parameters.tolist(),unit_parameters=unit.tolist(),evaluation=io.relative(folder/'training/result.json'),spec=spec,rank=ev['rank']);cache[key]=r;reused=False
                        io.write(folder/'candidate.json',dict(record=r,reused=reused));records.append(r)
                        if best is None or tuple(r['rank'])>tuple(best['rank']):best=r
                        print('tactical CEM',seed,generation,i,'rank',r['rank'],'reused',reused,flush=True)
                    elite=population[sorted(range(len(records)),key=lambda i:tuple(records[i]['rank']),reverse=True)[:plan['elites']]]
                    mean=.5*mean+.5*elite.mean(0);std=np.maximum(plan['min_std'],.5*std+.5*elite.std(0));incumbent=np.array(best['unit_parameters'])
                    io.write(out/f's{seed}/generation_{generation:02d}.json',dict(records=records,selected=best,next_mean=mean.tolist(),next_std=std.tolist()))
                io.write(out/f's{seed}/frozen_selection.json',best)
                dev=evaluate(pool,best['spec'],plan['development_opponents'],plan['development_band'],plan['development_n'],out/f's{seed}/development')
                completed.append(dict(seed=seed,selected=best,development=dev,unique_training_candidates=len(cache)))
                io.write(out/'completed_searches.json',dict(searches=completed))
            base_dev=evaluate(pool,baseline,plan['development_opponents'],plan['development_band'],plan['development_n'],out/'baseline_development')
        io.verify(plan)
        io.write(out/'completion.json',dict(status='extend_parameter_cem_complete',searches=completed,baseline=base_dev,qualification_games=16,policy_promoted=False,final_opened=False,heldout_opened=False))
    except BaseException:io.write(out/'failure.json',dict(traceback=traceback.format_exc(),resources=io.resources()));raise


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.source.resolve(),a.out.resolve())
    from tools.league_extend_parameter_analysis import analyze
    result=analyze(a.out.resolve());print(result['status'],result['games'],flush=True)
