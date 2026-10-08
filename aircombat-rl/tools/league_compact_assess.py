"""Small fresh-condition, inactive-archive screen after compact48 learning."""
from argparse import ArgumentParser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import os
import shutil
from tools import league_thread_benchmark as io
from tools import league_sampled_assess as engine
from tools import league_policy_assessment_metrics as metrics
from tools.league_rate_budget_assess import unique_specs, verify_results
from tools.league_tournament_metrics import code_groups
from tools.league_numpy_matches import numpy_capable

PROTOCOL=io.ROOT/'experiments/league/compact_inactive_screen_protocol_20261008.json'


def prepare(source):
    assert not PROTOCOL.exists()
    p=io.read(source/'plan.json');io.verify(p)
    parent=io.read(io.ROOT/p['source']/'plan.json')
    active=set(p['groups']); excluded=set(active)
    for key,members in p['roster_manifest']['rng_aliases'].items():
        if key in active:excluded.update(members)
    eligible=[s for s in p['full_retained_archive'] if s['id'] not in excluded]
    groups=code_groups(eligible);estimates=parent['opponent_win_estimates'];selected=[];counts=Counter()
    def rank(s):return (counts[groups[s['id']]],estimates[s['id']]['win_estimate'],s['id'])
    while len(selected)<24:
        ids={s['id'] for s in selected}
        candidates=[s for s in eligible if s['id'] not in ids and counts[groups[s['id']]]<4]
        s=min(candidates,key=rank);selected.append(s);counts[groups[s['id']]]+=1
    protocol=dict(training_run=io.relative(source),prepared_at=datetime.now(timezone.utc).isoformat(),
        opponents=selected,excluded_active_and_rng_alias_ids=sorted(excluded),eligible_count=len(eligible),
        groups=code_groups(selected),band=112000000,n=8,maximum_games=1344,
        selection_rule='24 inactive archive representatives; exclude active IDs and their verified fixed-RNG aliases. Fill least represented code groups first, then weakest consumed109M/110M baseline estimates, cap4/group.',
        nomination_rule='After both pilots finish, evaluate retained models from both seeds; candidate is higher development rank. Source aliases execute once. Run only if at least one pilot passes preregistered extension screen.',
        decision_rule='Use paired candidate-source and other retained-source outcomes, group/tail scores and uncertainty. No automatic extension merely for positive point estimate; no promotion from this small screen.',
        scope='Known archive policies omitted from this48-foe stage; ancestors may have trained against them. Not unseen opponents. Four fresh112M ICs,both seats,two action replicas,plus preserved CEM comparator. Targeted weakness sample,not full205 population estimate.',
        source_sha256={io.relative(Path(__file__)):io.sha(Path(__file__))},
        input_sha256={io.relative(source/'plan.json'):io.sha(source/'plan.json')})
    io.write(PROTOCOL,protocol)
    print('inactive screen frozen',len(selected),'opponents',len(set(protocol['groups'].values())),'groups',flush=True)


def freeze(source,qualification,out):
    assert not out.exists() and not io.process_live(io.read(source/'runtime.json')['pid'])
    assert all(os.environ.get(k)=='1' for k in io.THREAD_KEYS)
    assert min(io.resources()[k] for k in ('commit_headroom_gib','available_memory_gib'))>=6
    old=io.read(source/'plan.json');io.verify(old)
    done=io.read(source/'completion.json');raw=io.read(source/'raw_compact_analysis.json')
    decision=io.read(source/'pilot_decision.json')
    assert raw['status']=='compact_ppo_raw_verified' and decision['passing_repeats']>=1
    protocol=io.read(PROTOCOL);io.verify(protocol);assert protocol['training_run']==io.relative(source)
    qp=io.read(qualification/'plan.json');io.verify(qp)
    qc=io.read(qualification/'completion.json')
    assert qc['status']=='sampled_numpy_execution_qualified' and qc['exact_episode_and_trace_equality']
    ep=io.ROOT/'experiments/league/execution_profile_budget_20261008.json';profile=io.read(ep)
    bench=io.ROOT/'runs/league_budget_pool_benchmark_20261008/completion.json'
    br=io.read(bench);assert br['status']=='mixed_pool_benchmark_verified'
    assert profile['approved_signatures']==qc['approved_signatures']
    for record in (raw,decision,br):
        for name,digest in record['input_sha256'].items():assert io.sha(io.ROOT/name)==digest
    searches=sorted(done['searches'],key=lambda r:tuple(r['selected']['rank']),reverse=True)
    archive=old['full_retained_archive'];cem=next(s for s in archive if s['id']=='extend_cem_s6800_g3_c9')
    roles=dict(candidate=[a['spec'] for a in searches[0]['selected']['replicas']],
        repeat=[a['spec'] for a in searches[1]['selected']['replicas']],
        baseline=[a['spec'] for a in done['baseline']['replicas']],cem=[cem])
    own=unique_specs(roles);opponents=protocol['opponents'];band=protocol['band'];n=protocol['n']
    jobs=[dict(own=s,foe=f,band=band,n=n,traced=False) for s in own for f in opponents]
    assert len(jobs)*n<=protocol['maximum_games']
    assert not set(s['id'] for s in opponents)&set(protocol['excluded_active_and_rng_alias_ids'])
    sources=dict(old['source_sha256']);sources.update(qp['source_sha256']);sources.update(protocol['source_sha256'])
    inputs=dict(old['input_sha256']);inputs.update(qp['input_sha256'])
    for f in [Path(__file__).resolve(),Path(metrics.__file__).resolve()]:sources[io.relative(f)]=io.sha(f)
    files=[source/'plan.json',source/'completion.json',source/'raw_compact_analysis.json',source/'pilot_decision.json',PROTOCOL,
        qualification/'plan.json',qualification/'completion.json',ep,bench]
    for s in own+opponents:
        if s['kind']=='submission':files+=list((io.ROOT/s['design']).glob('*.py'))+[io.ROOT/s['weights']]
    for f in files:inputs[io.relative(f)]=io.sha(f)
    pure=sum(all(numpy_capable(j[k],qc['approved_signatures']) for k in ('own','foe')) for j in jobs)
    plan=dict(source=io.relative(source),qualification=io.relative(qualification),roles=roles,
        selected_checkpoints=[s['selected'] for s in searches],nominee_seed=searches[0]['seed'],repeat_seed=searches[1]['seed'],
        opponents=opponents,groups=code_groups(opponents),partitions={'inactive_archive':[s['id'] for s in opponents]},
        heldout_opponent_ids=[],band=band,n=n,jobs=jobs,workers=profile['workers'],neural_workers=profile['neural_workers'],
        approved_signatures=qc['approved_signatures'],pure_warmup=profile['pure_warmup'],neural_warmup=profile['neural_warmup'],
        routing=dict(pure_jobs=pure,ordinary_jobs=len(jobs)-pure),
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),scope=protocol['scope'],next_rule=protocol['decision_rule'])
    claim=io.ROOT/f'runs/sampled_assessment_reservation_{band}.json';assert not claim.exists()
    io.verify(plan);io.write(claim,dict(run=io.relative(out),start=band,stop_exclusive=band+n//2));io.write(out/'plan.json',plan)
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,dest)
    return plan


if __name__=='__main__':
    parser=ArgumentParser();parser.add_argument('--source',type=Path,default=Path('runs/league_compact_ppo_train_20261008'))
    parser.add_argument('--out',type=Path);parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--qualification',type=Path,default=Path('runs/league_episode_mixture_qualification_20261008'))
    args=parser.parse_args()
    if args.prepare:prepare(args.source.resolve())
    else:
        assert args.out is not None
        engine.freeze=freeze;engine.analyze=metrics.analyze
        engine.run(args.source.resolve(),args.qualification.resolve(),args.out.resolve())
        verify_results(args.out.resolve())
