"""Real-flight endpoint reproduction and loader equivalence for episode mixtures."""
from argparse import ArgumentParser
from pathlib import Path
import traceback
from tools import league_thread_benchmark as io
from tools import league_sampled_numpy_qualify as engine
from tools.league_numpy_matches import code_signature,numpy_capable


def freeze(source,out):
    if out.exists():raise FileExistsError(out)
    training=io.ROOT/'runs/league_discount_train_20261008'
    if io.process_live(io.read(training/'runtime.json')['pid']):raise RuntimeError('Training still live')
    tr=io.read(training/'raw_discount_analysis.json');assert tr['status']=='discount_raw_verified'
    manifest=io.read(source/'manifest.json');io.verify(manifest)
    profilepath=io.ROOT/'experiments/league/execution_profile_extend_20261008.json';old=io.read(profilepath)
    qroot=io.ROOT/old['qualification'];qp=io.read(qroot/'plan.json');io.verify(qp)
    assert io.sha(qroot/'completion.json')==old['qualification_completion_sha256']
    candidates=[v['spec'] for v in manifest['variants']];originals=manifest['original_experts']
    signatures={code_signature(s) for s in candidates};assert len(signatures)==1
    # Export includes reviewed original stdlib controllers and NumPy only,
    # no dynamic expert paths, Torch imports, pickle or hidden observations.
    approved=sorted(set(old['approved_signatures'])|signatures)
    assessment=io.read(io.ROOT/manifest['source']/'plan.json');foes={s['id']:s for s in assessment['opponents']}
    jobs=[dict(own=s,foe=foes[f],band=170470000,n=2,traced=f=='temporal_extend_left') for s in candidates+originals for f in ['ace','temporal_extend_left','novel_opening_right']]
    jobs += [dict(own=s,foe=foes['ddqn_s0'],band=170470000,n=2,traced=False) for s in candidates[:2]]
    for j in jobs:j['expected_pure_route']=all(numpy_capable(j[k],approved) for k in ['own','foe'])
    sources=dict(manifest['source_sha256']);sources.update(qp['source_sha256'])
    sources[io.relative(Path(__file__).resolve())]=io.sha(Path(__file__).resolve())
    inputs=dict(manifest['input_sha256']);inputs.update(qp['input_sha256'])
    for p in [source/'manifest.json',profilepath,qroot/'completion.json',qroot/'plan.json',training/'raw_discount_analysis.json']:
        inputs[io.relative(p)]=io.sha(p)
    plan=dict(source=io.relative(source),jobs=jobs,approved_candidate_signatures=approved,new_candidate_signatures=sorted(signatures),
        pure_warmup=old['pure_warmup']+candidates,neural_warmup=old['neural_warmup']+candidates,
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),games_per_trial=sum(j['n'] for j in jobs),scenarios=['ordinary2','split4'],
        endpoint_pairs=[dict(endpoint=candidates[2]['id'],original=originals[0]['id']),dict(endpoint=candidates[3]['id'],original=originals[1]['id'])],
        scope='80actual game executions;endpoint equality to original experts and ordinary/NumPy execution equivalence. Public seed/selection/reset/export qualification,not a speed benchmark,training or performance selection.170470000 is consumed qualification only.No GitHub upload.')
    io.verify(plan);io.write(out/'plan.json',plan);return plan


def endpoints(out):
    plan=io.read(out/'plan.json');io.verify(plan)
    count=0
    for scenario in plan['scenarios']:
        rows=io.read(out/scenario/'result.json')['results']
        for pair in plan['endpoint_pairs']:
            for a in [r for r in rows if r['own']==pair['endpoint']]:
                b=next(r for r in rows if r['own']==pair['original'] and r['foe']==a['foe'])
                for k in ['episodes','summary','damage_traces']:assert a.get(k)==b.get(k),(scenario,pair,k)
                count+=len(a['episodes'])
    io.write(out/'endpoint_reproduction.json',dict(status='exact_original_expert_reproduction',endpoint_games_checked=count,
        input_sha256={io.relative(p):io.sha(p) for p in [out/'plan.json',out/'completion.json',out/'ordinary2/result.json',out/'split4/result.json',Path(__file__).resolve()]}))
    print('episode mixture loader and endpoints verified',count,flush=True)


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();engine.freeze=freeze
    try:engine.run(a.source.resolve(),a.out.resolve());endpoints(a.out.resolve())
    except BaseException:io.write(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
