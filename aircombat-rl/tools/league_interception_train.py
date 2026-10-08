"""Learn interceptor coefficients under a frozen gate; preserve all prior policies."""
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
import os
import shutil
import traceback
import numpy as np
from experiments.league.interception import LOW, HIGH
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_interception_gate_export import export
from tools.league_matches import initialize_worker
from tools.league_train import verify, unique_entrants, ordering
from tools.league_distinct_evaluation import evaluate_distinct_candidates as evaluate
from tools.league_cem_population import sample_population, distinct_elites
from tools.league_symmetric_train import immutable_json, comparison, development_pass, final_analysis, paired_ci
from tools.league_validate import process_live

INITIAL = np.array([15.,650.,650.,6.,4000.,.3,3.])


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def candidate(folder, plan, parameters, identity):
    config = dict(parent=plan['parent']['parameters'], interceptor=np.asarray(parameters).tolist())
    if (folder/'entrant.json').exists():
        if read(folder/'policy_net.json') != config:
            raise ValueError('Changed candidate parameters')
        for name,digest in read(folder/'artifact_sha256.json').items():
            if sha(folder/name) != digest:
                raise ValueError('Changed candidate artifact')
        spec = read(folder/'entrant.json')
        if spec['id'] != identity:
            raise ValueError('Changed candidate identity')
        return spec
    return export(folder, config, identity)


def freeze(out):
    if (out/'plan.json').exists():
        plan = read(out/'plan.json'); verify(plan); return plan
    previous = ROOT/'runs/league_symmetric_20261006'
    old = read(previous/'plan.json'); verify(old)
    if read(previous/'completion.json')['status'] != 'evaluation_failed':
        raise ValueError('Expected completed symmetric experiment')
    diagnostic = ROOT/'runs/league_interception_matched_20261006/completion.json'
    if read(diagnostic)['status'] != 'diagnostic_complete':
        raise ValueError('Matched diagnostic must complete')
    compatibility = ROOT/'runs/interception_gate_compatibility_20261006/completion.json'
    compat = read(compatibility)
    if compat['status'] != 'passed' or len(compat['episodes']) != 6:
        raise ValueError('Actual flight compatibility must pass')
    for name,digest in compat['source_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('Compatibility source changed')
    reference = read(previous/'frozen_selection.json')['chosen']
    sources = dict(old['source_sha256'])
    for name in ('experiments/league/interception.py','experiments/league/interception_gate.py',
                 'tools/league_interception_gate_export.py','tools/league_interception_train.py',
                 'tools/league_cem_population.py'):
        sources[name] = sha(ROOT/name)
    inputs = dict(old['input_sha256'])
    for path in (previous/'plan.json',previous/'completion.json',previous/'frozen_selection.json',diagnostic,compatibility):
        inputs[relative(path)] = sha(path)
    model = ROOT/reference['design']
    for name,digest in read(model/'artifact_sha256.json').items():
        if sha(model/name) != digest: raise ValueError('Reference changed')
        inputs[relative(model/name)] = digest
    plan = dict(parent=old['parent'], reference=reference,
        opponents=unique_entrants(old['opponents']+[reference]),
        seeds=[2500,2501,2502],workers=3,generations=4,population=12,elites=3,
        screen_n=4,confirmation_n=8,development_n=20,selection_n=40,final_n=160,
        training_band=150000000,development_band=37000000,selection_band=37010000,final_band=44000000,
        initial=INITIAL.tolist(),initial_std=.15,minimum_std=.04,
        source_sha256=sources,input_sha256=inputs,
        method='Seven interceptor coefficients; fixed .5-second symmetric gate and frozen parent. Three RNG searches share architecture, archive, parent and initialization.',
        training_rank='Existing weighted objective plus .02 * weighted clipped (1 - minimum_range/10000); tie-break original ordering. Dense distance term only ranks search, never changes official rewards/verdict or final criteria.',
        selection_rule='Existing parent-relative development_pass plus Evader gain >=.025 and no aggregate or per-opponent win regression versus preserved symmetric gate; highest candidate objective then smaller seed.',
        final_rule='Existing parent-relative final_analysis must pass; additionally Evader paired win-gain 95% CI lower >0 versus symmetric gate and no per-opponent win regression greater than .05 versus that gate.',
        scope='New independent conditions; band 43000000 consumed and excluded. No GitHub publication. No student-submission generalization claim.')
    immutable_json(out/'plan.json',plan)
    for name in sources:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
    return plan


def rank(record, weights):
    progress=float(np.dot(weights,[np.clip(1-r['summary']['min_range']/10000,0,1) for r in record['results']]))
    return (record['metrics']['objective']+.02*progress,*ordering(record['metrics']))


def search(out,plan,seed,ordinal,initial):
    if (out/'result.json').exists():return read(out/'result.json')
    rng=np.random.default_rng(seed)
    mean=(INITIAL-LOW)/(HIGH-LOW);incumbent=mean.copy();std=np.full(7,plan['initial_std'])
    normalize=lambda x:(x-LOW)/(HIGH-LOW)
    history=[]
    with ProcessPoolExecutor(max_workers=3,initializer=initialize_worker) as pool:
        base=evaluate(pool,[plan['parent'],plan['reference'],initial],plan['opponents'],plan['development_band'],plan['development_n'],out/'initial')
        weakness=np.array([1-r['summary']['rate'] for r in base[0]['results']])+.1
        weights=(.5/len(weakness)+.5*weakness/weakness.sum()).tolist()
        # Best checkpoint may remain the preserved reference; report that honestly.
        idx=max((1,2),key=lambda i:ordering(base[i]['metrics']))
        best=dict(spec=[plan['parent'],plan['reference'],initial][idx],metrics=base[idx]['metrics'],generation=-1)
        for generation in range(plan['generations']):
            verify(plan)
            population,anchors=sample_population(rng,mean,std,[incumbent,mean,normalize(INITIAL)],plan['population'])
            specs=[candidate(out/f'g{generation}/models/c{i}',plan,LOW+v*(HIGH-LOW),f'intercept_s{seed}_g{generation}_c{i}') for i,v in enumerate(population)]
            band=plan['training_band']+ordinal*10000+generation*100
            screen=evaluate(pool,specs,plan['opponents'],band,plan['screen_n'],out/f'g{generation}/screen',weights)
            order=sorted(range(len(specs)),key=lambda i:rank(screen[i],weights),reverse=True)
            indices=list(dict.fromkeys(distinct_elites(order,population,plan['elites'])+[anchors[0]]))
            confirmed=evaluate(pool,[specs[i] for i in indices],plan['opponents'],band+20,plan['confirmation_n'],out/f'g{generation}/confirm',weights)
            order=sorted(range(len(indices)),key=lambda i:rank(confirmed[i],weights),reverse=True)
            elites=distinct_elites([indices[i] for i in order],population,plan['elites'])
            incumbent=population[elites[0]].copy()
            mean=.5*mean+.5*population[elites].mean(axis=0)
            std=np.maximum(plan['minimum_std'],.5*std+.5*population[elites].std(axis=0))
            challenger=specs[elites[0]]
            checked=evaluate(pool,[challenger],plan['opponents'],plan['development_band'],plan['development_n'],out/f'g{generation}/development')[0]
            if ordering(checked['metrics'])>ordering(best['metrics']):
                best=dict(spec=challenger,metrics=checked['metrics'],generation=generation)
            state=dict(generation=generation,best=best,challenger=challenger,metrics=checked['metrics'],mean=mean.tolist(),std=std.tolist(),elite_indices=elites,anchor_indices=anchors,population=population.tolist(),rng=rng.bit_generator.state,training_games=sum(r['metrics']['games'] for r in screen+confirmed))
            immutable_json(out/f'g{generation}/state.json',state);history.append(state)
            write(out/'progress.json',dict(generation=generation+1,generations=plan['generations'],selected=best))
    result=dict(status='complete',seed=seed,selected=best,history=history,training_games=sum(x['training_games'] for x in history))
    write(out/'result.json',result);return result


def run(out):
    out=out.resolve()
    if (out/'runtime.json').exists() and process_live(read(out/'runtime.json')['pid']):
        raise ValueError('Training already running')
    plan=freeze(out)
    if (out/'completion.json').exists():return
    write(out/'runtime.json',dict(pid=os.getpid(),workers=3,started_at=datetime.now(timezone.utc).isoformat()))
    initial=candidate(out/'models/initial',plan,INITIAL,'intercept_initial')
    results=[]
    for ordinal,seed in enumerate(plan['seeds']):
        write(out/'progress.json',dict(stage='training',seed=seed,replication=ordinal+1,replications=3))
        results.append(search(out/f's{seed}',plan,seed,ordinal,initial))
    challengers=[r['selected']['spec'] for r in results]
    with ProcessPoolExecutor(max_workers=3,initializer=initialize_worker) as pool:
        write(out/'progress.json',dict(stage='selection'))
        records=evaluate(pool,[plan['parent'],plan['reference']]+challengers,plan['opponents'],plan['selection_band'],plan['selection_n'],out/'selection')
        comps=[comparison(r['results'],records[0]['results']) for r in records[2:]]
        refs=[comparison(r['results'],records[1]['results']) for r in records[2:]]
        eligible=[i for i,c in enumerate(comps) if development_pass(c) and refs[i]['per_opponent_win_gain']['evader']>=.025-1e-12 and refs[i]['win_gain']>=-1e-12 and refs[i]['worst_opponent_win_gain']>=-1e-12]
        choice=max(eligible,key=lambda i:(*ordering(comps[i]['candidate']),-plan['seeds'][i])) if eligible else None
        chosen=challengers[choice] if choice is not None else plan['reference']
        immutable_json(out/'frozen_selection.json',dict(chosen=chosen,chosen_index=choice,challengers=challengers,comparisons=comps,reference_comparisons=refs,eligible=eligible,plan_sha256=sha(out/'plan.json')))
        completion=dict(status='no_development_improvement',final_opened=False,parent_preserved=True)
        if choice is not None:
            immutable_json(out/'final_claim.json',dict(band=plan['final_band'],selection_sha256=sha(out/'frozen_selection.json')))
            write(out/'progress.json',dict(stage='final_evaluation',chosen=chosen['id']))
            final=evaluate(pool,[plan['parent'],plan['reference'],chosen],plan['opponents'],plan['final_band'],plan['final_n'],out/'final')
            cmp=final_analysis(final[2]['results'],final[0]['results'])
            ref=comparison(final[2]['results'],final[1]['results'])
            gain=paired_ci([r for r in final[2]['results'] if r['foe']=='evader'],[r for r in final[1]['results'] if r['foe']=='evader'])
            passed=cmp['passed'] and gain['ci95'][0]>0 and ref['worst_opponent_win_gain']>=-.05-1e-12
            completion=dict(status='evaluation_passed' if passed else 'evaluation_failed',selected=chosen,selected_before_test=True,final_opened=True,parent_preserved=True,comparison=cmp,reference_comparison=ref,evader_gain_vs_reference=gain)
    verify(plan);write(out/'completion.json',completion);write(out/'progress.json',dict(stage='complete',status=completion['status']))


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    try:run(args.out)
    except Exception:
        write(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
