"""Wait for league development, independently refine, then open a frozen test.

Waiting is tied to the launched controller PID, not just an old progress file.
Failed gates request further analysis and do not declare the overall goal done.
"""
from argparse import ArgumentParser
from pathlib import Path
import ctypes
import json
import os
import shutil
import time
import numpy as np
from tools.autolab_cem import ROOT,read,write,sha
from tools.league_matches import evaluate_jobs,initial_roster
from tools.league_train import metrics,search,export,unique_entrants,verify


def process_live(pid):
    if os.name != 'nt':
        try:
            os.kill(pid,0)
            return True
        except ProcessLookupError:
            return False
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.argtypes=[ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
    kernel.OpenProcess.restype=ctypes.c_void_p
    kernel.GetExitCodeProcess.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes=[ctypes.c_void_p]
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:
        if ctypes.get_last_error()==87:
            return False
        raise OSError(ctypes.get_last_error(),'Cannot inspect controller process')
    try:
        code=ctypes.c_ulong()
        if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):
            raise ctypes.WinError(ctypes.get_last_error())
        return code.value==259
    finally:
        kernel.CloseHandle(handle)


def screen_gate(candidate,original,head):
    c,o=metrics(candidate),metrics(original)
    h=head['summary']
    passed=(c['mean_win_rate']>=o['mean_win_rate']+.08 and
            c['mean_score']>=o['mean_score']+.03 and c['worst_score']>=.35 and
            h['score']>=.55 and h['rate']>=.2)
    return dict(passed=bool(passed),candidate=c,original=o,head_to_head=h,
                rule='mean win +.08, mean score +.03, worst score >=.35, head score >=.55 and head wins >=.2')


def rank_pool_candidates(records):
    """All entrants face the same pool; no veto by one previous champion.

    Exploitability is still represented by lower-quarter performance and the
    subsequent minimum-score gate. Head-to-head raw outcomes remain reported.
    """
    return sorted(records,key=lambda r:(r['metrics']['objective'],r['metrics']['mean_win_rate'],
                                        r['metrics']['health']),reverse=True)


def choose_pool_parent(out,history,opponents,original,config):
    candidates=unique_entrants([original]+[r['challenger'] for r in history])
    jobs=[dict(own=a,foe=b,band=config['screen_band'],n=config['pool_screen_n'])
          for a in candidates for b in opponents]
    screened=evaluate_jobs(jobs,out/'pool_screen')
    n=len(opponents)
    ranking=rank_pool_candidates([dict(spec=s,metrics=metrics(screened[i*n:(i+1)*n]))
                                  for i,s in enumerate(candidates) if s['id']!=original['id']])
    write(out/'pool_screen_ranking.json',ranking)
    if not ranking:
        return original,dict(passed=False,reason='No distinct trained challenger was found')
    finalists=[r['spec'] for r in ranking[:2]]
    jobs=[dict(own=a,foe=b,band=config['confirmation_band'],n=config['screen_n'])
          for a in finalists+[original] for b in opponents]
    confirmed=evaluate_jobs(jobs,out/'pool_confirmation')
    original_rows=confirmed[len(finalists)*n:]
    original_index=next(i for i,p in enumerate(opponents) if p['id']==original['id'])
    records=[]
    for i,spec in enumerate(finalists):
        rows=confirmed[i*n:(i+1)*n]
        gate=screen_gate(rows,original_rows,rows[original_index])
        records.append(dict(spec=spec,metrics=metrics(rows),gate=gate))
    # Select only using this new development panel, before independent runs
    # and before any sealed test. If both fail, retain failure for diagnosis.
    eligible=[r for r in records if r['gate']['passed']]
    chosen=rank_pool_candidates(eligible or records)[0]
    write(out/'pool_selection.json',dict(candidates=candidates,confirmation=records,chosen=chosen,
          purpose='Development selection across all archived round models, not final-test selection'))
    return chosen['spec'],chosen['gate']


def heldout_analysis(results,selected_ids,original_id,transfer_ids):
    by={(r['own'],r['foe']):r for r in results}
    foes=sorted({r['foe'] for r in results})
    if len(by)!=len(results):
        raise ValueError('Duplicate held-out matchup')
    def vector(identity,foe):
        return {(e['seed'],e['seat']):e for e in by[identity,foe]['episodes']}
    keys=sorted(vector(original_id,foes[0]))
    seeds=sorted({k[0] for k in keys})
    if len(keys)!=2*len(seeds):
        raise ValueError('Held-out observations are not complete paired seats')
    differences=[]
    for seed in seeds:
        wins=[]
        for identity in selected_ids:
            for foe in foes:
                own=vector(identity,foe)
                base=vector(original_id,foe)
                if set(own)!=set(keys) or set(base)!=set(keys):
                    raise ValueError('Held-out opponents must share ICs for paired analysis')
                for seat in ('red','blue'):
                    wins.append(float(own[seed,seat]['won'])-float(base[seed,seat]['won']))
        differences.append(float(np.mean(wins)))
    rng=np.random.default_rng(719)
    differences=np.asarray(differences)
    boot=differences[rng.integers(len(seeds),size=(10000,len(seeds)))].mean(axis=1)
    ci=np.quantile(boot,[.025,.975]).tolist()
    original=metrics([by[original_id,f] for f in foes])
    original_transfer=metrics([by[original_id,f] for f in transfer_ids])
    records=[]
    for identity in selected_ids:
        all_metrics=metrics([by[identity,f] for f in foes])
        transfer=metrics([by[identity,f] for f in transfer_ids])
        records.append(dict(id=identity,metrics=all_metrics,transfer=transfer,
                            against_original=by[identity,original_id]['summary']))
    improvements=sum(r['metrics']['mean_win_rate']>original['mean_win_rate'] for r in records)
    mean_win=float(np.mean([r['metrics']['mean_win_rate'] for r in records]))
    mean_score=float(np.mean([r['metrics']['mean_score'] for r in records]))
    transfer_score=float(np.mean([r['transfer']['mean_score'] for r in records]))
    head_score=float(np.mean([r['against_original']['score'] for r in records]))
    head_win=float(np.mean([r['against_original']['rate'] for r in records]))
    worst=min(r['metrics']['worst_score'] for r in records)
    passed=(ci[0]>0 and mean_win>=original['mean_win_rate']+.08 and
            mean_score>=original['mean_score']+.03 and improvements>=2 and worst>=.35 and
            transfer_score>=original_transfer['mean_score']-.05 and head_score>=.55 and head_win>=.2)
    return dict(passed=bool(passed),original=original,original_transfer=original_transfer,records=records,
        selected_mean_win_rate=mean_win,selected_mean_score=mean_score,mean_transfer_score=transfer_score,
        head_to_head_score=head_score,head_to_head_win_rate=head_win,worst_opponent_score=worst,
        mean_win_gain=float(differences.mean()),paired_ic_bootstrap_95_ci=ci,improved_repeats=improvements,
        shared_ic_seeds=len(seeds),opponents=len(foes),matches_per_policy=len(keys)*len(foes),
        scope='Three independent refinement RNG streams share one learned league parent and one frozen opponent archive. '
              'CI conditions on these policies/opponents and resamples IC clusters across all opponents and both seats. '
              'Excluded local opponents are not real student submissions; no universal-win claim.')


def run(main,out,pid,all_candidates=False):
    out.mkdir(parents=True,exist_ok=True)
    config=dict(main=str(main),screen_band=32000000,screen_n=40,
        all_candidates=all_candidates,pool_screen_n=12,confirmation_band=32000500,
        replication_seeds=[2200,2201,2202],replication_training_band=120000000,
        replication_development_band=32001000,replication_development_n=20,
        test_band=40000000,test_n=80,transfer_opponents=['lead','circler','ddqn_s0'],
        source_sha256=sha(Path(__file__)),
        selection='Highest development objective, then win rate, then health, then smaller seed; never by final test',
        exclusions='No access to actual competitors; hold-out is only the named local policies',
        heldout_opponent_scope='Full archived training pool plus excluded opponents' if all_candidates else 'Original roster plus excluded opponents')
    if (out/'plan.json').exists() and read(out/'plan.json')!=config:
        raise ValueError('Follow-up configuration changed')
    write(out/'plan.json',config)
    shutil.copyfile(Path(__file__),out/'league_validate.snapshot.py')
    while not (main/'completion.json').exists():
        if (main/'failure.json').exists():
            raise RuntimeError(f'Main run failed: {read(main/"failure.json")}')
        if not process_live(pid):
            # A final atomic completion write may race a process-exit poll.
            time.sleep(1)
            if not (main/'completion.json').exists():
                raise RuntimeError('Main controller exited without completion; resume its existing output')
            break
        write(out/'progress.json',dict(stage='waiting_for_development',controller_pid=pid,checked_at=time.time()))
        time.sleep(15)
    main_plan=read(main/'plan.json')
    verify(main_plan)
    completion=read(main/'completion.json')
    if completion['status']!='development_batch_complete':
        raise ValueError('Unexpected main completion')
    champion=completion['champion']
    history=read(main/'history.json')
    roster=main_plan['roster']
    opponents=unique_entrants(roster+[r['challenger'] for r in history])
    original=roster[0]
    write(out/'progress.json',dict(stage='development_screen',champion=champion['id']))
    if all_candidates:
        champion,gate=choose_pool_parent(out,history,opponents,original,config)
    else:
        jobs=[dict(own=a,foe=b,band=config['screen_band'],n=config['screen_n'])
              for a in (champion,original) for b in opponents]
        jobs.append(dict(own=champion,foe=original,band=config['screen_band'],n=config['screen_n']))
        screened=evaluate_jobs(jobs,out/'screen')
        size=len(opponents)
        gate=screen_gate(screened[:size],screened[size:2*size],screened[-1])
    write(out/'screen_gate.json',gate)
    if not gate['passed']:
        write(out/'completion.json',dict(status='analysis_required',stage='screen',gate=gate,
                                        next='Analyze weak opponents and run another development batch; sealed test not opened'))
        return
    repeats=[]
    settings=dict(main_plan['search'])
    settings.update(development_n=config['replication_development_n'],initial_std=.09,minimum_std=.025)
    for i,seed in enumerate(config['replication_seeds']):
        verify(main_plan)
        write(out/'progress.json',dict(stage='independent_refinement',seed=seed,index=i+1,total=3))
        result=search(out/f'replication/s{seed}',champion,opponents,seed,
            config['replication_training_band']+i*10000,config['replication_development_band'],settings)
        repeats.append(result)
    # Freeze all selected policies and the decision before any test is opened.
    selected=[dict(r['selected'],id=f'refine_{r["seed"]}') for r in repeats]
    chosen=max(range(len(repeats)),key=lambda i:(repeats[i]['selected_metrics']['objective'],
         repeats[i]['selected_metrics']['mean_win_rate'],repeats[i]['selected_metrics']['health'],-repeats[i]['seed']))
    heldout_roster=(opponents if all_candidates else roster)+[
        dict(id='lead',kind='bot',name='lead'),dict(id='circler',kind='bot',name='circler'),
        dict(id='ddqn_s0',kind='submission',design='runs/plan_a_20261005/double/ddqn/s0',
             weights='runs/plan_a_20261005/double/ddqn/s0/policy_net.zip')]
    chosen_spec=selected[chosen]
    transfer_inputs={heldout_roster[-1]['weights'],*[f"{heldout_roster[-1]['design']}/{name}"
                    for name in ('policy.py','wrappers.py','utils.py')]}
    freeze=dict(selected=selected,chosen=chosen_spec,opponents=heldout_roster,plan=config,
        repetitions=[dict(seed=r['seed'],selected_generation=r['selected_generation'],
             selected_metrics=r['selected_metrics'],final_metrics=r['final_metrics']) for r in repeats],
        transfer_input_sha256={p:sha(ROOT/p) for p in sorted(transfer_inputs)})
    write(out/'frozen_selection.json',freeze)
    export(out/'chosen',chosen_spec)
    claim=ROOT/'runs/league_20261006/heldout_40000000.claim.json'
    if claim.exists():
        if read(claim)['owner']!=str(out):
            raise ValueError('Held-out IC band already claimed by another experiment')
    else:
        with claim.open('x',encoding='utf-8') as f:
            json.dump(dict(owner=str(out),band=config['test_band'],n=config['test_n'],selection_sha256=sha(out/'frozen_selection.json')),f,indent=2)
    if read(claim)['selection_sha256']!=sha(out/'frozen_selection.json'):
        raise ValueError('Selection changed after held-out band was claimed')
    write(out/'progress.json',dict(stage='sealed_evaluation',selected=chosen_spec['id']))
    jobs=[dict(own=a,foe=b,band=config['test_band'],n=config['test_n'])
          for a in [original]+selected for b in heldout_roster]
    results=evaluate_jobs(jobs,out/'heldout')
    verify(main_plan)
    for path,digest in freeze['transfer_input_sha256'].items():
        if sha(ROOT/path)!=digest:
            raise ValueError(f'Transfer opponent changed during test: {path}')
    verdict=heldout_analysis(results,[s['id'] for s in selected],original['id'],config['transfer_opponents'])
    write(out/'verdict.json',verdict)
    # The independently refined models also face each other. This is diagnostic
    # test data only and cannot replace the policy already frozen by development.
    peer_jobs=[dict(own=a,foe=b,band=config['test_band'],n=config['test_n'])
               for i,a in enumerate(selected) for b in selected[i+1:]]
    peers=evaluate_jobs(peer_jobs,out/'refinement_crossplay')
    write(out/'refinement_crossplay.json',peers)
    write(out/'completion.json',dict(status='evaluation_passed' if verdict['passed'] else 'analysis_required',
          stage='heldout',chosen=chosen_spec,verdict=verdict,
          next='Package and audit results' if verdict['passed'] else 'This test is consumed; diagnose and allocate a fresh future test before further training'))
    write(out/'progress.json',dict(stage='evaluation_passed' if verdict['passed'] else 'analysis_required'))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--main',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--pid',required=True,type=int)
    p.add_argument('--all-candidates',action='store_true')
    a=p.parse_args()
    try:
        run(a.main.resolve(),a.out.resolve(),a.pid,a.all_candidates)
    except Exception as e:
        write(a.out/'failure.json',dict(error=repr(e)))
        raise
