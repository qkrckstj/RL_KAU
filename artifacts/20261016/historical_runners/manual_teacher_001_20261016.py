"""Freeze one manual demonstration and evaluate the unchanged224-game panel."""
from pathlib import Path
import argparse,fcntl,json,multiprocessing,os,shutil,time,zipfile
import psutil
from experiments.rear_async.storage import read,write,sha,identity
from experiments.direct_pilot.manual_replay_policy import Policy
from experiments.direct_pilot.evaluate import cell
from tools.seven_model_league_v15 import verify_hashes,verify_run_hashes,paired_jobs,score_panel
from tools.policies import load
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/direct_pilot_20261016/manual_001_screen';MANUAL=OUT.parent/'manual_001'
FIELDS=('opponent','seed','seat','action_seed')
def prepare():
    assert not OUT.exists()
    config=read(ROOT/'runs/seven_league_range8000_20261016/config.json');verify_run_hashes(config)
    manifest=ROOT.parent/'artifacts/20261015/manifest.json';assert config['frozen_sha256'].get(str(manifest))==sha(manifest)
    files=dict(config['frozen_sha256']);files.update({str(ROOT.parent/k):v for k,v in read(manifest)['sha256'].items()});verify_hashes(files)
    completion=read(MANUAL/'completion.json');assert completion['status']=='terminal' and completion['result']=='win';verify_hashes(completion['input_sha256'])
    parameters=dict(commands=[dict(step=r['step'],action=r['action']) for r in map(json.loads,(MANUAL/'commands.jsonl').read_text().splitlines())]);Policy(parameters=parameters)
    OUT.mkdir();folder=OUT/'candidate';folder.mkdir()
    shutil.copyfile(ROOT/'experiments/direct_pilot/manual_replay_policy.py',folder/'policy.py')
    (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n')
    with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('parameters.json',json.dumps(dict(parameters=parameters)))
    write(folder/'policy_net.json',dict(kind='fixed_clock_replay_of_manual_actions_not_neural_or_feedback_generalization',observation_dim=39,actions=9,source=str(MANUAL),commands=parameters['commands']))
    own=dict(id='manual_001_time_replay_v1',kind='submission',design=str(folder),weights=str(folder/'policy_net.zip'));write(folder/'entrant.json',own)
    model_files={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='entrant.json'}
    act,_,mode=load(folder,folder/'policy_net.zip');assert mode=='discrete';count=0
    for line in (MANUAL/'trace.jsonl').read_text().splitlines():
        r=json.loads(line);assert act(r['raw_before'])==r['action'];count+=1
    assert count==completion['steps']==2401
    write(OUT/'LOADER_QUALIFICATION.json',dict(actual_manual_actions_exact=True,predictions=count,source_trace_sha256=sha(MANUAL/'trace.jsonl')))
    pool=config['initial_pool'];protocol=dict(stage='screen',opponents=[p['id'] for p in pool],seeds=list(range(config['screen_band'],config['screen_band']+8)),seats=['red','blue'],action_seeds=[4900,4901],model_id=identity(model_files),pool_id=identity(pool),smoke=False)
    jobs=list(paired_jobs({str(a):own for a in (4900,4901)},pool,protocol));assert len(jobs)==224
    for p in (Path(__file__).resolve(),ROOT/'experiments/direct_pilot/manual_replay_policy.py',ROOT/'experiments/direct_pilot/evaluate.py',OUT/'LOADER_QUALIFICATION.json',*folder.iterdir(),*(MANUAL/k for k in ('plan.json','commands.jsonl','trace.jsonl','completion.json'))):
        if p.is_file():files[str(p)]=sha(p)
    verify_hashes(files)
    write(OUT/'plan.json',dict(protocol=protocol,own=own,pool=pool,jobs=jobs,model_files=model_files,input_sha256=files,processes=3,new_neural_training_decisions=0,
        scope='Exactly onefixedcandidate and unchanged224screen. Publicclockreplay reproduces a manuallypiloted consumedepisode;it is not an adaptivefeedbackpolicy or a trained neuralteacher. No variable-size development grid.',
        own_action_rng='unused;deterministicpublic-clock schedule',manual_session=82614,manual_observed_exit=0))
    write(OUT/'PREPARED.json',dict(plan_sha256=sha(OUT/'plan.json'),games=224,ready=True));print('One manualcandidate frozen;224screen prepared;2401actions exact.',flush=True)
def job(item):
    own,foe,seed,seat,action=item
    return dict(cell(own,foe,seed,seat),action_seed=action)
def run():
    assert not (OUT/'runtime.json').exists(),'Inspect original process;never duplicate'
    assert read(OUT/'PREPARED.json')['plan_sha256']==sha(OUT/'plan.json')
    plan=read(OUT/'plan.json');verify_hashes(plan['input_sha256'])
    if psutil.virtual_memory().available<12*2**30:raise MemoryError('Need12GiB available')
    with (ROOT/'runs/seven_league_range8000_20261016/controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write(OUT/'runtime.json',dict(pid=os.getpid(),create_time=psutil.Process().create_time()));rows=[];started=time.monotonic()
        with multiprocessing.get_context('spawn').Pool(3) as workers:
            for row in workers.imap(job,plan['jobs']):
                write(OUT/'cells'/f"{identity({k:row[k] for k in FIELDS})}.json",row);rows.append(row)
                if len(rows)%16==0:write(OUT/'progress.json',dict(completed=len(rows),expected=224,elapsed_seconds=time.monotonic()-started))
        verify_hashes(plan['input_sha256'])
        manual_plan=read(MANUAL/'plan.json');ref=next(r for r in rows if (r['opponent'],r['seed'],r['seat'],r['action_seed'])==(manual_plan['foe']['id'],manual_plan['seed'],manual_plan['seat'],manual_plan['action_seed']))
        last=json.loads((MANUAL/'trace.jsonl').read_text().splitlines()[-1]);manual_complete=read(MANUAL/'completion.json')
        replay_match=(ref['result']==manual_complete['result'] and ref['steps']==manual_complete['steps'] and all(ref[k]==last['info'][k] for k in ('own_health','opp_health')) and ref['duration']==last['info']['t'])
        result=dict(protocol=plan['protocol'],score=score_panel(rows,plan['protocol']),elapsed_seconds=time.monotonic()-started,rows_sha256=identity(sorted(rows,key=lambda r:tuple(r[k] for k in FIELDS))),manual_reference_terminal_exact=replay_match,manual_reference_cell=ref,input_sha256=plan['input_sha256'],new_neural_training_decisions=0,admitted=False,fresh_confirmation_started=False)
        write(OUT/'completion.json',result);print(json.dumps(dict(score=result['score'],manual_reference_terminal_exact=replay_match,elapsed_seconds=result['elapsed_seconds']),indent=2),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);a=parser.parse_args();prepare() if a.command=='prepare' else run()
