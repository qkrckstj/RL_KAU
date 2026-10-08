"""Fit six defensive-break parameters over a frozen preserved CEM pilot."""
from argparse import ArgumentParser
from pathlib import Path
import json,os,shutil,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools import league_extend_parameter_train as engine
from tools.league_tournament_metrics import code_groups
from tools.league_residual_train import development_panel
from experiments.league.risk_break_policy import Policy,BASE_EXPERT,RISK_NAMES,RISK_LOW,RISK_HIGH,RISK_BASELINE

original_qualify=engine.qualify


def qualify(job):
    return original_qualify(dict(job,band=170490000))


def export(folder,parameters,identity):
    cfg=dict(zip(RISK_NAMES,map(float,parameters)));a=Policy(configuration=cfg)
    base=io.ROOT/'experiments/league/extend_parameter_policy.py';template=io.ROOT/'experiments/league/risk_break_policy.py'
    code=base.read_text('utf-8')+'\nBasePolicy=Policy\n'+template.read_text('utf-8').replace('from experiments.league.extend_parameter_policy import Policy as BasePolicy\n','')
    space={};exec(compile(code,'bundled_risk.py','exec'),space);b=space['Policy'](configuration=cfg)
    rng=np.random.default_rng(7219)
    for tick in range(120):
        x=rng.normal(size=39).astype(np.float32);x[[0,1,15,16]]*=1800.;x[3:6]*=180.;x[18:21]*=180.;x[30:32]=rng.uniform(0,1,2);x[32:34]=rng.uniform(0,3,2);x[34:36]=rng.integers(0,2,2);x[38]=120.-(tick%60)*.05
        assert a.act(x)==b.act(x)
    folder.mkdir(parents=True,exist_ok=False);(folder/'policy.py').write_text(code,encoding='utf-8')
    (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
    with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('parameters.json',json.dumps(cfg,sort_keys=True))
    identity=identity.replace('extend_parameter_baseline','risk_break_baseline').replace('extend_cem_','risk_break_cem_')
    spec=dict(id=identity,kind='submission',design=io.relative(folder),weights=io.relative(folder/'policy_net.zip'))
    io.write(folder/'entrant.json',spec);io.write(folder/'parameters.json',cfg);return spec


def freeze(source,out):
    if out.exists():raise FileExistsError(out)
    if any(os.environ.get(k)!='1' for k in io.THREAD_KEYS):raise ValueError('Thread limits required')
    if io.process_live(io.read(source/'runtime.json')['pid']):raise RuntimeError('Screen still live')
    old=io.read(source/'plan.json');io.verify(old);raw=io.read(source/'raw_mixture_screen.json')
    assert raw['status']=='mixture_screen_raw_verified'
    for p,h in raw['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    teacher=old['roles']['cem'][0]
    with zipfile.ZipFile(io.ROOT/teacher['weights']) as z:assert json.loads(z.read('parameters.json'))==BASE_EXPERT
    foes=old['full_retained_archive']+old['roles']['candidate'];assert len(foes)==len({s['id'] for s in foes})==156
    groups=code_groups(foes);scores=raw['analysis']['roles']['cem']['opponents']
    weak=sorted(scores,key=lambda k:((scores[k]['wins']+.5*scores[k]['draws'])/scores[k]['games'],k))[:8]
    required=weak+['temporal_extend_left','temporal_delayed_left','novel_opening_right']+[s['id'] for s in old['roles']['ppo_new']+old['roles']['ppo_repeat']]
    training=development_panel(foes,groups,required,teacher['id'],size=32)
    dev=development_panel(foes,groups,required+[s['id'] for s in foes if s['id'].startswith('switching_')],teacher['id'],size=48)
    sources=dict(old['source_sha256']);inputs=dict(old['input_sha256'])
    for p in [Path(__file__).resolve(),Path(engine.__file__).resolve(),io.ROOT/'tools/league_risk_break_analysis.py',io.ROOT/'experiments/league/risk_break_policy.py',io.ROOT/'tests/test_risk_break_policy.py']:sources[io.relative(p)]=io.sha(p)
    checks=io.ROOT/'runs/risk_break_checks_20261008.xml'
    import xml.etree.ElementTree as ET
    suites=list(ET.parse(checks).iter('testsuite'));assert suites and all(int(s.get('failures','0'))==int(s.get('errors','0'))==0 for s in suites) and sum(int(s.get('tests','0')) for s in suites)>=6
    for p in [source/'plan.json',source/'completion.json',source/'raw_mixture_screen.json',checks]:inputs[io.relative(p)]=io.sha(p)
    plan=dict(source=io.relative(source),teacher=teacher,opponents=foes,groups=groups,training_opponents=training,development_opponents=dev,
        weakness_ids=weak,seeds=[7200,7201],training_bands=[296000000,297000000],development_band=105000000,
        training_n=4,development_n=8,workers=8,population=10,generations=4,elites=3,
        parameter_names=list(RISK_NAMES),low=list(RISK_LOW),high=list(RISK_HIGH),baseline_parameters=list(RISK_BASELINE),initial_std=.3,min_std=.04,
        source_sha256=sources,input_sha256=inputs,environment=io.environment(),base_expert=BASE_EXPERT,qualification_band=170490000,
        scope='Two independent CEM search RNGs/condition sets for6public-state defensive-break parameters over frozenCEM6800. Not PPO or independent neural initialization. Both starts reproduce original CEM when break duration0.156archived policies preserved;stratified training/development panels. Allows a learned break while own weapon solution exists;only action choice changes. Official physics/actions/ICs/weapon/verdict unchanged.104Mscreen consumed;fresh105Mdevelopment.No promotion/final/GitHub claim.',
        selection_rule='Same inherited CEM objective=.75*(uniform+codegroup win)/2+.25lower-quarter score,then tail/worst/losses.Each generation includes baseline and incumbent;exact parameter duplicates cached per search.Each nominee fixed by training before shared development. Retain all older policies;evaluate actual transfer before adopting risk gate.')
    claims=[io.ROOT/f'runs/training_reservation_{b}.json' for b in plan['training_bands']]+[io.ROOT/f'runs/sampled_assessment_reservation_{plan["development_band"]}.json']
    if any(p.exists() for p in claims):raise FileExistsError('Conditions reserved')
    io.verify(plan)
    for p,b,n in zip(claims,plan['training_bands']+[plan['development_band']],[4,4,8]):io.write(p,dict(run=io.relative(out),start=b,stop_exclusive=b+n//2))
    io.write(out/'plan.json',plan)
    for name in sources:
        p=out/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,p)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    engine.freeze=freeze;engine.export=export;engine.qualify=qualify
    engine.LOW=RISK_LOW;engine.HIGH=RISK_HIGH;engine.BASELINE=RISK_BASELINE
    engine.run(a.source.resolve(),a.out.resolve())
    from tools.league_risk_break_analysis import analyze
    r=analyze(a.out.resolve());print(r['status'],r['games'],flush=True)
