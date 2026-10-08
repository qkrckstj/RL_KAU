"""Freeze new switching opponents without observing match outcomes."""
from argparse import ArgumentParser
from pathlib import Path
import json,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from experiments.league.switching_audit_policy import Policy


def freeze(out,source):
    if out.exists():raise FileExistsError(out)
    plan=io.read(source/'plan.json');io.verify(plan)
    raw=io.read(source/'raw_parameter_analysis.json');assert raw['status']=='extend_parameter_raw_verified'
    for p,h in raw['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    specs=[raw['baseline']['spec']]+[r['spec'] for r in sorted(raw['searches'],key=lambda r:r['seed'])]
    configs=[]
    for s in specs:
        with zipfile.ZipFile(io.ROOT/s['weights']) as z:configs.append(json.loads(z.read('parameters.json')))
    base=io.ROOT/'experiments/league/extend_parameter_policy.py';template=io.ROOT/'experiments/league/switching_audit_policy.py'
    code=base.read_text('utf-8')+'\nExpertPolicy=Policy\n'+template.read_text('utf-8').replace('from experiments.league.extend_parameter_policy import Policy as ExpertPolicy\n','')
    cases=[('periodic_fast','periodic',4.,.5,0.,[0,1]),('periodic_slow','periodic',12.,1.,6.,[2,1]),
        ('range_close','distance',1800.,1.,0.,[0,1]),('range_wide','distance',3000.,2.,0.,[1,0]),
        ('turn_sensitive','turn',.12,.5,0.,[2,0]),('turn_strong','turn',.25,1.,0.,[0,1]),
        ('threat_close','threat',2000.,.5,0.,[1,0]),('threat_wide','threat',3500.,2.,0.,[2,1])]
    opponents=[]
    for name,mode,threshold,dwell,offset,indices in cases:
        config=dict(experts=[configs[i] for i in indices],mode=mode,threshold=threshold,dwell=dwell,offset=offset)
        original=Policy(configuration=config);space={};exec(compile(code,'bundled_policy.py','exec'),space);bundled=space['Policy'](configuration=config)
        rng=np.random.default_rng(861);observations=[]
        for i in range(2400):
            obs=rng.normal(size=39);obs[[0,1,15,16]]*=1800.;obs[3:6]*=180.;obs[18:21]*=180.;obs[34]=float(i%7==0);obs[38]=120.-(i%1200)*.1
            a,b=original.act(obs),bundled.act(obs);assert a==b and 0<=a<=8;observations.append(obs)
        a=bundled.act(observations[0]);bundled.reset();assert a==bundled.act(observations[0])
        folder=out/name;folder.mkdir(parents=True)
        (folder/'policy.py').write_text(code,encoding='utf-8');(folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
        with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('parameters.json',json.dumps(config,sort_keys=True))
        spec=dict(id='switching_'+name,kind='submission',design=io.relative(folder),weights=io.relative(folder/'policy_net.zip'))
        io.write(folder/'entrant.json',spec);io.write(folder/'parameters.json',config);opponents.append(spec)
    inputs={io.relative(p):io.sha(p) for p in out.rglob('*') if p.is_file()}
    inputs[io.relative(source/'raw_parameter_analysis.json')]=io.sha(source/'raw_parameter_analysis.json')
    panel=dict(source=io.relative(source),opponents=opponents,experts=specs,outcomes_observed=False,
        synthetic_bundling_reset_checks=19200,source_sha256={io.relative(p):io.sha(p) for p in [base,template,Path(__file__).resolve()]},input_sha256=inputs,
        scope='Eight untrained switching rules over three previously exposed frozen controllers. New compositions/rules,not independent architectures or wholly unseen expert components. Public observations and time only; all expert states warmed on actual prefix. No match outcomes observed at freeze; no opponent selection/tuning from candidate performance. Future use for learning consumes this panel.')
    io.verify(panel);io.write(out/'panel.json',panel);print('switching panel frozen,8opponents,19200synthetic checks,no matches')


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args();freeze(a.out.resolve(),a.source.resolve())
