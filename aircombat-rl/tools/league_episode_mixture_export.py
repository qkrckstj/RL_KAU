"""Export the previously fitted episode mixture, with synthetic state checks."""
from argparse import ArgumentParser
from pathlib import Path
import json,zipfile
import numpy as np
from tools import league_thread_benchmark as io
from experiments.league.episode_mixture_policy import Policy


def export(out):
    if out.exists():raise FileExistsError(out)
    fitpath=io.ROOT/'runs/mixture_feasibility_reproduced_20261008.json';fit=io.read(fitpath)
    source=io.ROOT/fit['source'];rawpath=source/'raw_expanded_comparison.json'
    assert fit['source_raw_sha256']==io.sha(rawpath)
    assert fit['weights']==dict(candidate=0.,repeat=0.,baseline=0.,teacher=.6,cem=.4)
    plan=io.read(source/'plan.json');io.verify(plan);raw=io.read(rawpath)
    for p,h in raw['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    specs=[plan['roles']['teacher'][0],plan['roles']['cem'][0]];configs=[]
    for s in specs:
        with zipfile.ZipFile(io.ROOT/s['weights']) as z:configs.append(json.loads(z.read('parameters.json')))
    teacher=io.ROOT/specs[0]['design']/'policy.py';cem=io.ROOT/specs[1]['design']/'policy.py'
    template=io.ROOT/'experiments/league/episode_mixture_policy.py'
    code=teacher.read_text('utf-8')+'\nTeacherPolicy=Policy\n'+cem.read_text('utf-8')+'\nCemPolicy=Policy\n'+template.read_text('utf-8')
    code=code.replace('from experiments.league.unseen_temporal_repair_20261007.temporal_extend_right.policy import Policy as TeacherPolicy\n','').replace('from experiments.league.extend_parameter_policy import Policy as CemPolicy\n','')
    namespace={};exec(compile(code,'bundled_mixture.py','exec'),namespace)
    variants=[];checks=0
    for name,probability,seed in [('sampled_4900',.6,4900),('sampled_4901',.6,4901),('teacher_only',1.,4900),('cem_only',0.,4900)]:
        cfg=dict(teacher=configs[0],cem=configs[1],teacher_probability=probability,action_seed=seed)
        a=Policy(configuration=cfg);b=namespace['Policy'](configuration=cfg);rng=np.random.default_rng(819)
        for ep in range(40):
            a.reset();b.reset();fixed=None
            refs=[namespace['TeacherPolicy'](configuration=configs[0]),namespace['CemPolicy'](configuration=configs[1])]
            for tick in range(60):
                obs=rng.normal(size=39).astype(np.float32);obs[[0,1,15,16]]*=1800.;obs[3:6]*=180.;obs[18:21]*=180.;obs[34]=float(tick%7==0);obs[38]=120.-tick*.05
                x,y=a.act(obs),b.act(obs);assert x==y and 0<=x<=8
                if fixed is None:fixed=b.selected
                assert b.selected==fixed and y==refs[fixed].act(obs)
                if probability in (0.,1.):assert fixed==int(probability==0.)
                if tick==0:first=obs.copy();first_action=y
                checks+=1
            assert b.act(first)==first_action
        folder=out/name;folder.mkdir(parents=True)
        (folder/'policy.py').write_text(code,encoding='utf-8');(folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
        with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:z.writestr('parameters.json',json.dumps(cfg,sort_keys=True))
        spec=dict(id='episode_mix_'+name,kind='submission',design=io.relative(folder),weights=io.relative(folder/'policy_net.zip'))
        io.write(folder/'entrant.json',spec);variants.append(dict(name=name,spec=spec,configuration=cfg))
    inputs=dict(plan['input_sha256'])
    for p in [fitpath,rawpath,source/'plan.json']+list(out.rglob('*')):
        if p.is_file():inputs[io.relative(p)]=io.sha(p)
    sources=dict(plan['source_sha256'])
    for p in [teacher,cem,template,Path(__file__).resolve()]:sources[io.relative(p)]=io.sha(p)
    manifest=dict(status='episode_mixture_exported',source=io.relative(source),original_experts=specs,variants=variants,synthetic_action_checks=checks,source_sha256=sources,input_sha256=inputs,
        scope='Frozen60/40 fit from consumed102M. Private per-episode choice based only on public first-frame words and preregistered action seed. Selected expert held for whole episode. Exact original expert code bundled. Endpoints1/0 are qualification only. No actual flight or fresh performance evidence yet;not neural training or promotion.')
    io.verify(manifest);io.write(out/'manifest.json',manifest);print('episode mixture exported',checks,'synthetic actions;no matches')


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();export(a.out.resolve())
