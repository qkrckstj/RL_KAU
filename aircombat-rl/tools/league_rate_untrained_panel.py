"""Freeze eight untrained tactical parameter combinations without match outcomes."""
from pathlib import Path
from datetime import datetime, timezone
import json, zipfile
import numpy as np
from tools import league_thread_benchmark as io
from tools.league_extend_parameter_train import export
from tools.league_matches import Actor
from tools.league_numpy_matches import numpy_capable
from experiments.league.extend_parameter_policy import Policy,NAMES,LOW,HIGH


def run():
    out=io.ROOT/'experiments/league/rate_untrained_parameters_20261008'
    if out.exists(): raise FileExistsError(out)
    source=io.ROOT/'runs/league_rate_ppo_train_20261008'
    plan=io.read(source/'plan.json');io.verify(plan)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    assert io.read(source/'raw_rate_analysis.json')['status']=='rate_ppo_raw_verified'
    qualification=io.ROOT/'runs/league_episode_mixture_qualification_20261008'
    signatures=io.read(qualification/'completion.json')['approved_signatures']
    rng=np.random.default_rng(7501)
    unit=np.column_stack([(rng.permutation(8)+rng.random(8))/8 for _ in NAMES])
    parameters=np.asarray(LOW)+(np.asarray(HIGH)-np.asarray(LOW))*unit
    prior=set()
    for s in plan['opponents']:
        if s['kind']=='submission' and zipfile.is_zipfile(io.ROOT/s['weights']):
            with zipfile.ZipFile(io.ROOT/s['weights']) as z:
                if 'parameters.json' in z.namelist():
                    c=json.loads(z.read('parameters.json'))
                    if set(c)==set(NAMES):prior.add(tuple(c[k] for k in NAMES))
    foes=[];checks=0;inputs={};sources={}
    for i,values in enumerate(parameters):
        assert tuple(values) not in prior
        spec=export(out/f'variant_{i:02d}',values,f'untrained_parameter_{i:02d}')
        assert numpy_capable(spec,signatures), 'Require already-qualified unchanged policy family'
        direct=Policy(configuration=dict(zip(NAMES,values)));bundled=Actor(spec)
        assert bundled.mode=='discrete'
        for j in range(256):
            if j%64==0:direct.reset();bundled.reset()
            obs=np.zeros(39,dtype=np.float32)
            obs[3:6]=[0.,rng.uniform(180,330),0.]
            obs[11]=rng.uniform(-np.pi,np.pi);obs[14]=rng.uniform(-.5,.5)
            bearing=rng.uniform(-np.pi,np.pi);distance=rng.uniform(500,5000)
            obs[15:17]=[distance*np.sin(bearing),distance*np.cos(bearing)]
            obs[18:20]=rng.uniform(-300,300,2);obs[30:32]=rng.uniform(.1,1,2)
            obs[34]=j%2;obs[38]=120.-(j%64)*.5
            a=direct.act(obs);b=int(bundled.predict(obs));assert a==b and 0<=a<9;checks+=1
        foes.append(spec)
        for p in list((io.ROOT/spec['design']).glob('*.py'))+[io.ROOT/spec['weights']]:inputs[io.relative(p)]=io.sha(p)
    for p in [Path(__file__).resolve(),io.ROOT/'tools/league_extend_parameter_train.py',io.ROOT/'experiments/league/extend_parameter_policy.py']:
        sources[io.relative(p)]=io.sha(p)
    panel=dict(created_at=datetime.now(timezone.utc).isoformat(),opponents=foes,parameters=parameters.tolist(),
        parameter_names=list(NAMES),generator_seed=7501,generation='Eight fixed stratified random combinations across existing seven-parameter bounds; no outcome-based selection.',
        source=io.relative(source),outcomes_observed=False,synthetic_action_checks=checks,source_sha256=sources,input_sha256=inputs,
        scope='Not present in the173 training opponents. New parameter combinations of a known controller family,not new architectures or actual student entries. Untrained evidence only until this assessment; then consumed.')
    io.verify(panel);io.write(out/'panel.json',panel);print('untrained panel frozen',len(foes),'synthetic checks',checks)


if __name__=='__main__':run()
