"""Freeze eight new opponents before their first outcomes are observed."""
from argparse import ArgumentParser
from pathlib import Path
import json, math, zipfile
from tools import league_thread_benchmark as io
from experiments.league.reactive_audit_policy import Policy


def freeze(out, training):
    if out.exists():
        raise FileExistsError(out)
    old = io.read(training/'plan.json');io.verify(old)
    code = io.ROOT/'experiments/league/reactive_audit_policy.py'
    signature = io.sha(code)
    assert all(s['kind'] != 'submission' or io.sha(io.ROOT/s['design']/'policy.py') != signature for s in old['opponents'])
    opponents=[]
    for mode in ['lead_lag','range_escape','brake_pursuit','threat_reversal']:
        for side,period,speed in [(-1,3.,400.),(1,9.,600.)]:
            identity=f'reactive_{mode}_{"left" if side<0 else "right"}'
            config=dict(mode=mode,side=side,period=period,speed=speed)
            policy=Policy(configuration=config)
            # Exercise direct time-reset handling and legal outputs on varied
            # finite public states, without running any policy-vs-policy games.
            observations=[]
            for i in range(500):
                obs=[0.]*39;obs[38]=120.-i*.2
                obs[15]=700.+(i%7)*600.;obs[16]=(-1)**i*1200.
                obs[3]=200.;obs[18]=-150.;obs[19]=80.
                obs[11]=math.sin(i)*math.pi;obs[26]=math.cos(i)*math.pi
                obs[34]=float(i%5==0)
                assert 0<=policy.act(obs)<=8
                observations.append(obs)
            first=policy.act(observations[0]);policy.reset()
            assert policy.act(observations[0])==first
            folder=out/identity;folder.mkdir(parents=True)
            (folder/'policy.py').write_bytes(code.read_bytes())
            (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
            with zipfile.ZipFile(folder/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:
                z.writestr('parameters.json',json.dumps(config,sort_keys=True))
            spec=dict(id=identity,kind='submission',design=io.relative(folder),weights=io.relative(folder/'policy_net.zip'))
            io.write(folder/'entrant.json',spec);io.write(folder/'parameters.json',config)
            opponents.append(spec)
    inputs={io.relative(p):io.sha(p) for p in out.rglob('*') if p.is_file()}
    inputs[io.relative(training/'plan.json')]=io.sha(training/'plan.json')
    panel=dict(opponents=opponents,training_source=io.relative(training),outcomes_observed=False,
        synthetic_legal_reset_checks=4000,source_sha256={io.relative(p):io.sha(p) for p in [code,Path(__file__).resolve()]},input_sha256=inputs,
        scope='Eight newly written untrained scripted variants, four related public-state heuristics. Not eight independent architectures, expert doctrine, actual student submissions or exhaustive opponents. No outcome-based selection. Once used for follow-up learning they cease to be heldout.')
    io.verify(panel);io.write(out/'panel.json',panel)
    print('reactive audit panel frozen',len(opponents),'no match outcomes observed')


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--training',type=Path,required=True);a=p.parse_args()
    freeze(a.out.resolve(),a.training.resolve())
