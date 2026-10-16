"""Both-direction native response screening before the learning comparison."""
import argparse
import math
from pathlib import Path
import statistics
import time
from aircombat_gym.core.aircraft import Aircraft
from aircombat_gym.core.control.autopilot import wrap_pi
from experiments.controller_probe.learnability import heading_rate
from experiments.controller_study.controllers import EffortController
from experiments.rear_async.storage import write,sha

SETTINGS={'effort':{},'feedback02':{'feedback':.2},'feedback05':{'feedback':.5},
          'slew30':{'slew_deg':30.},'slew60':{'slew_deg':60.}}


def flight(settings,speed,kind,sign,level):
    ac=Aircraft(autopilot=EffortController(**settings));initial=ac.reset(v_kt=speed)
    frames=[];duration=20.
    for i in range(round(duration*20)):
        t=i*.05
        if kind=='release':u=sign if t<4. else 0.
        elif kind=='reversal':u=sign if t<4. else -sign if t<8. else 0.
        elif kind=='sinusoid':u=.4*math.sin(2*math.pi*t/8.)
        else:u=level
        ac.autopilot.effort=u;ac.step(0.);s=ac.state
        if not s.flyable:raise RuntimeError('Unflyable response screen')
        rate=math.degrees(heading_rate(s));nominal=math.degrees(9.80665*math.tan(ac.last.max_bank)/s.v)
        frames.append(dict(t=(i+1)*.05,u=u,heading=s.psi,bank=s.phi,rate=rate,
            requested_rate=u*nominal,altitude=s.h_ft,speed=s.v_kt,aileron=ac.last.aileron))
    release_at=4. if kind=='release' else 8.
    index=round(release_at*20)-1
    stopped=None;reverse=0.;movement=0.
    if kind in ('release','reversal'):
        direction=sign if kind=='release' else -sign
        changes=[direction*math.degrees(wrap_pi(f['heading']-frames[index]['heading'])) for f in frames[index+1:]]
        reverse=max(0.,-min(changes))
        movement=sum(abs(math.degrees(wrap_pi(b['heading']-a['heading']))) for a,b in zip(frames[index:],frames[index+1:]))
        for j in range(index+1,len(frames)-20):
            if all(abs(f['rate'])<.5 for f in frames[j:]):stopped=frames[j]['t']-release_at;break
    tail=frames[-80:]
    return dict(initial={k:float(v).hex() for k,v in vars(initial).items()},frames=frames,
        metrics=dict(max_reverse_deg=reverse,stop_seconds=stopped,total_post_release_movement_deg=movement,
            rate_rmse_deg=math.sqrt(statistics.mean((f['rate']-f['requested_rate'])**2 for f in frames)),
            final_mean_rate_deg=statistics.mean(f['rate'] for f in tail),
            max_altitude_error_ft=max(abs(f['altitude']-20000.) for f in frames),
            max_speed_loss_kt=max(speed-f['speed'] for f in frames)))


def main(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);start=time.perf_counter();initials={};summaries={}
    root=Path(__file__).resolve().parents[2]
    sources=list((root/'aircombat_gym').rglob('*.py'))+list((root/'experiments/controller_study').glob('*.py'))+[
        root/'experiments/controller_probe/learnability.py',root/'experiments/rear_async/storage.py']
    frozen={str(p.relative_to(root)):sha(p) for p in sources};write(out/'inputs.json',frozen)
    for name,settings in SETTINGS.items():
        rows=[]
        for speed in (300.,450.,650.):
            cases=[(k,s,0.) for k in ('release','reversal') for s in (-1.,1.)]
            cases += [('sinusoid',1.,0.)]+[('constant',1.,u) for u in (.25,.5,1.)]
            for kind,sign,level in cases:
                trial=flight(settings,speed,kind,sign,level)
                if speed in initials and trial['initial']!=initials[speed]:raise ValueError('Initial states differ')
                initials[speed]=trial['initial']
                row=dict(controller=name,speed=speed,kind=kind,sign=sign,level=level,**trial['metrics'])
                rows.append(row);write(out/'traces'/f'{name}_{int(speed)}_{kind}_{sign}_{level}.json',dict(row=row,**trial))
        release=[r for r in rows if r['kind']=='release']
        sine=[r for r in rows if r['kind']=='sinusoid']
        rev=statistics.mean(r['max_reverse_deg'] for r in release)
        stop=statistics.mean(16. if r['stop_seconds'] is None else r['stop_seconds'] for r in release)
        rms=statistics.mean(r['rate_rmse_deg'] for r in sine)
        safe=all(r['max_altitude_error_ft']<150. for r in rows)
        summaries[name]=dict(settings=settings,rows=rows,mean_release_reverse_deg=rev,
            mean_release_stop_seconds=stop,sinusoid_rate_rmse_deg=rms,safe=safe,
            score=rev+.5*stop+.1*rms)
        print(name,'reverse',rev,'stop',stop,'rate_rms',rms,'safe',safe,flush=True)
    eligible=[name for name,s in summaries.items() if name!='effort' and s['safe']]
    if not eligible:raise RuntimeError('No safe stabilizer; retain all response evidence')
    selected=min(eligible,key=lambda n:summaries[n]['score'])
    for name,digest in frozen.items():
        if sha(root/name)!=digest:raise RuntimeError('Response input changed: '+name)
    write(out/'results.json',dict(status='complete',summaries=summaries,selected=selected,
        selected_settings=summaries[selected]['settings'],
        selected_improves_score=summaries[selected]['score']<summaries['effort']['score'],
        rule='Among nonbaseline candidates with all altitude errors<150ft: minimize mean release reverse degrees +0.5*mean settling seconds +0.1*sinusoidal rate RMSE. Exploratory tradeoff, not a global optimum.',
        traces=sum(len(s['rows']) for s in summaries.values()),training_steps=0,combat_games=0,
        wall_seconds=time.perf_counter()-start,initial_states_exact=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out.resolve())
