"""Paired native JSBSim controller probes; no learning or global replacement."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time
from collections import Counter

from experiments.controller_probe.controllers import VARIANTS, make_controller
from experiments.rear_async.storage import read, write, sha, identity, verify

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'runs/aim_cpu_dealt0625_20261014'
PARENT = ROOT/'runs/aim_cpu_feature_train1e6_20261014/checkpoints/s47150_step_001966080'


def verify_inputs(plan):
    for name, digest in plan['inputs'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Changed experiment input: '+name)


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    source = read(SOURCE/'plan.json')
    ready = verify(PARENT)
    files = set(ROOT.glob('aircombat_gym/**/*.py'))
    files.update(ROOT.glob('experiments/controller_probe/*.py'))
    files.add(ROOT/'tests/test_controller_probe.py')
    files.update((SOURCE/'plan.json', PARENT/'READY.json'))
    for spec in [*source['development_opponents'], *ready['sampled_specs'].values()]:
        if spec['kind'] == 'submission':
            files.update(p for p in (ROOT/spec['design']).glob('*') if p.is_file())
            files.add(ROOT/spec['weights'])
    for pattern in ('experiments/league/*.py', 'experiments/aim_async/*.py'):
        files.update(ROOT.glob(pattern))
    files.update(ROOT/p for p in ('tools/policies.py', 'tools/league_matches.py'))
    baseline = {}
    for completion in (SOURCE/'evaluations').glob('*/completion.json'):
        item = read(completion)
        if item['baseline'] and item['kind'] == 'development':
            for path in (completion.parent/'cells').glob('*.json'):
                row = read(path)
                key = identity({k: row[k] for k in ('opponent', 'seed', 'seat', 'action_seed')})
                if key in baseline:
                    raise ValueError('Duplicate baseline fixture')
                baseline[key] = row
                files.add(path)
    if len(baseline) != 672 or sum(r['result']=='win' for r in baseline.values()) != 510:
        raise ValueError('Expected qualified original-controller 510/672 baseline')
    out.mkdir(parents=True)
    write(out/'baseline.json', baseline)
    files.add(out/'baseline.json')
    plan = dict(scope='Consumed development panels; existing PPO transfer only; no promotion or training.',
        variants=list(VARIANTS), speeds=[300., 450., 650.],
        flight_cases=['heading_step', 'policy_reversal', 'moving_aim'],
        own_specs=ready['sampled_specs'], opponents=source['monitor_opponents'],
        full_opponents=source['development_opponents'], band=source['development_band'],
        samplers=[4900, 4901], screen_ics=4, full_ics=8,
        installed_sides='learner only; opponent original. Same-controller-both-sides follow-up if promising.',
        input_model_sha256=sha(PARENT/'learner.zip'), training_steps=0,
        inputs={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)})
    write(out/'plan.json', plan)
    write(out/'PLAN.md', '# Isolated controller study\n\n'
        'Compare three heading/roll changes independently using identical native JSBSim initial states, '
        'fixed commands, and the preserved 510 PPO. Altitude, speed, lift budget, physics, actions, '
        'guns, termination and end-health classification remain unchanged.\n\n'
        'Run 36 flight traces (four controllers, three speeds, three tasks). Then run '
        'three 128-game matched screens (eight opponents, four initial seeds, two seats, two action samplers). '
        'Verify two original-controller games against the cached 510 baseline before using any cached result. '
        'Archive every candidate. If a safe candidate improves the screen, evaluate its full 672-game panel '
        'and assess whether the benefit remains when both aircraft use the candidate. No automatic installation.\n')
    return plan


def flight(variant, speed, case, output):
    from aircombat_gym.core.aircraft import Aircraft
    from aircombat_gym.core.control.autopilot import wrap_pi
    ac = Aircraft(autopilot=make_controller(variant))
    initial = ac.reset(psi=0., v_kt=speed)
    initial_word = {k: float(v).hex() for k, v in vars(initial).items()}
    frames = []
    start = time.perf_counter()
    duration = 35. if case == 'heading_step' else 30.
    for i in range(round(duration*20)):
        t = i*.05
        if case == 'heading_step':
            target = math.radians(90.)
            ac.psi_cmd = target
            ac.hold()
        elif case == 'policy_reversal':
            command = 30. if t<4. else -30. if t<8. else 0.
            ac.step(command)
            target = ac.psi_cmd
        else:
            target = math.radians(8.)*math.sin(2*math.pi*t/6.)
            ac.psi_cmd = target
            ac.hold()
        s = ac.state
        if not s.flyable:
            raise RuntimeError('Unflyable controller trace')
        frames.append(dict(t=(i+1)*.05, target_deg=math.degrees(target),
            heading_deg=math.degrees(s.psi), error_deg=math.degrees(wrap_pi(target-s.psi)),
            bank_deg=math.degrees(s.phi), p=s.p, speed=s.v_kt, altitude=s.h_ft,
            aileron=ac.last.aileron, elevator=ac.last.elevator,
            throttle=ac.last.throttle, bank_target_deg=math.degrees(ac.last.phi_cmd),
            bank_limit_deg=math.degrees(ac.last.max_bank)))
    write(output, dict(variant=variant, speed=speed, case=case, initial=initial_word, frames=frames))
    errors=[f['error_deg'] for f in frames]
    measured = frames[120:] if case=='moving_aim' else frames
    settled=None
    if case=='heading_step':
        for i in range(len(frames)-20):
            if all(abs(f['error_deg'])<=2. for f in frames[i:]):
                settled=frames[i]['t'];break
    return dict(variant=variant, speed=speed, case=case,
        rms_error_deg=math.sqrt(sum(f['error_deg']**2 for f in measured)/len(measured)),
        integrated_abs_error_deg_seconds=.05*sum(abs(e) for e in errors),
        step_settle_2deg_seconds=settled,
        step_overshoot_deg=max(0., -min(errors)) if case=='heading_step' else None,
        max_altitude_error_ft=max(abs(f['altitude']-20000.) for f in frames),
        max_speed_error_kt=max(abs(f['speed']-speed) for f in frames),
        final_heading_error_deg=errors[-1],
        aileron_saturation_fraction=sum(abs(f['aileron'])>=.999 for f in frames)/len(frames),
        wall_seconds=time.perf_counter()-start)


def native_cell(own, foe, seed, seat, variant, sides='own'):
    # Local class only: the official MatchEnv symbol and active workers are untouched.
    from tools.league_matches import MatchEnv
    from experiments.aim_async.evaluate import cell
    import experiments.aim_async.evaluate as metric_module
    class ProbeEnv(MatchEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            installed = ('red', 'blue') if sides=='both' else (self.seat,)
            for side in installed:
                old=self._combat.ac[side].autopilot
                self._combat.ac[side].autopilot=make_controller(variant,
                    h0_ft=old.h0_ft, throttle_cap=old.throttle_cap)
    # The metric routine has no env injection argument. Substitution is scoped
    # to this isolated, single-threaded process and restored even on exceptions.
    previous=metric_module.MatchEnv
    try:
        metric_module.MatchEnv=ProbeEnv
        return cell(own, foe, seed, seat)
    finally:
        metric_module.MatchEnv=previous


def jobs(plan, variant, full=False, sides='own'):
    foes=plan['full_opponents'] if full else plan['opponents']
    n=plan['full_ics'] if full else plan['screen_ics']
    return [dict(variant=variant, sides=sides, opponent=f, seed=plan['band']+i,
                 seat=seat, action_seed=a) for f in foes for i in range(n)
            for seat in ('red','blue') for a in plan['samplers']]


def game_key(job):
    return identity(dict(opponent=job['opponent']['id'], seed=job['seed'],
                         seat=job['seat'], action_seed=job['action_seed']))


def panel(out, plan, variant, baseline, full=False, sides='own'):
    from experiments.aim_async.evaluate import profile
    rows=[];base=[];started=time.perf_counter();new=0
    for job in jobs(plan,variant,full,sides):
        key=game_key(job)
        path=out/'cells'/variant/sides/(key+'.json')
        if path.exists():
            row=read(path)
        else:
            row=dict(native_cell(plan['own_specs'][str(job['action_seed'])], job['opponent'],
                job['seed'], job['seat'], variant, sides), action_seed=job['action_seed'],
                variant=variant, sides=sides)
            write(path,row);new+=1
        rows.append(row);base.append(baseline[key])
        if len(rows)%16==0:
            print(variant,sides,'full' if full else 'screen',len(rows),flush=True)
            write(out/'progress.json',dict(phase='combat',variant=variant,sides=sides,
                full=full,completed=len(rows),wall_seconds=time.perf_counter()-started))
    summary=profile(rows);bs=profile(base)
    pairs=Counter(b['result']+'->'+r['result'] for b,r in zip(base,rows))
    per={f['id']:dict(candidate=profile([r for r in rows if r['opponent']==f['id']]),
                      original=profile([b for b in base if b['opponent']==f['id']]))
         for f in (plan['full_opponents'] if full else plan['opponents'])}
    result=dict(summary=summary,original=bs,win_gain=summary['wins']-bs['wins'],
        hp_gain=(summary['damage_dealt']-summary['damage_received'])-(bs['damage_dealt']-bs['damage_received']),
        transitions=dict(pairs),per_opponent=per,new_games=new,
        wall_seconds=time.perf_counter()-started,variant=variant,sides=sides,full=full)
    write(out/f'{variant}_{sides}_{"full" if full else "screen"}.json',result)
    return result


def run(out, plan):
    import psutil
    if psutil.virtual_memory().available < 10*2**30:
        raise MemoryError('Need 10 GiB free for isolated diagnostic')
    for proc in psutil.process_iter(['cmdline']):
        args=proc.info['cmdline'] or []
        if any(Path(a).name=='aim_cpu_runner.py' for a in args) and any(a in args for a in ('launch','evaluate')):
            raise RuntimeError('Learning still active; do not contend with existing experiment')
    verify_inputs(plan)
    baseline=read(out/'baseline.json');start=time.perf_counter()
    with (out/'runtime.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'completion.json').exists():
            raise RuntimeError('Already complete; inspect stored results')
        write(out/'runtime.json',dict(pid=os.getpid(),status='running'))
        checks=[]
        for job in jobs(plan,'original')[:2]:
            row=native_cell(plan['own_specs'][str(job['action_seed'])],job['opponent'],job['seed'],job['seat'],'original')
            reference=baseline[game_key(job)]
            differences={k:[v,reference.get(k)] for k,v in row.items() if k!='wall_seconds' and v!=reference.get(k)}
            if differences:
                raise ValueError('Cached original mismatch: '+str(differences))
            checks.append(dict(key=game_key(job),exact_metrics=True))
        write(out/'original_reproduction.json',dict(new_games=2,checks=checks))
        print('Original reproduction exact on two games',flush=True)
        traces=[];initials={}
        for variant in VARIANTS:
            for speed in plan['speeds']:
                for case in plan['flight_cases']:
                    path=out/'flight_traces'/f'{variant}_{int(speed)}_{case}.json'
                    traces.append(flight(variant,speed,case,path))
                    initial=read(path)['initial'];key=(speed,case)
                    if key in initials and initial!=initials[key]:
                        raise ValueError('Flight initial state differs')
                    initials[key]=initial
                    print('Flight',variant,speed,case,flush=True)
        write(out/'flight_results.json',dict(traces=traces,initial_states_exact=True))
        screens={v:panel(out,plan,v,baseline) for v in VARIANTS[1:]}
        best=max(screens,key=lambda v:(screens[v]['win_gain'],screens[v]['hp_gain']))
        full={};both={}
        safe=all(r['max_altitude_error_ft']<150. for r in traces if r['variant']==best)
        if screens[best]['win_gain']>0 and safe:
            full[best]=panel(out,plan,best,baseline,full=True)
            both[best]=panel(out,plan,best,baseline,sides='both')
        verify_inputs(plan);verify(PARENT)
        result=dict(status='controller_probes_complete',screens=screens,full=full,both=both,
            flight_results='flight_results.json',new_combat_games=2+sum(s['new_games'] for s in [*screens.values(),*full.values(),*both.values()]),
            flight_traces=len(traces),training_steps=0,automatic_installation=False,
            best_screen=best,full_gate_safe=safe,wall_seconds=time.perf_counter()-start,
            scope=plan['scope'],original_sources_unchanged=True)
        write(out/'completion.json',result)
        write(out/'runtime.json',dict(pid=os.getpid(),status='complete'))
        print(json.dumps({k:v for k,v in result.items() if k not in ('screens','full','both')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();out=args.out.resolve()
    if not out.is_relative_to(ROOT/'runs'):
        raise ValueError('Diagnostic output must be under runs')
    plan=read(out/'plan.json') if (out/'plan.json').exists() else prepare(out)
    run(out,plan)
