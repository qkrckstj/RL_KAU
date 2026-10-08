"""Frozen compact48 continuation from preserved PPO7400; archive205 retained."""
from argparse import ArgumentParser
from pathlib import Path
import copy
import os
import shutil
from tools import league_thread_benchmark as io
from tools import league_prioritized_ppo_train as runner
from tools import league_rate_ppo_train as rate
from tools.league_compact_roster import select
from tools.league_tournament_metrics import code_groups, group_weights

PREVIOUS = None


def freeze(source, out, smoke, qualification):
    assert not out.exists()
    assert all(os.environ.get(k) == '1' for k in io.THREAD_KEYS)
    assert not io.process_live(io.read(source/'runtime.json')['pid'])
    assert min(io.resources()[k] for k in ('commit_headroom_gib', 'available_memory_gib')) >= 6
    old = io.read(source/'plan.json'); io.verify(old)
    transition = io.read(source/'user_requested_transition.json')
    assert (source/'refreshed/s7600/chunk_01.json').exists()
    previous = io.read(PREVIOUS) if PREVIOUS else None
    assert previous is not None
    control=io.ROOT/'runs/league_compact_ppo_train_20261008'
    assert not io.process_live(io.read(control/'runtime.json')['pid'])
    cp=io.read(control/'plan.json');io.verify(cp)
    cr=io.read(control/'raw_compact_analysis.json')
    assert cr['status']=='compact_ppo_raw_verified' and cr['additional_steps']==2621440
    screen=io.ROOT/'runs/league_compact_inactive_assess_20261008'
    assert not io.process_live(io.read(screen/'runtime.json')['pid'])
    sr=io.read(screen/'raw_budget_screen.json')
    assert sr['status']=='budget_screen_raw_verified'
    for record in (cr,sr):
        for name,digest in record['input_sha256'].items():assert io.sha(io.ROOT/name)==digest
    foes, roster = select(old['opponents'], old['opponent_win_estimates'], previous)
    assert len(roster['rotation'])==8
    assert len(set(roster['selected_ids'])&set(previous['selected_ids']))==40
    assert old['warm_start']==cp['warm_start']
    groups = code_groups(foes); balanced = group_weights(groups)
    estimates = {s['id']: old['opponent_win_estimates'][s['id']] for s in foes}
    weak = {k: .05+1-v['win_estimate'] for k,v in estimates.items()}
    total = sum(weak.values()); weak = {k:v/total for k,v in weak.items()}
    probs = [.4/48+.3*balanced[s['id']]+.3*weak[s['id']] for s in foes]
    assert min(probs)>0 and abs(sum(probs)-1)<1e-12
    plan = copy.deepcopy(old)
    arm = dict(old['arms'][0], name='rotated', probabilities=probs)
    plan.update(source=io.relative(source), smoke=smoke, paired_roster_control='runs/league_compact_ppo_train_20261008', full_retained_archive=old['opponents'],
        roster_manifest=roster, opponents=foes, groups=groups, opponent_win_estimates=estimates,
        weakness=weak, probabilities=probs, arms=[arm],
        seeds=[7997] if smoke else [7800,7801],
        training_bands=[170600000] if smoke else [304000000,305000000],
        development_band=170610000 if smoke else old['development_band'],
        development_n=2 if smoke else old['development_n'],
        development_opponents=[s for s in old['development_opponents'] if s['id'] in ['ace','evader','untrained_parameter_01']] if smoke else old['development_opponents'],
        chunk_steps=40960 if smoke else 1310720, base_chunks=1, maximum_chunks=1,
        cached_baseline={}, matched_control_run=None,
        quality_scope='Matched7800/7801 continuation RNGs from identical PPO7400 actor/critic/Adam; reuse completed compact controls with same304/305M streams. Rotate8 opponents within code groups; compare this curriculum treatment,not independent evidence.48active foes,205archive,27code groups,cap4/group,verified fixed-RNG aliases collapsed. Same PPO/reward/teacher/8physical32virtual envs. Not an isolated opponent-count experiment or independent from-scratch replication.',
        budget_rule='First stage:1,310,720new steps per seed;2,621,440total. No automatic extra chunk in this frozen run. Analyze both repeats and broad-panel regressions before a new continuation experiment.',
        experiment_scope='Intentionally reuse304/305M training paired with completed compact controls; separate paired reservations preserve originals. Reuse111M development and its original40foes for comparability with interrupted205 run; it is consumed development,not heldout evidence.109M/110M select active roster. No policy promotion or GitHub upload.')
    assert len(plan['development_opponents']) == (3 if smoke else 40)
    sources = dict(old['source_sha256']); inputs = dict(old['input_sha256'])
    for name in ['league_compact_roster.py','league_compact_ppo_train.py','league_compact_ppo_analysis.py','league_compact_rotated_train_v2.py']:
        f=io.ROOT/'tools'/name; sources[io.relative(f)]=io.sha(f)
    extra=[source/'plan.json',source/'user_requested_transition.json',source/'refreshed/s7600/chunk_01.json']
    extra += [control/'plan.json',control/'completion.json',control/'raw_compact_analysis.json',
              screen/'plan.json',screen/'completion.json',screen/'raw_budget_screen.json']
    if PREVIOUS: extra.append(PREVIOUS)
    if not smoke:
        qp=io.read(qualification/'plan.json'); io.verify(qp)
        qr=io.read(qualification/'raw_compact_analysis.json')
        assert qr['status']=='compact_ppo_raw_verified'
        assert qp['source_sha256']==sources and qp['warm_start']==plan['warm_start']
        assert qp['opponents']==foes and qp['roster_manifest']==roster
        for name,digest in qr['input_sha256'].items(): assert io.sha(io.ROOT/name)==digest
        extra += [qualification/'plan.json',qualification/'completion.json',qualification/'raw_compact_analysis.json']
        claims=[io.ROOT/f'runs/paired_training_compact_rotated_{b}.json' for b in plan['training_bands']]
        paired=io.ROOT/'runs/paired_development_compact_rotated_111000000.json'
        assert not any(f.exists() for f in claims+[paired])
        original=io.ROOT/'runs/sampled_assessment_reservation_111000000.json'
        assert io.read(original)['run']==io.relative(source)
        extra.append(original)
        for f,b in zip(claims,plan['training_bands']):
            original=io.ROOT/f'runs/training_reservation_{b}.json'
            assert io.read(original)['run']==io.relative(control)
            extra.append(original)
            io.write(f,dict(run=io.relative(out),start=b,stop_exclusive=b+1000000,
                           original_reservation=io.relative(original),paired_control=io.relative(control)))
        io.write(paired,dict(run=io.relative(out),original=io.relative(original),scope='Reused consumed development,not fresh validation'))
    for f in extra: inputs[io.relative(f)]=io.sha(f)
    plan.update(source_sha256=sources,input_sha256=inputs,environment=io.environment())
    io.verify(plan); io.write(out/'plan.json',plan); io.write(out/'active_roster.json',roster)
    for name in sources:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(io.ROOT/name,dest)
    return plan


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/league_refreshed_ppo_train_20261008'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--previous-roster',type=Path,default=Path('runs/league_compact_ppo_train_20261008/active_roster.json'))
    p.add_argument('--qualification',type=Path,default=Path('runs/league_compact_rotated_smoke_v2_20261008'))
    a=p.parse_args();PREVIOUS=a.previous_roster.resolve() if a.previous_roster else None
    runner.freeze=freeze;runner.audited_load=rate.rate_load;runner.extend=lambda *args:False
    runner.run(a.source.resolve(),a.out.resolve(),a.smoke,a.qualification.resolve())
    from tools.league_compact_ppo_analysis import analyze
    r=analyze(a.out.resolve());print(r['status'],r['additional_steps'],flush=True)


