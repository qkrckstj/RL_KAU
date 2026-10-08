"""Apply the pre-outcome learning-rate comparison protocol to a completed run."""
from argparse import ArgumentParser
from pathlib import Path
from tools import league_thread_benchmark as io


def analyze(out):
    plan = io.read(out/'plan.json'); io.verify(plan)
    raw = io.read(out/'raw_rate_analysis.json')
    done = io.read(out/'completion.json')
    protocol = io.read(out/'comparison_protocol.json')
    assert raw['status'] == 'rate_ppo_raw_verified' and not plan['smoke']
    assert protocol['plan_sha256'] == io.sha(out/'plan.json')
    assert protocol['status'] == 'frozen_before_first_checkpoint_outcome'
    assert raw['additional_steps'] == done['additional_steps'] == 10485760
    for path, digest in raw['input_sha256'].items():
        assert io.sha(io.ROOT/path) == digest
    paths = [out/f for f in ('plan.json','raw_rate_analysis.json','completion.json','comparison_protocol.json')]
    paths.append(Path(__file__).resolve())

    def point(candidate):
        folder = (io.ROOT/candidate['learner']).parent
        files = ([out/f'control/baseline_action_{a}.json' for a in plan['action_replicas']]
                 if candidate['step'] == 0 else [folder/f'development_{a}.json' for a in plan['action_replicas']])
        profiles = [io.read(f)['profile'] for f in files]; paths.extend(files)
        win = sum(p['mean_win_rate'] for p in profiles)/len(profiles)
        group = sum(p['group_balanced_win_rate'] for p in profiles)/len(profiles)
        lower = min(p['lower_quarter_score'] for p in profiles)
        worst = min(p['worst_score'] for p in profiles)
        assert abs(.5*(win+group)-candidate['rank'][0]) < 1e-12
        assert lower == candidate['rank'][1] and worst == candidate['rank'][2]
        return dict(step=candidate['step'],win=win,group_win=group,lower_quarter=lower,worst=worst,learner=candidate['learner'])

    initial = point(done['baseline']); branches = {}
    for r in done['searches']:
        assert not r['extended'] and len(r['history']) == 2
        assert r['history'][1]['steps'] == 2*plan['chunk_steps'] == 2621440
        branches[(r['arm'],r['seed'])] = dict(first=point(r['history'][0]['candidate']),
            final=point(r['history'][1]['candidate']),retained=point(r['selected']),
            episodes=r['training_episodes'],opponent_episode_counts=r['opponent_episode_counts'])
    assert set(branches) == {(arm,s) for arm in ('control','low_rate') for s in plan['seeds']}
    minimum_gain = protocol['extension_screen']['minimum_win_gain']
    assert minimum_gain == .005 and protocol['extension_screen']['require_each_seed']
    rows = []
    for seed in plan['seeds']:
        control,low = branches[('control',seed)],branches[('low_rate',seed)]
        checks = dict(win_vs_source=low['final']['win']-initial['win'] >= minimum_gain-1e-12,
            win_vs_control=low['final']['win']-control['final']['win'] >= minimum_gain-1e-12,
            lower_quarter=low['final']['lower_quarter'] >= max(initial['lower_quarter'],control['final']['lower_quarter'])-1e-12,
            worst=low['final']['worst'] >= max(initial['worst'],control['final']['worst'])-1e-12)
        rows.append(dict(seed=seed,control=control,low_rate=low,
            final_win_gain_low_minus_control=low['final']['win']-control['final']['win'],
            final_win_gain_low_minus_source=low['final']['win']-initial['win'],
            extension_checks=checks,extension_screen_pass=all(checks.values())))
    means = {arm:{which:sum(r[arm][which]['win'] for r in rows)/len(rows)
                  for which in ('first','final','retained')} for arm in ('control','low_rate')}
    result = dict(status='matched_rate_comparison_verified',initial=initial,paired_seeds=rows,mean_win=means,
        extension_screen_pass=all(r['extension_screen_pass'] for r in rows),
        policy_promoted=False,scope=protocol['scope'],
        interpretation='Equal additional budgets and identical retention rule. Method extension screen and individual promising checkpoint selection are different decisions. No confidence over independent training initializations is inferred from two shared-ancestor continuations.',
        input_sha256={io.relative(f):io.sha(f) for f in paths})
    io.write(out/'matched_rate_analysis.json',result)
    return result


if __name__ == '__main__':
    p=ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();r=analyze(a.out.resolve())
    print(r['status'],r['mean_win'],'extension_screen_pass',r['extension_screen_pass'],flush=True)
