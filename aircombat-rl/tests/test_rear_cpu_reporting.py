import json
from pathlib import Path


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_report_keeps_reused_partial_and_diagnostic_counts_separate(tmp_path):
    from tools.rear_cpu_reporting import collect, render
    put(tmp_path/'plan.json', {'seeds': [42200, 42201], 'processes': 8})
    put(tmp_path/'started.json', {'time': 100})
    put(tmp_path/'training_progress.json', {'seed': 42200, 'additional_steps': 5242880, 'training_games': 5600})
    put(tmp_path/'s42200_result.json', {'additional_steps': 5242880, 'training_games': 5600, 'training_seconds': 600})
    for name, baseline, seed, games, reused, seconds in [
        ('base-monitor', True, 42100, 128, True, 0),
        ('base-development', True, 42100, 768, True, 0),
        ('new-monitor', False, 42200, 128, False, 62),
    ]:
        put(tmp_path/'evaluations'/name/'completion.json', {
            'baseline': baseline, 'seed': seed, 'kind': 'monitor',
            'additional_steps': 0 if baseline else 1310720,
            'summary': {'games': games, 'win_rate': .71, 'rear_damage': .1},
            'actual_new_games': 0 if reused else games,
            'reused_from_run': 'control' if reused else None,
            'reused_cells_in_final_attempt': games if reused else 0,
            'wall_seconds': seconds,
            'checkpoint_id': name,
            'win_gain': .01 if not baseline else None,
        })
    put(tmp_path/'queue'/'pending.json', {'job_id': 'pending', 'kind': 'development', 'checkpoint_id': 'next'})
    for i in range(4):
        put(tmp_path/'evaluations'/'pending'/'cells'/f'{i}.json', {})
    put(tmp_path/'diagnostics'/'replay'/'summary.json', {'diagnostic_games': 2, 'wall_seconds': 1.6})
    result = collect(tmp_path, now=1100)
    assert result['new_evaluation_games'] == 128
    assert result['reused_baseline_games'] == 896
    assert result['diagnostic_games'] == 2
    assert result['pending_jobs'][0]['completed_cells'] == 4
    assert result['overall_elapsed_seconds'] == 1000
    assert result['repeats'][0]['training_seconds'] == 600
    assert result['repeats'][0]['evaluation_seconds'] == 62
    assert result['repeats'][1]['additional_steps'] == 0
    assert result['monitor_results'][0]['win_gain_pp'] == 1
    rendered = render(result)
    assert '896' in rendered and 'Stage times overlap' in rendered
    assert 'replayed consumed conditions' in rendered
