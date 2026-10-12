"""Cache qualification without replaying consumed native simulation games."""
import copy
import shutil

import pytest

from experiments.rear_async.evaluate import profile
from experiments.rear_async.storage import publish, write


def panel(tmp_path):
    checkpoint = tmp_path / 'checkpoint'
    staging = tmp_path / 'staging'
    staging.mkdir()
    (staging / 'learner.zip').write_bytes(b'artifact identity; never loaded')
    (staging / 'sampled_actor').write_bytes(b'actor A')
    publish(staging, checkpoint, dict(checkpoint_id='selected', additional_steps=10))
    folder = tmp_path / 'evaluation'
    shutil.copytree(checkpoint, folder / 'input')
    plan = dict(monitor_band=100, monitor_ics=1,
                monitor_opponents=[dict(id='foe')])
    rows = [dict(opponent='foe', seed=100, seat=seat, action_seed=4900,
                 result='win', rear_damage=.6, rear_wez_seconds=2.,
                 max_rear_wez_streak=2., damage_dealt=1., damage_received=0.,
                 head_on_damage=0., clean_rear_win=True,
                 rear_before_first_received=True) for seat in ('red', 'blue')]
    for i, row in enumerate(rows):
        write(folder / 'cells' / f'{i}.json', row)
    entry = dict(checkpoint_id='selected', kind='monitor', summary=profile(rows),
                 per_opponent={'foe': profile(rows)})
    write(folder / 'completion.json', entry)
    return plan, checkpoint, folder, entry


def test_selected_parent_cache_is_relabelled_without_new_games(tmp_path):
    from tools.prepare_rear_cpu_selected_parent import validate_panel, reused_baseline
    plan, checkpoint, folder, entry = panel(tmp_path)
    assert validate_panel(plan, checkpoint, folder) == entry
    original = copy.deepcopy(entry)
    entry.update(seed=1, additional_steps=10, baseline=False, wall_seconds=7.,
                 actual_new_games=2, win_gain=.1)
    reused = reused_baseline(entry, folder, tmp_path, seed=2)
    assert reused['baseline'] and reused['additional_steps'] == 0
    assert reused['actual_new_games'] == 0 and reused['wall_seconds'] == 0
    assert reused['source_additional_steps'] == 10 and reused['win_gain'] is None
    assert reused['seed'] == 2 and reused['summary'] == original['summary']
    assert entry['baseline'] is False


@pytest.mark.parametrize('corruption', ['cell', 'actor', 'summary'])
def test_cache_from_changed_panel_or_export_is_rejected(tmp_path, corruption):
    from tools.prepare_rear_cpu_selected_parent import validate_panel
    from experiments.rear_async.storage import read, sha
    plan, checkpoint, folder, entry = panel(tmp_path)
    if corruption == 'cell':
        row = read(folder / 'cells' / '0.json')
        row['action_seed'] = 4901
        write(folder / 'cells' / '0.json', row)
    elif corruption == 'summary':
        entry['summary']['rear_damage'] += .1
        write(folder / 'completion.json', entry)
    else:
        # A different, internally valid export must still fail identity check.
        actor = folder / 'input' / 'sampled_actor'
        actor.write_bytes(b'actor B')
        ready = read(folder / 'input' / 'READY.json')
        ready['files']['sampled_actor'] = sha(actor)
        write(folder / 'input' / 'READY.json', ready)
    with pytest.raises(ValueError):
        validate_panel(plan, checkpoint, folder)
