"""Portability and unchanged evaluation contract, without new native episodes."""
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_fixed_panel_uses_both_seats_and_both_preserved_action_exports():
    from tools.cpu_handoff import jobs_and_protocol
    jobs, panel = jobs_and_protocol(ROOT, ROOT/'artifacts/20261016/direct/feedback_002/candidate')
    assert len(jobs) == 224
    assert panel['seeds'] == list(range(3401500000, 3401500008))
    assert panel['seats'] == ['red', 'blue']
    assert panel['action_seeds'] == [4900, 4901]
    assert len(panel['opponents']) == 7
    assert len({(foe['id'], seed, seat, action) for own, foe, seed, seat, action in jobs}) == 224
    for own, foe, _, _, action in jobs:
        assert Path(own['weights']).is_file() and Path(foe['weights']).is_file()
        assert f'sampled_{action}' in Path(foe['weights']).parts
        assert Path(own['weights']).is_relative_to(ROOT)


def test_existing_strict_scorer_rejects_all_draws_any_loss_and_incomplete_panel():
    from tools.cpu_handoff import jobs_and_protocol, score_panel
    jobs, panel = jobs_and_protocol(ROOT, ROOT/'artifacts/20261016/direct/feedback_002/candidate')
    rows = [dict(opponent=foe['id'], seed=seed, seat=seat, action_seed=action,
                 result='draw', own_health=0., opp_health=0.) for _,foe,seed,seat,action in jobs]
    assert score_panel(rows, panel)['opponent_wins'] == 0
    rows[0].update(result='win', own_health=.5)
    assert score_panel(rows, panel)['opponent_wins'] == 1
    rows[1].update(result='loss', own_health=0., opp_health=.5)
    assert score_panel(rows, panel)['opponent_wins'] == 0
    with pytest.raises(ValueError): score_panel(rows[:-1], panel)


def test_existing_output_is_rejected_before_running_any_match(tmp_path):
    from tools.cpu_handoff import evaluate
    with pytest.raises(FileExistsError):
        evaluate(ROOT, ROOT/'artifacts/20261016/direct/feedback_002/candidate', tmp_path, 1, False)


def test_candidate_identity_includes_additional_local_helper(tmp_path):
    import shutil
    from tools.cpu_handoff import jobs_and_protocol
    candidate = tmp_path/'candidate'
    shutil.copytree(ROOT/'artifacts/20261016/direct/feedback_002/candidate',candidate)
    helper = candidate/'utils.py'
    helper.write_text('value = 1\n')
    _, before = jobs_and_protocol(ROOT,candidate)
    helper.write_text('value = 2\n')
    _, after = jobs_and_protocol(ROOT,candidate)
    assert before['model_id'] != after['model_id']
