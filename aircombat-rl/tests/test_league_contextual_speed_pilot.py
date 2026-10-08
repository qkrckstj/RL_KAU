import json
import subprocess
import sys
from copy import deepcopy
import pytest
from tools.league_contextual_speed_pilot import eligible_status,comparable,CONTEXT_GRID


def test_waiting_imports_no_numpy_or_torch():
    result=subprocess.run([sys.executable,'-c',
        "import sys,json;import tools.league_contextual_speed_pilot;print(json.dumps([x for x in ('numpy','torch','jsbsim') if x in sys.modules]))"],
        capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==[]


def test_only_completed_failed_searches_are_eligible_and_success_retains_model():
    assert eligible_status('no_development_improvement')
    assert eligible_status('repair_profile_failed')
    assert not eligible_status('repair_profile_passed')
    for value in ('running','crashed','not_eligible','evaluation_passed'):
        with pytest.raises(ValueError):eligible_status(value)
    assert len(CONTEXT_GRID)==len(set(CONTEXT_GRID))==8


def test_loader_parity_checks_full_episode_content_beyond_wins():
    original=dict(own='base',foe='foe',band=7,summary=dict(wins=1),
        episodes=[dict(seed=7,seat='red',own_health=.4,steps=123)],elapsed_seconds=1.)
    same=deepcopy(original);same.update(own='disabled_copy',elapsed_seconds=2.)
    assert comparable(original)==comparable(same)
    changed=deepcopy(same);changed['episodes'][0]['steps']=124
    assert comparable(original)!=comparable(changed)
    changed=deepcopy(same);changed['episodes'][0]['own_health']=.5
    assert comparable(original)!=comparable(changed)
