import json
import subprocess
import sys
import pytest

from tools.league_residual_repair_followup import select_omitted,eligible_status


def row(seed,step,identity,rank,passed=True,digest=None):
    return dict(seed=seed,step=step,spec=dict(id=identity),rank=[rank,.3,0.,-3],
                repair_passed=passed,weights_sha256=digest or identity)


def test_keeps_eligible_nonleader_and_uses_only_development_records():
    records=[row(3300,2,'winner',.8,False),row(3300,4,'repair',.7),row(3300,6,'later',.69),
             row(3301,2,'second_main',.9),row(3301,4,'second_repair',.75)]
    selected=select_omitted(records,{'winner','second_main'})
    assert [r['spec']['id'] for r in selected]==['repair','second_repair']
    assert len(selected)==2


def test_deduplicates_artifacts_breaks_ties_and_skips_ineligible():
    records=[row(3300,4,'late',.7,digest='same'),row(3300,2,'early',.7,digest='same'),
             row(3301,2,'duplicate',.8,digest='same'),row(3301,4,'other',.6),
             row(3302,2,'failed_gate',1.,False)]
    assert [r['spec']['id'] for r in select_omitted(records,set())]==['early','other']
    assert select_omitted(records,{'late','early','duplicate','other'})==[]


def test_only_clean_declared_outcomes_can_enter_followup():
    assert not eligible_status('final_profile_passed')
    for status in ('selection_failed','final_profile_failed','no_development_improvement'):
        assert eligible_status(status)
    for status in ('running','failed','memory_error','training_complete'):
        with pytest.raises(ValueError):eligible_status(status)


def test_waiting_module_does_not_import_neural_or_simulation_libraries():
    child=subprocess.run([sys.executable,'-c',
        "import sys,json; import tools.league_residual_repair_followup; print(json.dumps({x:x in sys.modules for x in ['numpy','torch','jsbsim']}))"],
        check=True,capture_output=True,text=True)
    assert json.loads(child.stdout)==dict(numpy=False,torch=False,jsbsim=False)
