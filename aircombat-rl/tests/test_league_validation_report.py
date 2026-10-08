import json
import subprocess
import sys
import pytest
from tools import league_validation_report as module
from tools.league_matches import summary
from tools.league_tournament_metrics import compare
from tests.test_league_regret_train import record, FOES


def fixture(source):
    models=[dict(id=n) for n in ('anchor','parent','warm','chosen')]
    groups={f:f for f in FOES};requests=[];records=[]
    for spec,rate in zip(models,(.3,.4,.7,.9)):
        r=record([rate]*len(FOES),spec['id'],20,1000)
        for f in r['results']:
            f['own']=spec['id']
            for e in f['episodes']:
                e.update(won=e['outcome']=='kill',opp_health=1-e['own_health'],t=40.,wez_time=1.,wez_time_foe=1.,
                    first_wez=10.,longest_track=1.,min_range=100.,out_of_bounds=False)
            f['summary']=summary(f['episodes'])
            f['damage_traces']=[dict(seed=e['seed'],seat=e['seat'],first_hit=10.,damage_seconds=.3,
                timeline_1s=[dict(t=0.,hp=1.)],terminal=dict(t=40.,hp=e['own_health']),
                snapshots={'30':dict(t=30.,hp=e['own_health'])}) for e in f['episodes']]
        module.write(source/f'final/candidate_{len(records):03d}.json',r)
        records.append(r);requests.append(r['request'])
    plan=dict(source_sha256={},input_sha256={},final_models=models,groups=groups,final_band=1000,
        final_n=20,opponents=requests[0]['opponents'])
    completed=dict(status='tournament_profile_passed',selected=models[-1],selected_before_test=True,temporal_opened=False,
        comparisons=[compare(records[-1]['results'],r['results'],groups,final=True) for r in records[:2]],
        warm_comparison=compare(records[-1]['results'],records[2]['results'],groups,final=True))
    module.write(source/'plan.json',plan);module.write(source/'completion.json',completed)
    module.write(source/'final_decision.json',completed)
    module.write(source/'frozen_selection.json',dict(chosen=models[-1]))
    module.write(source/'final/schedule.json',dict(requests=requests,traced=True))


def test_waiter_imports_no_numerical_libraries():
    r=subprocess.run([sys.executable,'-c',"import sys; import tools.league_validation_report; print(int('numpy' in sys.modules),int('torch' in sys.modules))"],capture_output=True,text=True,check=True)
    assert r.stdout.strip()=='0 0'


def test_recomputes_results_and_rejects_corrupted_summary(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'ROOT',tmp_path)
    source=tmp_path/'source';fixture(source)
    result=module.collect_report(source,tmp_path/'report')
    assert result['profile_passed'] and not result['promotion_performed']
    assert result['records'][-1]['wdl']['wins']==144
    assert (tmp_path/'report/report.md').exists()
    path=source/'final/candidate_003.json';r=module.read(path);r['results'][0]['summary']['wins']+=1;module.write(path,r)
    with pytest.raises(ValueError,match='summary does not reproduce'):
        module.collect_report(source,tmp_path/'bad_report')
