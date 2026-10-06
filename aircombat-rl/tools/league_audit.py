"""Audit the completed first league batch from raw evidence, not progress labels."""
from argparse import ArgumentParser
from pathlib import Path
import gzip
import json
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_train import verify
from tools.league_validate import heldout_analysis
from tools.league_replay import load_bundle


def audit(run,bundle,out):
    plan=read(run/'main/plan.json')
    verify(plan)
    history=read(run/'main/history.json')
    main=read(run/'main/completion.json')
    assert len(history)==main['rounds']==plan['rounds']==6
    assert sum(r['admission']['promoted'] for r in history)==main['promotions']==3
    follow=run/'followup_pool'
    freeze=read(follow/'frozen_selection.json')
    selected=freeze['selected']
    results=[]
    foes={s['id'] for s in freeze['opponents']}
    own_ids={s['id'] for s in selected}|{'cem_original'}
    assert len(foes)==16 and len(own_ids)==4
    expected_keys={(40000000+i,seat) for i in range(40) for seat in ('red','blue')}
    for p in sorted((follow/'heldout').glob('match_*.json')):
        row=read(p)
        job,result=row['job'],row['result']
        assert job['n']==80 and job['band']==result['band']==40000000
        assert result['own']==job['own']['id'] and result['foe']==job['foe']['id']
        assert len(result['episodes'])==80
        assert {(e['seed'],e['seat']) for e in result['episodes']}==expected_keys
        results.append(result)
    assert len(results)==64
    assert {(r['own'],r['foe']) for r in results}=={(a,b) for a in own_ids for b in foes}
    reanalysis=heldout_analysis(results,[s['id'] for s in selected],'cem_original',freeze['plan']['transfer_opponents'])
    assert reanalysis==read(follow/'verdict.json') and reanalysis['passed']
    claim=read(run/'heldout_40000000.claim.json')
    assert claim['selection_sha256']==sha(follow/'frozen_selection.json')
    assert read(follow/'chosen/entrant.json')==freeze['chosen']
    transfer=set(freeze['plan']['transfer_opponents'])
    for r in history:
        assert not transfer.intersection(r['pool_ids'])
    for seed in freeze['plan']['replication_seeds']:
        record=read(follow/f'replication/s{seed}/plan.json')
        assert not transfer.intersection(s['id'] for s in record['opponents'])
        assert read(follow/f'replication/s{seed}/result.json')['status']=='complete'
    peers=read(follow/'refinement_crossplay.json')
    assert len(peers)==3 and sum(len(r['episodes']) for r in peers)==240
    manifest=load_bundle(bundle)
    assert manifest['stage']=='evaluation_passed'
    assert read(bundle/'models/final/entrant.json')==freeze['chosen']
    assert json.loads(gzip.decompress((bundle/'evidence/final/episodes.json.gz').read_bytes()))==results
    loader=read(ROOT/'experiments/league/results/loader_check.json')
    assert loader['status']=='passed' and loader['bundle_manifest_sha256']==sha(bundle/'manifest.json')
    assert loader['official_and_search_episodes_identical'] and loader['original_policy_preserved'] and loader['original_weights_preserved']
    original=ROOT/'experiments/plan_a/cem_policy'
    old_selection=read(original/'selection.json')
    assert sha(original/'policy.py')==old_selection['policy_sha256']
    assert sha(original/'policy_net.zip')==old_selection['archive_sha256']
    finish=read(run/'finishing/completion.json')
    assert finish['actual_loader_and_cli_checks_passed']
    for name in ('loader_check','duel_check','replay_check'):
        assert read(run/f'finishing/{name}.json')['returncode']==0
    assert len(read(run/'finishing/duel/result.json')['episodes'])==2
    assert read(run/'finishing/replay/completion.json')['status']=='complete'
    confirm=read(run/'runner_confirm/completion.json')
    assert confirm['status']=='complete'
    state=read(ROOT.parent/'docs/LEAGUE_STATE.json')
    for path in state['evidence'].values():
        assert (ROOT.parent/path).is_file(),path
    audit_result=dict(status='passed',scope='Completed first league batch and portable artifacts; not a universal tournament claim or completion of all follow-up work.',
        rounds=6,promotions=3,independent_refinements_of_one_parent=3,
        final_matchups=64,final_games=sum(len(r['episodes']) for r in results),
        shared_initial_conditions=40,seats=2,opponents=16,refinement_peer_games=240,
        recomputed_verdict_matches_saved=True,choice_matches_pretest_claim=True,
        excluded_opponents_absent_from_training=True,frozen_source_and_input_hashes_verified=True,
        original_cem_preserved=True,portable_bundle_files_verified=len(manifest['files']),
        bundle_manifest_sha256=sha(bundle/'manifest.json'),raw_results_identical_in_bundle=True,
        actual_loader_and_duel_and_replay_checks_passed=True,
        current_state_references_exist=True,
        runner_confirmation=confirm,overall_goal_complete=False,
        remaining='Train and evaluate a conditionally selected pursuit response against the archived population on new development/final conditions.')
    write(out,audit_result)
    print(json.dumps({k:audit_result[k] for k in ('status','rounds','final_games','portable_bundle_files_verified','overall_goal_complete')}))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'runs/league_20261006')
    p.add_argument('--bundle',type=Path,default=ROOT/'experiments/league/bundle')
    p.add_argument('--out',type=Path,default=ROOT.parent/'docs/verification/league_final_audit.json')
    a=p.parse_args()
    audit(a.run.resolve(),a.bundle.resolve(),a.out.resolve())
