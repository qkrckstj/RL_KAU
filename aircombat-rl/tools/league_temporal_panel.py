"""Export and freeze a temporal opponent panel before observing its results."""
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
import json
import zipfile

from experiments.league.temporal_opponents import Policy
from tools.autolab_cem import ROOT, read, write, sha
from tools.league_symmetric_train import immutable_json
from tools.league_train import verify


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def export(folder, configuration, identity):
    Policy(configuration=configuration)
    folder.mkdir(parents=True, exist_ok=False)
    (folder/'policy.py').write_bytes((ROOT/'experiments/league/temporal_opponents.py').read_bytes())
    (folder/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n', encoding='utf-8')
    write(folder/'policy_net.json', configuration)
    with zipfile.ZipFile(folder/'policy_net.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('parameters.json', json.dumps(configuration, sort_keys=True))
    spec = dict(id=identity, kind='submission', design=relative(folder), weights=relative(folder/'policy_net.zip'))
    write(folder/'entrant.json', spec)
    write(folder/'artifact_sha256.json', {n:sha(folder/n) for n in
        ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec


def freeze(panel, archive, training_plan, band=52000000, n=80):
    if panel.exists():
        previous = read(panel)
        if previous['band'] != band or previous['n'] != n or previous['training_plan'] != relative(training_plan):
            raise ValueError('Cannot change frozen audit panel')
        verify(previous)
        return previous
    if n <= 0 or n % 2:
        raise ValueError('Complete seat pairs required')
    training = read(training_plan)
    verify(training)
    if (ROOT/f'runs/holdout_claim_{band}.json').exists():
        raise ValueError('Audit band already claimed')
    source_names = ['experiments/league/temporal_opponents.py','tools/league_temporal_panel.py']
    family_sha = sha(ROOT/source_names[0])
    for foe in training['opponents']:
        if foe['kind']=='submission' and sha(ROOT/foe['design']/'policy.py') == family_sha:
            raise ValueError('Temporal policy code already appeared in training')
    opponents = []
    for mode in ('weave','extend','delayed','pulsed'):
        for side, period, speed in ((-1,4.,400.),(1,8.,600.)):
            name = f'temporal_{mode}_{"left" if side<0 else "right"}'
            opponents.append(export(archive/name, dict(mode=mode,side=side,period=period,speed=speed),name))
    inputs = {relative(training_plan):sha(training_plan)}
    for foe in opponents:
        folder = ROOT/foe['design']
        for name in read(folder/'artifact_sha256.json'):
            inputs[relative(folder/name)] = sha(folder/name)
        inputs[relative(folder/'artifact_sha256.json')] = sha(folder/'artifact_sha256.json')
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
        purpose='Post-selection audit against eight untrained temporal opponents',
        training_plan=relative(training_plan), band=band,n=n,workers=16,
        opponents=opponents,source_sha256={name:sha(ROOT/name) for name in source_names},
        input_sha256=inputs, outcomes_observed_at_freeze=False,
        limits='Eight scripted temporal opponents, not expert pilots, real student policies, exhaustive situations, or eight independent architectures. No candidate or opponent selection from audit outcomes. Once evaluated these opponents cannot be called unseen in later training.',
        intended_comparison='Original CEM, current validated anchor, and one policy selected before final testing. Run only if that selected policy passes the preceding final evaluation. Report paired IC uncertainty and every opponent; audit completion is not a performance pass.')
    immutable_json(panel,result)
    return result


if __name__ == '__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--panel',type=Path,default=Path('experiments/league/temporal_panel_20261006.json'))
    parser.add_argument('--archive',type=Path,default=Path('experiments/league/unseen_temporal_20261006'))
    parser.add_argument('--training-plan',type=Path,default=Path('runs/league_conditions_20261006/plan.json'))
    args=parser.parse_args()
    panel=freeze(args.panel.resolve(),args.archive.resolve(),args.training_plan.resolve())
    print(json.dumps(dict(opponents=len(panel['opponents']),band=panel['band'],n=panel['n'],outcomes_observed=False)))
