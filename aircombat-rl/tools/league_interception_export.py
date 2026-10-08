"""Immutable standalone artifacts for the experimental interceptor."""
from pathlib import Path
import json
import zipfile
from experiments.league.interception import Policy
from tools.autolab_cem import ROOT, write, sha


def export(out, parameters, identity):
    parameters = Policy(parameters=parameters).parameters.tolist()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    (out/'policy.py').write_text(
        (ROOT/'experiments/league/interception.py').read_text(encoding='utf-8'), encoding='utf-8')
    (out/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n', encoding='utf-8')
    configuration = dict(parameters=parameters)
    write(out/'policy_net.json', configuration)
    with zipfile.ZipFile(out/'policy_net.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('parameters.json', json.dumps(configuration))
    def shown(path):
        try:
            return path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            return str(path.resolve())
    spec = dict(id=identity, kind='submission', design=shown(out), weights=shown(out/'policy_net.zip'))
    write(out/'entrant.json', spec)
    write(out/'artifact_sha256.json', {name:sha(out/name) for name in
        ('policy.py', 'wrappers.py', 'policy_net.json', 'policy_net.zip', 'entrant.json')})
    return spec
