"""Export the adaptive policy as one standalone policy.py accepted by the grader."""
from pathlib import Path
import json
import zipfile
from tools.autolab_cem import ROOT,write


def export(out,configuration,identity):
    out=Path(out)
    if out.exists():
        raise FileExistsError('Adaptive candidate artifacts are immutable')
    out.mkdir(parents=True)
    expert=(ROOT/'experiments/league/controller.py').read_text(encoding='utf-8')
    adaptive=(ROOT/'experiments/league/adaptive_symmetric.py').read_text(encoding='utf-8')
    # The controller is self-contained; embedding it avoids module-name
    # collisions when several submissions coexist in one tournament process.
    standalone=adaptive.replace('from experiments.league.controller import Policy as Expert',expert+'\nExpert = Policy\n')
    (out/'policy.py').write_text(standalone,encoding='utf-8')
    (out/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
    write(out/'policy_net.json',configuration)
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('parameters.json',json.dumps(configuration))
    def shown(path):
        path=path.resolve()
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)
    spec=dict(id=identity,kind='submission',design=shown(out),weights=shown(out/'policy_net.zip'))
    write(out/'entrant.json',spec)
    return spec
