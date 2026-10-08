"""Export a self-contained gated interceptor without experiment imports."""
from pathlib import Path
import json
import zipfile
import re
from experiments.league.interception_gate import Policy
from tools.autolab_cem import ROOT, write, sha


def export(out, configuration, identity):
    Policy(configuration=configuration)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    source = lambda p:(ROOT/p).read_text(encoding='utf-8')
    expert = source('experiments/league/controller.py')
    expert = re.sub(r'\b(LOW|HIGH)\b', lambda m:'EXPERT_'+m[0], expert)
    gate = source('experiments/league/adaptive_symmetric.py').replace(
        'from experiments.league.controller import Policy as Expert', expert+'\nExpert = Policy\n')
    interceptor = source('experiments/league/interception.py')
    interceptor = re.sub(r'\b(LOW|HIGH|INITIAL)\b', lambda m:'INTERCEPT_'+m[0], interceptor)
    hybrid = source('experiments/league/interception_gate.py').replace(
        'from experiments.league.adaptive_symmetric import Policy as Gate', gate+'\nGate = Policy\n').replace(
        'from experiments.league.interception import Policy as Interceptor', interceptor+'\nInterceptor = Policy\n')
    (out/'policy.py').write_text(hybrid, encoding='utf-8')
    (out/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n', encoding='utf-8')
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
