"""Export the preserved teacher and learned residual as one NumPy submission."""
from io import BytesIO
from pathlib import Path
import ast
import json
import zipfile
import numpy as np

from tools.autolab_cem import ROOT, sha
from tools.league_reliable_io import write
from experiments.league.residual_ppo_policy import actor_parameters


ACTOR_SOURCE = '''
from io import BytesIO

class Policy:
    ACTION_MODE = 'discrete'

    def __init__(self, weights=None, device='cpu'):
        if weights is None:
            raise ValueError('A teacher-plus-residual weight archive is required')
        self.weights, self.device = weights, device
        with zipfile.ZipFile(weights) as archive:
            with np.load(BytesIO(archive.read('residual_actor.npz')), allow_pickle=False) as arrays:
                self.parameters = {k: arrays[k].copy() for k in arrays.files}
        p = self.parameters
        if set(p) != {'w0','b0','w1','b1','wa','ba','prior_bias'}:
            raise ValueError('Unexpected residual actor arrays')
        if not all(np.isfinite(v).all() for v in p.values()):
            raise ValueError('Nonfinite actor parameters')
        if (p['w0'].ndim != 2 or p['w0'].shape[1] != FEATURE_DIM
                or p['b0'].shape != (p['w0'].shape[0],)
                or p['w1'].ndim != 2 or p['w1'].shape[1] != p['w0'].shape[0]
                or p['b1'].shape != (p['w1'].shape[0],)
                or p['wa'].shape != (9,p['w1'].shape[0]) or p['ba'].shape != (9,)
                or p['prior_bias'].shape != () or float(p['prior_bias']) <= 0):
            raise ValueError('Incompatible residual actor architecture')
        self.reset()

    def reset(self):
        self.teacher = TeacherPolicy(self.weights, device=self.device)
        self.last_clock = None

    def act(self, obs):
        raw = np.asarray(obs,dtype=np.float32)
        if raw.shape != (39,) or not np.isfinite(raw).all():
            raise ValueError('Expected39 finite public channels')
        clock = float(raw[38])
        if self.last_clock is not None and clock > self.last_clock+1e-6:
            self.reset()
        self.last_clock = clock
        prior = self.teacher.act(raw)
        logits = numpy_logits(features(raw,prior),self.parameters)
        if not np.isfinite(logits).all():
            raise ValueError('Nonfinite action logits')
        return int(np.argmax(logits))

    def __str__(self):
        return 'Preserved public-state teacher with learned PPO residual actions'
'''


def export(out, teacher, policy, identity):
    out = Path(out)
    if out.exists(): raise FileExistsError('Use a new policy output folder')
    if teacher['kind'] != 'submission':
        raise ValueError('Export requires a self-contained archived teacher submission')
    teacher_source = ROOT/teacher['design']/'policy.py'
    teacher_weights = ROOT/teacher['weights']
    original = teacher_source.read_text(encoding='utf-8')
    feature_source = (ROOT/'experiments/league/residual_features.py').read_text(encoding='utf-8')
    source = original+'\n\nTeacherPolicy = Policy\n\n'+feature_source+'\n'+ACTOR_SOURCE
    allowed = {'math','numpy','json','zipfile','pathlib','operator','io','aircombat_gym'}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import): modules = [n.name for n in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level: raise ValueError('Relative imports are not self-contained')
            modules = [node.module or '']
        else: continue
        if any(name.split('.')[0] not in allowed for name in modules):
            raise ValueError('Teacher or features require an unbundled import')
    parameters = actor_parameters(policy)
    payload = BytesIO(); np.savez_compressed(payload, **parameters)
    with zipfile.ZipFile(teacher_weights) as archive:
        if archive.namelist() != ['parameters.json']:
            raise ValueError('Expected a preserved controller parameter archive')
        teacher_configuration = archive.read('parameters.json')
    out.mkdir(parents=True)
    (out/'policy.py').write_text(source, encoding='utf-8')
    (out/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n', encoding='utf-8')
    metadata = dict(kind='residual_ppo_actor', teacher=teacher,
        teacher_source_sha256=sha(teacher_source), teacher_weights_sha256=sha(teacher_weights),
        prior_bias=policy.prior_bias, observation_dim=40, actions=9,
        scope='Teacher source/configuration and residual actor are bundled. Original teacher paths are provenance only, not runtime dependencies.')
    write(out/'policy_net.json', metadata)
    with zipfile.ZipFile(out/'policy_net.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('parameters.json', teacher_configuration)
        archive.writestr('residual_actor.npz', payload.getvalue())
        archive.writestr('residual_metadata.json', json.dumps(metadata))
    def shown(path):
        try: return path.resolve().relative_to(ROOT).as_posix()
        except ValueError: return str(path.resolve())
    spec = dict(id=identity, kind='submission', design=shown(out), weights=shown(out/'policy_net.zip'))
    write(out/'entrant.json', spec)
    write(out/'artifact_sha256.json', {n:sha(out/n) for n in
          ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec
