"""Bundle the exact state-dependent prior with the preserved public teacher."""
import ast
from io import BytesIO
import json
from pathlib import Path
import zipfile
import numpy as np
from tools.aim_cpu_export import export as aim_export
from tools.league_residual_export import ACTOR_SOURCE,ROOT
from experiments.rear_async.storage import write,sha


def export(out,teacher,policy,identity):
    if policy.prior_bias!=6. or policy.conflict_prior_bias!=3.:
        raise ValueError('Expected base6/conflict3policy')
    out=Path(out);spec=aim_export(out,teacher,policy,identity)
    original=(ROOT/teacher['design']/'policy.py').read_text()
    relative=(ROOT/'experiments/league/residual_features.py').read_text()
    aim=(ROOT/'experiments/aim_async/features.py').read_text()
    line='from experiments.league.residual_features import features as relative_features'
    if aim.count(line)!=1:raise ValueError('Aim feature boundary differs')
    aim=aim.replace(line,'relative_features = features')
    gate=(ROOT/'experiments/aim_async/state_prior_numpy.py').read_text()
    line='from experiments.aim_async.features import numpy_logits as base_numpy_logits'
    if gate.count(line)!=1:raise ValueError('Gate feature boundary differs')
    gate=gate.replace(line,'base_numpy_logits = numpy_logits')
    actor=ACTOR_SOURCE
    old="{'w0','b0','w1','b1','wa','ba','prior_bias'}"
    if actor.count(old)!=1:raise ValueError('Actor array boundary differs')
    actor=actor.replace(old,"{'w0','b0','w1','b1','wa','ba','prior_bias','conflict_prior_bias'}")
    old="or p['prior_bias'].shape != () or float(p['prior_bias']) <= 0):"
    if actor.count(old)!=1:raise ValueError('Actor scalar validation boundary differs')
    actor=actor.replace(old,"or p['prior_bias'].shape != () or float(p['prior_bias']) <= 0\n                or p['conflict_prior_bias'].shape != ()\n                or not 0 < float(p['conflict_prior_bias']) < float(p['prior_bias'])):")
    source=original+'\n\nTeacherPolicy = Policy\n\n'+relative+'\n'+aim+'\n'+gate+'\n'+actor
    allowed={'math','numpy','json','zipfile','pathlib','operator','io','aircombat_gym'}
    for node in ast.walk(ast.parse(source)):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        if isinstance(node,ast.ImportFrom) and node.level:raise ValueError('Unbundled relative import')
        if any(n.split('.')[0] not in allowed for n in names):raise ValueError('Unbundled import')
    (out/'policy.py').write_text(source)
    with zipfile.ZipFile(out/'policy_net.zip') as archive:
        members={n:archive.read(n) for n in archive.namelist()}
    with np.load(BytesIO(members['residual_actor.npz']),allow_pickle=False) as arrays:
        parameters={k:arrays[k].copy() for k in arrays.files}
    parameters['conflict_prior_bias']=np.asarray(policy.conflict_prior_bias,dtype=np.float32)
    payload=BytesIO();np.savez_compressed(payload,**parameters);members['residual_actor.npz']=payload.getvalue()
    metadata=json.loads((out/'policy_net.json').read_text())
    metadata.update(kind='state_conflict_prior_aim_ppo_actor',prior_bias=6.,conflict_prior_bias=3.,
        prior_rule='enemy_solution_and_teacher_opposes_lead_v1',lead_threshold_deg=30.,
        state_rule='Existing public43features only; no elapsed-time switch,foe identity or forced action.')
    members['residual_metadata.json']=json.dumps(metadata).encode()
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in members.items():archive.writestr(name,value)
    write(out/'policy_net.json',metadata)
    write(out/'artifact_sha256.json',{n:sha(out/n) for n in ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec
