"""Bundle analytic lead features with the preserved controller and NumPy actor."""
import ast
import json
import zipfile
from pathlib import Path
from tools.league_residual_export import export as residual_export, ACTOR_SOURCE, ROOT
from experiments.rear_async.storage import write, sha


def export(out,teacher,policy,identity):
    out=Path(out)
    spec=residual_export(out,teacher,policy,identity)
    original=(ROOT/teacher['design']/'policy.py').read_text()
    relative=(ROOT/'experiments/league/residual_features.py').read_text()
    aim=(ROOT/'experiments/aim_async/features.py').read_text()
    line='from experiments.league.residual_features import features as relative_features'
    if aim.count(line)!=1:raise ValueError('Aim feature import boundary changed')
    aim=aim.replace(line,'relative_features = features')
    source=original+'\n\nTeacherPolicy = Policy\n\n'+relative+'\n'+aim+'\n'+ACTOR_SOURCE
    allowed={'math','numpy','json','zipfile','pathlib','operator','io','aircombat_gym'}
    for node in ast.walk(ast.parse(source)):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        if isinstance(node,ast.ImportFrom) and node.level:raise ValueError('Unbundled relative import')
        if any(n.split('.')[0] not in allowed for n in names):raise ValueError('Unbundled import')
    (out/'policy.py').write_text(source)
    metadata=json.loads((out/'policy_net.json').read_text())
    metadata.update(kind='predictive_aim_ppo_actor',observation_dim=43,
        feature_contract='31 relative +3 public analytic lead +9 prior. Submission remains39 raw channels.',
        first_layer='Original40 and new3 dot products computed separately; initial addedweights0 preserve original logits.')
    with zipfile.ZipFile(out/'policy_net.zip') as archive:members={n:archive.read(n) for n in archive.namelist()}
    members['residual_metadata.json']=json.dumps(metadata).encode()
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in members.items():archive.writestr(name,value)
    write(out/'policy_net.json',metadata)
    write(out/'artifact_sha256.json',{n:sha(out/n) for n in ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec
