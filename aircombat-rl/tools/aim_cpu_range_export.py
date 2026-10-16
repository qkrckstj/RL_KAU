"""Bundle the public-threat prior with the preserved teacher and pure NumPy actor."""
import ast
from io import BytesIO
import json
from pathlib import Path
import zipfile
import numpy as np
from experiments.aim_async.range_policy import RangePriorPolicy
from experiments.aim_async.range_numpy import range_limit
from tools.aim_cpu_export import export as aim_export
from tools.league_residual_export import ACTOR_SOURCE,ROOT
from experiments.rear_async.storage import write,sha


def export(out,teacher,policy,identity):
    if type(policy) is not RangePriorPolicy or policy.prior_bias!=6. or policy.threat_prior_bias!=3.:
        raise ValueError('Expected RangePriorPolicy with base6/lower3')
    limit=range_limit(policy.range_threshold_m)
    if policy.scaled_range_limit!=float(limit):raise ValueError('Policy range cache differs from explicit threshold')
    out=Path(out)
    spec=aim_export(out,teacher,policy,identity)
    original=(ROOT/teacher['design']/'policy.py').read_text()
    relative=(ROOT/'experiments/league/residual_features.py').read_text()
    aim=(ROOT/'experiments/aim_async/features.py').read_text()
    line='from experiments.league.residual_features import features as relative_features'
    if aim.count(line)!=1:raise ValueError('Aim feature boundary differs')
    aim=aim.replace(line,'relative_features = features')
    helper=(ROOT/'experiments/aim_async/threat_prior_numpy.py').read_text()
    definitions=[n for n in ast.parse(helper).body if isinstance(n,ast.FunctionDef) and n.name=='prior_values']
    if len(definitions)!=1:raise ValueError('Prior validation boundary differs')
    validation=ast.get_source_segment(helper,definitions[0])
    gate=(ROOT/'experiments/aim_async/range_numpy.py').read_text()
    validation_import='from experiments.aim_async.threat_prior_numpy import prior_values'
    if gate.count(validation_import)!=1:raise ValueError('Prior validation import differs')
    gate=gate.replace(validation_import,validation)
    line='from experiments.aim_async.features import numpy_logits as base_numpy_logits'
    if gate.count(line)!=1:raise ValueError('Threat feature boundary differs')
    gate=gate.replace(line,'base_numpy_logits = numpy_logits')
    actor='import json\n'+ACTOR_SOURCE
    line="{'w0','b0','w1','b1','wa','ba','prior_bias'}"
    if actor.count(line)!=1:raise ValueError('Actor array boundary differs')
    actor=actor.replace(line,"{'w0','b0','w1','b1','wa','ba','prior_bias','threat_prior_bias','range_threshold_m'}")
    line='\n        self.reset()\n'
    if actor.count(line)!=1:raise ValueError('Actor validation boundary differs')
    actor=actor.replace(line,"\n        prior_values(p['prior_bias'],p['threat_prior_bias'])\n        self.reset()\n")
    finite_line='        if not all(np.isfinite(v).all() for v in p.values()):'
    if actor.count(finite_line)!=1:raise ValueError('Actor range validation boundary differs')
    actor=actor.replace(finite_line,"        range_limit(p['range_threshold_m'])\n"+finite_line)
    metadata_validation="""
        with zipfile.ZipFile(weights) as archive:
            range_metadata=json.loads(archive.read('residual_metadata.json'))
        range_limit(range_metadata['range_threshold_m'])
        if (float(range_metadata['range_threshold_m'])!=float(p['range_threshold_m'])
                or range_metadata.get('float32_scaled_range_limit')!=float(range_limit(p['range_threshold_m']))
                or range_metadata.get('prior_rule')!='public_wez_or_damaged_or_parameterized_float32_range_v1'
                or range_metadata.get('range_feature_index')!=0
                or range_metadata.get('prior_bias')!=float(p['prior_bias'])
                or range_metadata.get('threat_prior_bias')!=float(p['threat_prior_bias'])):
            raise ValueError('Archive range metadata differs from actor parameters')
"""
    actor=actor.replace("        prior_values(p['prior_bias'],p['threat_prior_bias'])","        prior_values(p['prior_bias'],p['threat_prior_bias'])"+metadata_validation)
    source=original+'\n\nTeacherPolicy = Policy\n\n'+relative+'\n'+aim+'\n'+gate+'\n'+actor
    allowed={'math','numpy','json','zipfile','pathlib','operator','io','aircombat_gym'}
    for node in ast.walk(ast.parse(source)):
        names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        if isinstance(node,ast.ImportFrom) and node.level:raise ValueError('Unbundled relative import')
        if any(n.split('.')[0] not in allowed for n in names):raise ValueError('Unbundled import')
    (out/'policy.py').write_text(source)
    with zipfile.ZipFile(out/'policy_net.zip') as archive:members={n:archive.read(n) for n in archive.namelist()}
    with np.load(BytesIO(members['residual_actor.npz']),allow_pickle=False) as archive:parameters={k:archive[k].copy() for k in archive.files}
    parameters['threat_prior_bias']=np.asarray(policy.threat_prior_bias,dtype=np.float32)
    parameters['range_threshold_m']=np.asarray(policy.range_threshold_m,dtype=np.float64)
    buffer=BytesIO();np.savez_compressed(buffer,**parameters);members['residual_actor.npz']=buffer.getvalue()
    metadata=json.loads((out/'policy_net.json').read_text())
    metadata.update(kind='public_parameterized_range_prior_aim_ppo_actor',prior_bias=6.,threat_prior_bias=3.,
        prior_rule='public_wez_or_damaged_or_parameterized_float32_range_v1',threat_feature_index=24,threshold=.5,
        own_health_feature_index=19,own_health_threshold=1.,range_feature_index=0,range_threshold_m=policy.range_threshold_m,
        float32_scaled_range_limit=float(limit),
        state_rule='Existing public43features;opponentWEZ OR ownhealth<1 OR float32scaledrange<=explicitthreshold;no newstate,forcedaction,clock switch orfoeidentity.')
    members['residual_metadata.json']=json.dumps(metadata).encode()
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name,value in members.items():archive.writestr(name,value)
    write(out/'policy_net.json',metadata)
    write(out/'artifact_sha256.json',{name:sha(out/name) for name in ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec
