"""Self-contained history actor exports and checkpoint adapter for the new run."""
from io import BytesIO
from pathlib import Path
import ast,json,random,shutil,zipfile
import numpy as np
import torch
from tools import league_thread_benchmark as io
from tools.league_residual_export import ACTOR_SOURCE
from tools.league_residual_action_mode import SAMPLER_SOURCE,build_variant as plain_variant
from tools.league_residual_train import save_checkpoint as plain_checkpoint
from experiments.league.residual_ppo_policy import actor_parameters
from experiments.league.history_features import HISTORY_DIM,PUBLIC_DIM,numpy_logits


def export(out,teacher,policy,identity):
    if out.exists():raise FileExistsError('Preserve exported policy')
    original=(io.ROOT/teacher['design']/'policy.py').read_text('utf-8')
    base=(io.ROOT/'experiments/league/residual_features.py').read_text('utf-8')
    history=(io.ROOT/'experiments/league/history_features.py').read_text('utf-8').replace('from experiments.league.residual_features import FEATURE_DIM as BASE_DIM','BASE_DIM = FEATURE_DIM')
    actor=ACTOR_SOURCE.replace('FEATURE_DIM','HISTORY_DIM').replace('self.last_clock = None','self.last_clock = None\n        self.history = HistoryFeatures()').replace('numpy_logits(features(raw,prior),self.parameters)','numpy_logits(self.history.update(features(raw,prior), -clock),self.parameters)')
    if 'self.history.update' not in actor:raise ValueError('History actor template mismatch')
    source=original+'\n\nTeacherPolicy = Policy\n\n'+base+'\n'+history+'\n'+actor
    allowed={'math','numpy','json','zipfile','pathlib','operator','io','aircombat_gym','collections'}
    for node in ast.walk(ast.parse(source)):
        modules=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
        if isinstance(node,ast.ImportFrom) and node.level:raise ValueError('Relative runtime dependency')
        if any(m.split('.')[0] not in allowed for m in modules):raise ValueError('Unbundled runtime import')
    p=actor_parameters(policy);payload=BytesIO();np.savez_compressed(payload,**p)
    with zipfile.ZipFile(io.ROOT/teacher['weights']) as archive:
        if archive.namelist()!=['parameters.json']:raise ValueError('Expected preserved CEM teacher archive')
        teacher_bytes=archive.read('parameters.json')
    metadata=dict(kind='history_residual_ppo_actor',teacher=teacher,observation_dim=HISTORY_DIM,actions=9,history_lags_seconds=[1,5],prior_bias=policy.prior_bias,scope='Bundled public-state history; no external prototype source required at runtime.')
    out.mkdir(parents=True);(out/'policy.py').write_text(source,encoding='utf-8');(out/'wrappers.py').write_text('class State:\n    def __call__(self, obs): return obs\n',encoding='utf-8')
    with zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('parameters.json',teacher_bytes);archive.writestr('residual_actor.npz',payload.getvalue());archive.writestr('residual_metadata.json',json.dumps(metadata))
    spec=dict(id=identity,kind='submission',design=io.relative(out),weights=io.relative(out/'policy_net.zip'))
    io.write(out/'policy_net.json',metadata);io.write(out/'entrant.json',spec)
    io.write(out/'artifact_sha256.json',{n:io.sha(out/n) for n in ('policy.py','wrappers.py','policy_net.json','policy_net.zip','entrant.json')})
    return spec


def build_variant(spec,out,seed,identity):
    original=io.ROOT/spec['design'];source=(original/'policy.py').read_text('utf-8')
    if 'self.history = HistoryFeatures()' not in source:return plain_variant(spec,out,seed,identity)
    if out.exists():raise FileExistsError('Preserve sampled actor')
    sampler=SAMPLER_SOURCE.replace('numpy_logits(features(raw, prior), self.parameters)','numpy_logits(self.history.update(features(raw, prior), -clock), self.parameters)')
    metadata=dict(action_seed=seed,temperature=1.,source=spec,source_weights_sha256=io.sha(io.ROOT/spec['weights']),rng_protocol='Same39 public first-frame words and fixed action seed as plain actor; no IDs/private seeds.')
    out.mkdir(parents=True);(out/'policy.py').write_text(source+sampler,encoding='utf-8');shutil.copyfile(original/'wrappers.py',out/'wrappers.py')
    with zipfile.ZipFile(io.ROOT/spec['weights']) as old,zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as new:
        for name in old.namelist():new.writestr(name,old.read(name))
        new.writestr('sampling_metadata.json',json.dumps(metadata))
    variant=dict(id=identity,kind='submission',design=io.relative(out),weights=io.relative(out/'policy_net.zip'))
    io.write(out/'policy_net.json',metadata);io.write(out/'entrant.json',variant)
    return variant


def save_checkpoint(model,folder,teacher,identity):
    if model.observation_space.shape!=(HISTORY_DIM,):return plain_checkpoint(model,folder,teacher,identity)
    if folder.exists():raise FileExistsError('Preserve checkpoint')
    folder.mkdir(parents=True);model.save(folder/'learner.zip');torch.save(dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()),folder/'rng.pt')
    rng=np.random.default_rng(5872);x=rng.uniform(-1,1,(2048,HISTORY_DIM)).astype(np.float32);x[:,-9:]=np.eye(9,dtype=np.float32)[rng.integers(9,size=len(x))]
    with torch.no_grad():
        t=torch.as_tensor(x);pi=model.policy.mlp_extractor.forward_actor(model.policy.extract_features(t));logits=(model.policy.action_net(pi)+model.policy.prior_bias*t[:,-9:]).cpu().numpy()
    error=float(np.max(np.abs(logits-numpy_logits(x,actor_parameters(model.policy)))))
    if error>2e-5:raise AssertionError('History NumPy/Torch logits disagree')
    spec=export(folder/'submission',teacher,model.policy,identity)
    io.write(folder/'checkpoint.json',dict(step=model.num_timesteps,entrant=spec,agreement=dict(observations=len(x),maximum_logit_error=error),learner_sha256=io.sha(folder/'learner.zip'),resume_scope='Learner/optimizer/RNG saved; public histories and JSBSim mid-episode state are not restored. Fresh streams on continuation.'))
    return spec
