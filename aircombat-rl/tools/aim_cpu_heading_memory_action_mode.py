"""Memory-aware sampled actor with the existing public RNG protocol."""
import json
import shutil
import zipfile
from tools.league_residual_action_mode import io, SAMPLER_SOURCE as BASE_SAMPLER_SOURCE

SAMPLER_SOURCE=BASE_SAMPLER_SOURCE
for old,new in [
    ('features(raw, prior)','features(raw, prior,self.heading_memory.error(raw))'),
    ('return int(self.action_rng.choice(9, p=sampling_probabilities(logits)))','action = int(self.action_rng.choice(9, p=sampling_probabilities(logits)))\n        self.heading_memory.update(raw,action)\n        return action')]:
    if SAMPLER_SOURCE.count(old)!=1:raise ValueError('Sampler boundary changed: '+old)
    SAMPLER_SOURCE=SAMPLER_SOURCE.replace(old,new)


def build_variant(spec, out, seed, identity):
    if out.exists():
        raise FileExistsError('Use a fresh sampled submission folder')
    original = io.ROOT/spec['design']
    source = (original/'policy.py').read_text(encoding='utf-8')
    if source.count('action = int(np.argmax(logits))') != 1 or 'class Policy:' not in source:
        raise ValueError('Expected the preserved memory-aware residual export')
    metadata = dict(action_seed=seed, temperature=1.0, source=spec,
        source_weights_sha256=io.sha(io.ROOT/spec['weights']),
        rng_protocol='Independent NumPy generator seeded by fixed replicate seed and39 public float32 first-frame words; reset per episode. No global RNG consumption or private environment input.',
        scope='Execution-mode diagnostic only; original teacher and learned actor arrays unchanged.')
    out.mkdir(parents=True)
    (out/'policy.py').write_text(source+SAMPLER_SOURCE, encoding='utf-8')
    shutil.copyfile(original/'wrappers.py', out/'wrappers.py')
    with zipfile.ZipFile(io.ROOT/spec['weights']) as old, zipfile.ZipFile(out/'policy_net.zip','w',zipfile.ZIP_DEFLATED) as new:
        if set(old.namelist()) != {'parameters.json','residual_actor.npz','residual_metadata.json'}:
            raise ValueError('Unexpected source archive members')
        for name in old.namelist():
            new.writestr(name, old.read(name))
        new.writestr('sampling_metadata.json', json.dumps(metadata))
    io.write(out/'policy_net.json', metadata)
    variant = dict(id=identity, kind='submission', design=io.relative(out), weights=io.relative(out/'policy_net.zip'))
    io.write(out/'entrant.json', variant)
    return variant

