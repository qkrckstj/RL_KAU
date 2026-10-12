"""Verify portable roster hashes and load every actor, without training."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib
import json
from pathlib import Path
from tools.league_matches import Actor


def main():
    root = Path(__file__).resolve().parents[1]
    bundle = root / 'experiments/league/rear_desktop_20261012'
    hashes = json.loads((bundle / 'SHA256.json').read_text('utf-8'))
    for name, digest in hashes.items():
        assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == digest, name
    inputs = json.loads((bundle / 'inputs.json').read_text('utf-8'))
    assert len(inputs['opponents']) == 48
    assert len(inputs['development_opponents']) == 24
    specs = [inputs['teacher']] + inputs['opponents'] + inputs['development_opponents']
    for spec in specs:
        actor = Actor(spec)
        actor.reset()
    assert (root / inputs['parent']).is_file()
    print(json.dumps({'status': 'hashes_and_actor_loading_passed', 'files': len(hashes), 'actors': len(specs), 'training_started': False}))


if __name__ == '__main__':
    main()
