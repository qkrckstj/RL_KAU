"""Fresh, reproducible records for the optional DQN/CEM experiments."""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess

import torch

APP = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def run_record(args, algorithm, extra):
    """Never overwrite any previous run, including a failed one."""
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA simulation is required')
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.reset_peak_memory_stats()
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        config.update(algorithm=algorithm, experimental=True, drop_in_fidelity_passed=False,
                      torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                      physics_version=version('jsbsim-f16-cuda'), **extra)
        config['base_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=APP, text=True).strip()
        files = list((APP / 'experiments/gpu_sim').rglob('*.py'))
        files += list((APP / 'aircombat_gym').rglob('*.py'))
        files += [APP / 'experiments/league/controller.py', APP / 'experiments/plan_a/core.py']
        hashes = {}
        for source in sorted(set(files)):
            relative = source.relative_to(APP)
            data = source.read_bytes()
            target = args.out / 'source_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
        config['source_sha256'] = hashes
        write_json(args.out / 'config.json', config)
        yield config
    except Exception as exc:
        write_json(args.out / 'failure.json', {'error': repr(exc)})
        raise
