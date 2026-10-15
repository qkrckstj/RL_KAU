# Desktop reproduction and seven preserved models — 2026-10-15

This is a classroom reinforcement-learning simulation. This publication does not connect to real aircraft, weapons, or operational systems. Preserve the repository's native CPU JSBSim simulation, controller, observations and individual terminal-health verdict. The current local verdict is `user_terminal_health_round4_v2`; older reports used other rules and remain historical.

## Selection and scope

The seven models come from the latest [family report](HISTORICAL_FAMILY_WIN_RATE_RANKING_20261015.md): s44701, s45800, s47236, s47238, s46201, s46401 and s46500. They are representatives selected from historical evaluations with different panels, not a matched strength ranking. See [manifest](../artifacts/20261015/manifest.json) for exact checkpoint paths and SHA-256 hashes.

Each model includes original greedy and sampled inference exports, learner/optimizer state and READY metadata. Historical result summaries and compact training settings accompany the models. Full source plans, raw game logs, RNG/worker states and intermediate models remain local; large frozen-input hash tables are excluded from the compact settings, with original plan SHA-256 recorded. The original files are copied byte-for-byte; historical READY/entrant/plan absolute paths and temporary paths are provenance and must not be executed as portable paths. Use the manifest's repository-relative checkpoint paths as below. This bundle supports inference on a new computer; exact process-level resumption of historical runs is not claimed. All other local models and unpublished work remain preserved locally.

## Environment

Verified on Ubuntu x86_64 with Python 3.12.3, torch 2.14.1+cu130, NumPy 2.5.3, JSBSim 1.3.0, Gymnasium 1.3.0, stable-baselines3 2.9.0. The simulator runs on CPU. GPU physics is not an equivalent replacement. The measured smoke environment is recorded in [verification evidence](verification/seven_desktop_20261015.json).

From repository root:

```bash
cd aircombat-rl
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-portable.txt psutil
python -m pip install --no-deps -e .
python -m pip check
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

A CPU torch build is sufficient; numerical equivalence across library builds is not promised. Do not copy another machine's virtual environment. If the specified packages are unavailable on the destination, preserve the error and resolve the environment before claiming reproduction.

## Verify original bytes and run models

Run from `aircombat-rl` after environment activation. This verifies the entire bundle and plays two loader smoke games per model. These fixed conditions are consumed reproducibility checks, not fresh quality evidence. Each inference export embeds its teacher implementation and parameters.

```bash
python - <<'PYTHON'
from pathlib import Path
import hashlib, json
from experiments.aim_async.evaluate import cell
root = Path.cwd().parent
manifest = json.loads((root / 'artifacts/20261015/manifest.json').read_text())
for name, digest in manifest['sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
for model in manifest['models']:
    design = root / model['checkpoint'] / 'sampled_4900'
    own = dict(id=model['id'], kind='submission', design=str(design), weights=str(design/'policy_net.zip'))
    for seat in ('red', 'blue'):
        row = cell(own, dict(id='ace', kind='bot', name='ace'), 1801500000, seat)
        print(model['id'], seat, row['result'], row['own_health'], row['opp_health'])
PYTHON
```

## Authorized next league

Original seven opponents remain fixed and are never replaced. Train and evaluate separately. Candidate screen: 8 initial conditions, both seats, 2 action replicas, 224 games for seven opponents. Per opponent: any loss is failure; no losses and at least one win is success; all draws is failure. A complete screen pass triggers fresh, disjoint 16-condition confirmation (448 games). Only a complete confirmation pass adds the candidate to the pool. Future panels grow with the pool. Preserve every candidate, raw cell, training amount, game count and wall time. Never reuse a consumed confirmation panel as unseen. Passing finite panels is not a guarantee of victory under all conditions.

The user approved this protocol on 2026-10-15. Publication precedes the new Goal. This publication does not authorize any later GitHub upload. Subsequent training and results stay local until explicitly requested.

## Weight and setting inventory

[Historical weight and setting changes](HISTORICAL_WEIGHT_AND_SETTING_CHANGES_20261015.md) is included byte-for-byte from the local report requested by the user. It inventories 122 plans and identifies 90 runs with positive-step checkpoints; these counts do not imply completion or success. Its absolute links describe the source desktop. Only the seven selected models and accompanying evidence are bundled in this publication; other referenced local checkpoints remain preserved locally. For portable model locations use the bundle manifest above.

## Size control

Only files listed by `artifacts/20261015/manifest.json` are published from the new artifact directory. Temporary experiments, the complete training archive and unselected models are excluded. Existing repository history and older published archives are retained. Full historical training resumption requires additional local inputs and is not promised by this compact inference bundle.

## Publication verification

The compact artifact manifest covers 151 files / 3,450,005 bytes (plus manifest). An isolated staged snapshot verified every digest and ran all seven sampled exports in both seats: 14 native CPU games, exit 0. Earlier both-sampler loader checks completed 28 games. Seven applicable local policy test modules passed 42 tests. [Isolated snapshot evidence](verification/seven_minimal_snapshot_20261015.json). These are reproducibility checks, not model-selection results. The weight/settings inventory retains its original bytes, including its trailing blank line.
