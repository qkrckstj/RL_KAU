"""Explicit new-study reference binding without altering learned payloads."""
import hashlib
import json
import zipfile
from pathlib import Path


def rebind_reference_archive(source, destination, expected_old_hash):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError('Reference migration requires a fresh destination')
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Ambiguous source ZIP members')
        payloads = {name: archive.read(name) for name in names}
        infos = archive.infolist(); comment = archive.comment
    data = json.loads(payloads['data'])
    if data.get('reference_policy_hash') != expected_old_hash or data.get('reference_coefficient', 0.) <= 0.:
        raise ValueError('Source reference binding differs from the declared prior study')
    policy_hash = hashlib.sha256(payloads['policy.pth']).hexdigest()
    optimizer_hash = hashlib.sha256(payloads['policy.optimizer.pth']).hexdigest()
    if policy_hash == expected_old_hash:
        raise ValueError('Source is already anchored to its own policy')
    migrated = dict(data, reference_policy_hash=policy_hash, reference_archive=str(destination))
    new_data = json.dumps(migrated, indent=4).encode()
    with zipfile.ZipFile(destination, 'x') as archive:
        archive.comment = comment
        for info in infos:
            archive.writestr(info, new_data if info.filename == 'data' else payloads[info.filename])
    with zipfile.ZipFile(destination) as archive:
        exact = archive.namelist() == names and all(archive.read(name) == payloads[name] for name in names if name != 'data')
    if not exact:
        raise ValueError('Reference migration altered a learned payload')
    return dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), prepared_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                policy_sha256=policy_hash, optimizer_sha256=optimizer_hash, old_reference_policy_sha256=expected_old_hash,
                new_reference_policy_sha256=policy_hash, changed_data_fields=['reference_archive', 'reference_policy_hash'],
                non_data_payloads_exact=exact, source_num_timesteps=data['num_timesteps'], source_n_updates=data['_n_updates'])
