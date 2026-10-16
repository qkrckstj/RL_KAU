"""Reuse exact native cells across panels of the same immutable checkpoint."""
from pathlib import Path
from experiments.rear_async.storage import identity,read,sha,verify


CONDITIONS=('opponent','seed','seat','action_seed')


def shared_cell(out,job,metadata,key):
    if (metadata['checkpoint_id'],metadata['config_id'])!=(job['checkpoint_id'],job['config_id']):
        raise ValueError('Target checkpoint identity differs')
    for completion in sorted((out/'evaluations').glob('*/completion.json')):
        entry=read(completion)
        if entry['baseline'] or entry['checkpoint_id']!=job['checkpoint_id']:continue
        source_input=completion.parent/'input'
        original=verify(source_input)
        if (original['checkpoint_id'],original['config_id'],original['files'])!=(
                metadata['checkpoint_id'],metadata['config_id'],metadata['files']):
            raise ValueError('Shared checkpoint identity differs')
        source_file=completion.parent/'cells'/f'{identity(key)}.json'
        if not source_file.exists():continue
        row=read(source_file)
        if {k:row[k] for k in CONDITIONS}!=key:
            raise ValueError('Shared cell condition differs')
        return row,dict(source_file=str(source_file.resolve()),source_input=str(source_input.resolve()),
                        source_sha256=sha(source_file),checkpoint_id=metadata['checkpoint_id'],
                        config_id=metadata['config_id'],model_files=metadata['files'],conditions=key)
    return None


def reuse_counts(folder,rows,new_this_attempt):
    metadata=verify(folder/'input')
    manifest=read(folder/'reuse_manifest.json') if (folder/'reuse_manifest.json').exists() else {}
    by_key={identity({k:r[k] for k in CONDITIONS}):r for r in rows}
    if len(by_key)!=len(rows) or not set(manifest).issubset(by_key):
        raise ValueError('Reuse manifest/count conditions differ')
    for key,proof in manifest.items():
        source_file,source_input=Path(proof['source_file']),Path(proof['source_input'])
        if source_input.resolve().parent.parent!=folder.resolve().parent or source_file.resolve().parent!=source_input.resolve().parent/'cells':
            raise ValueError('Reuse source outside sibling evaluation')
        original=verify(source_input)
        if (original['checkpoint_id'],original['config_id'],original['files'])!=(
                metadata['checkpoint_id'],metadata['config_id'],metadata['files']):
            raise ValueError('Reuse checkpoint identity differs')
        if (proof['checkpoint_id'],proof['config_id'],proof['model_files'])!=(
                metadata['checkpoint_id'],metadata['config_id'],metadata['files']):
            raise ValueError('Reuse proof identity differs')
        target=folder/'cells'/f'{key}.json'
        if sha(source_file)!=proof['source_sha256'] or sha(target)!=proof['source_sha256']:
            raise ValueError('Changed reused cell')
        if {k:by_key[key][k] for k in CONDITIONS}!=proof['conditions'] or read(target)!=by_key[key]:
            raise ValueError('Reuse row condition differs')
    reused=len(manifest)
    if not 0<=new_this_attempt<=len(rows)-reused:raise ValueError('Invalid new native game count')
    return dict(actual_new_games=len(rows)-reused,new_games_in_final_attempt=new_this_attempt,
                reused_cells_in_final_attempt=len(rows)-new_this_attempt,
                reused_from_sibling_games=reused)
