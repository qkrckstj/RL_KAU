"""Inventory live run metadata without opening actively replaced JSON files."""
from argparse import ArgumentParser
from datetime import datetime,timezone
from pathlib import Path
import json


def snapshot(folder):
    root=Path(folder)
    searches=[]
    for search in sorted(root.glob('s[0-9]*')):
        if not search.is_dir(): continue
        generations=sorted((p for p in search.glob('g*') if p.is_dir() and p.name[1:].isdigit()),
                           key=lambda p:int(p.name[1:]))
        batches=[]
        if generations:
            for stage in ('screen','confirm','development'):
                path=generations[-1]/stage
                if not path.exists(): continue
                matches=list(path.glob('fresh_*/candidate_*_matches/foe_*.json'))
                newest=max((p.stat().st_mtime for p in matches),default=None)
                batches.append(dict(stage=stage,saved_match_jobs=len(matches),
                    candidate_records=len(list(path.glob('candidate_*.json'))),
                    last_match_at=datetime.fromtimestamp(newest,timezone.utc).isoformat() if newest else None))
        searches.append(dict(search=search.name,completed_generation_markers=len(list(search.glob('g*/state.json'))),
            result_present=(search/'result.json').exists(),latest_generation=generations[-1].name if generations else None,
            batches=batches))
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),run=str(root),searches=searches,
        completion_present=(root/'completion.json').exists(),failure_present=(root/'failure.json').exists(),
        scope='File metadata only; no actively overwritten JSON file was opened. Pair with actual process identity and immutable completed results.')


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('folder',type=Path)
    print(json.dumps(snapshot(parser.parse_args().folder),indent=2))
