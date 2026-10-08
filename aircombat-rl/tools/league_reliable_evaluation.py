"""Same match requests and resumable records, with reliable status writes."""
from concurrent.futures import as_completed
from tools.autolab_cem import read
from tools.league_reliable_io import write
from tools.league_train import metrics
from tools.league_matches import duel


def evaluate_candidates(pool,candidates,opponents,band,n,out,weights=None):
    out.mkdir(parents=True,exist_ok=True)
    futures,records,requests,partial = {},{},{},{}
    for i,spec in enumerate(candidates):
        request = dict(spec=spec,opponents=opponents,band=band,n=n,weights=weights)
        path = out/f'candidate_{i:03d}.json'
        if path.exists():
            record = read(path)
            if record['request'] != request:
                raise ValueError('Cannot resume a changed candidate schedule')
            records[i] = record
        else:
            requests[i]=request
            partial[i]={}
            for j,foe in enumerate(opponents):
                match_path=out/f'candidate_{i:03d}_matches/foe_{j:03d}.json'
                if match_path.exists():
                    prior=read(match_path)
                    if prior['request']!=request:
                        raise ValueError('Cannot resume changed opponent schedule')
                    partial[i][j]=prior['result']
                else:
                    futures[pool.submit(duel,spec,foe,band,n)]=(i,j,match_path)
    for future in as_completed(futures):
        i,j,path=futures[future]
        partial[i][j]=future.result()
        write(path,dict(request=requests[i],result=partial[i][j]))
        write(out/'progress.json',dict(matches_done=sum(len(v) for v in partial.values())+len(records)*len(opponents),
                                      matches_total=len(candidates)*len(opponents)))
    for i,values in partial.items():
        results=[values[j] for j in range(len(opponents))]
        record=dict(request=requests[i],results=results,metrics=metrics(results,weights))
        write(out/f'candidate_{i:03d}.json',record)
        records[i]=record
    write(out/'progress.json',dict(done=len(records),total=len(candidates)))
    return [records[i] for i in range(len(candidates))]

