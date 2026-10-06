"""Two submitted policy folders, official FairFight, the same seeds in both seats."""
from argparse import ArgumentParser
from pathlib import Path
from tools.autolab_cem import write,sha
from tools.policies import default_weights
from tools.league_matches import duel


def run(a,b,out,band,n):
    if out.exists():
        raise FileExistsError('Use a new match output directory')
    specs=[]
    inputs={}
    for identity,folder in (('policy_a',a),('policy_b',b)):
        weights=default_weights(folder)
        spec=dict(id=identity,kind='submission',design=str(folder),weights=str(weights))
        specs.append(spec)
        for p in [weights]+[folder/name for name in ('policy.py','wrappers.py','utils.py') if (folder/name).exists()]:
            inputs[str(p)]=sha(p)
    write(out/'plan.json',dict(entrants=specs,band=band,n=n,files_sha256=inputs,
        scope='Official physics and verdict, paired seats; this is a development match, not a sealed tournament test'))
    result=duel(specs[0],specs[1],band,n)
    for path,digest in inputs.items():
        if sha(Path(path))!=digest:
            raise ValueError(f'Policy input changed during the match: {path}')
    write(out/'result.json',result)
    s=result['summary']
    print(f"A versus B: {s['wins']} wins / {s['draws']} draws / {s['losses']} losses in {s['n']} games")
    print(f"A: {a}\nB: {b}")


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--a',required=True,type=Path)
    parser.add_argument('--b',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--band',type=int,default=33020000)
    parser.add_argument('--n',type=int,default=40)
    args=parser.parse_args()
    run(args.a.resolve(),args.b.resolve(),args.out.resolve(),args.band,args.n)
