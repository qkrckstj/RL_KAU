"""Offline empirical maximin grid; not a deployed or heldout-tested policy."""
from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from tools import league_thread_benchmark as io


def analyze(source, out):
    if out.exists():raise FileExistsError(out)
    raw_path=source/'raw_expanded_comparison.json';raw=io.read(raw_path)
    assert raw['status']=='expanded_comparison_raw_verified'
    for p,h in raw['input_sha256'].items():assert io.sha(io.ROOT/p)==h
    a=raw['analysis'];roles=list(a['roles']);foes=list(a['roles'][roles[0]]['opponents'])
    assert len(roles)==5
    score=np.array([[(a['roles'][r]['opponents'][f]['wins']+.5*a['roles'][r]['opponents'][f]['draws'])/a['roles'][r]['opponents'][f]['games'] for r in roles] for f in foes])
    wins=np.array([[a['roles'][r]['opponents'][f]['wins']/a['roles'][r]['opponents'][f]['games'] for r in roles] for f in foes])
    floor=a['roles']['baseline']['mean_win_rate']
    weights=np.array([(i,j,k,l,20-i-j-k-l) for i in range(21) for j in range(21-i) for k in range(21-i-j) for l in range(21-i-j-k)],dtype=float)/20
    assert len(weights)==10626 and np.allclose(weights.sum(1),1.) and (weights>=0).all()
    means=weights@wins.mean(0);tails=(score@weights.T).min(0)
    valid=np.flatnonzero(means>=floor-1e-12);best=max(valid,key=lambda i:(tails[i],means[i]));w=weights[best];s=score@w
    result=dict(status='offline_grid_feasibility_only',source=io.relative(source),
        source_raw_sha256=io.sha(raw_path),source_code_sha256=io.sha(Path(__file__).resolve()),
        roles=roles,weights=dict(zip(roles,w.tolist())),grid_step=.05,grid_size=len(weights),
        worst_expected_score=float(s.min()),mean_expected_win=float(means[best]),mean_win_constraint=floor,
        pure_worst_expected_scores={role:float(score[:,i].min()) for i,role in enumerate(roles)},
        weakest=[dict(foe=foes[i],score=float(s[i])) for i in np.argsort(s)[:8]],
        scope='Grid search of consumed102M empirical expected win+.5draw,mean-win>=baseline,5percent weight increments.Not a globally optimal linear-program solution.Each stochastic expert averaged over two fixed action replicas.No new matches,no heldout claim,no mixture policy exported/deployed,no training modification.Optimized and measured on same matrix,so optimistic feasibility only;fresh-condition execution required.Official verdicts unchanged.')
    io.write(out,result);return result


if __name__=='__main__':
    p=ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=analyze(a.source.resolve(),a.out.resolve());print(r['weights'],r['worst_expected_score'],r['mean_expected_win'])
