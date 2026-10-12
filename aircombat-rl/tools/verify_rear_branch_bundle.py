"""Verify portable selected policies and replay the two published rear wins."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import hashlib,json,math
from pathlib import Path
from tools.league_matches import MatchEnv


def main():
    root=Path(__file__).resolve().parents[1]
    bundle=root/'experiments/league/rear_branch_20261012'
    manifest=json.loads((bundle/'manifest.json').read_text('utf-8'))
    for file,digest in json.loads((bundle/'SHA256.json').read_text('utf-8')).items():
        assert hashlib.sha256((bundle/file).read_bytes()).hexdigest()==digest,file
    records=[]
    for game in manifest['games']:
        env=MatchEnv(game['own'],game['foe'],game['seat'])
        try:
            obs,info=env.reset(seed=game['seed']);hp=float(info['opp_health']);rear=0.
            while True:
                obs,_,term,trunc,info=env.step(env.own_action(obs))
                dx,dy=float(obs[15]-obs[0]),float(obs[16]-obs[1]);distance=math.hypot(dx,dy)
                tail=(dx*math.sin(float(obs[26]))+dy*math.cos(float(obs[26])))/max(distance,1e-9)
                eligible=tail>=math.cos(math.pi/4) and bool(obs[34])
                rear+=max(0.,hp-float(info['opp_health']))*eligible;hp=float(info['opp_health'])
                if term or trunc:break
            for key in ('own_health','opp_health'):assert float(info[key])==game[key]
            assert round(info['own_health'],4)>round(info['opp_health'],4)
            assert abs(rear-game['rear_damage'])<1e-12
            assert float(env._combat.t)==game['duration']
            records.append(dict(label=game['label'],rear_damage=rear,own_health=info['own_health'],opp_health=info['opp_health']))
        finally:env.close()
    print(json.dumps(dict(status='published_bundle_two_exact_wins_verified',games=records),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
