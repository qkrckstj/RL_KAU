"""Freeze one policy using development results only, with a standard-loadable archive."""
from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import zipfile
from tools.autolab_cem import read,write,sha


ARCHIVE_ADAPTER = '''

# The standard loader discovers .zip weights. This archive contains JSON
# controller coefficients, not an SB3 neural network. Inference is unchanged.
_CoefficientPolicy = Policy
class Policy(_CoefficientPolicy):
    def __init__(self, weights=None, device="cpu", parameters=None):
        if parameters is None:
            import zipfile
            if zipfile.is_zipfile(weights):
                with zipfile.ZipFile(weights) as archive:
                    parameters = json.loads(archive.read("parameters.json"))["parameters"]
            else:
                parameters = json.loads(Path(weights).read_text(encoding="utf-8"))["parameters"]
        super().__init__(parameters=parameters, device=device)
'''


def run(replication,out):
    if out.exists():
        raise FileExistsError("Refusing to replace an exported policy")
    plan=read(replication/"plan.json")
    records=[read(replication/f"s{s}/result.json") for s in plan["seeds"]]
    def key(r):
        m=r["best_validation"]
        return m["kills"],m["own_health"]-m["opp_health"],-r["seed"]
    selected=max(records,key=key)
    source=replication/f"s{selected['seed']}"
    out.mkdir(parents=True)
    (out/"policy.py").write_text((source/"policy.py").read_text(encoding="utf-8")+ARCHIVE_ADAPTER,encoding="utf-8")
    for name in ("wrappers.py","policy_net.json"):
        shutil.copyfile(source/name,out/name)
    with zipfile.ZipFile(out/"policy_net.zip","w",compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("parameters.json",(source/"policy_net.json").read_bytes())
    (out/".gitignore").write_text("!policy_net.zip\n",encoding="utf-8")
    write(out/"selection.json",dict(seed=selected["seed"],generation=selected["best_generation"],
        development_band=plan["validation_band"],development=selected["best_validation"],source=str(source),
        rule="Development wins, then health advantage, then smaller seed; heldout never read by this script",
        source_policy_sha256=sha(source/"policy.py"),source_parameters_sha256=sha(source/"policy_net.json"),
        archive_sha256=sha(out/"policy_net.zip"),policy_sha256=sha(out/"policy.py"),
        format="ZIP containing JSON coefficients; no neural weights or torch dependency"))
    (out/"README.md").write_text('''# FairFight CEM 정책

공개39관측으로 선회·목표 속도·추적 전환을 결정하는7개 파라미터 정책이다.
경기 결과로 값을 학습했다. DQN/PPO 신경망 정책과 구분한다.
원본 JSBSim 물리, Ace 상대, 무장,9행동,20Hz 주기와 공식 승패 판정을 사용한다.
선택 기준과 출처는 `selection.json`, 파라미터는 `policy_net.json`에 있다.
`policy_net.zip`은 JSON 파라미터를 담은 압축 파일로 공식 정책 로더가 바로 발견한다.

`aircombat-rl` 폴더에서 기존 Python 환경으로 실행:

```powershell
python -m tools.watch templates/project_04_fair --design experiments/plan_a/cem_policy --seed 904000
```

개발 조건 점검(최종 시험을 반복 튜닝에 사용하지 않음):

```powershell
python -m tools.grade templates/project_04_fair --design experiments/plan_a/cem_policy --band 906000 --n 40 --out experiments/plan_a/cem_policy/development_check.json
```

최종 검증과 한계는 상위 폴더의 `autolab.md`와 완료 보고서를 참고한다.
미세조정 반복들은 같은 학습된 부모를 공유하며, 독립 최초 학습6회로 세지 않는다.
''' ,encoding="utf-8")
    print(json.dumps(dict(exported=str(out),seed=selected["seed"],generation=selected["best_generation"])))


if __name__=="__main__":
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--replication",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    run(args.replication.resolve(),args.out.resolve())
