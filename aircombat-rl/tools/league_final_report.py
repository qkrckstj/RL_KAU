"""Report the policy frozen before evaluation; never rank by final test results."""
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
import json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def totals(rows):
    counts = Counter(e['outcome'] for r in rows for e in r['episodes'])
    n = sum(counts.values())
    wins, losses = counts['kill'], counts['died']
    draws = n - wins - losses
    return dict(n=n, wins=wins, draws=draws, losses=losses,
                win_rate=wins/n, score=(wins+.5*draws)/n, outcomes=dict(counts))


def wdl(record):
    return f"{record['wins']}승 {record['draws']}무 {record['losses']}패"


def run(root, out, plot=False):
    followup = root/'followup_pool'
    done = read(followup/'completion.json')
    if done['stage'] != 'heldout':
        raise ValueError('No completed final evaluation')
    freeze = read(followup/'frozen_selection.json')
    chosen = freeze['chosen']['id']
    rows = [read(p)['result'] for p in sorted((followup/'heldout').glob('match_*.json'))]
    foes = [p['id'] for p in freeze['opponents']]
    transfer = set(freeze['plan']['transfer_opponents'])
    selected_ids = [p['id'] for p in freeze['selected']]
    by = {(r['own'], r['foe']): r for r in rows}
    if len(by) != len(rows) or len(rows) != (len(selected_ids)+1)*len(foes):
        raise ValueError('Incomplete or duplicate evaluation matrix')
    models = {}
    for identity in ['cem_original']+selected_ids:
        models[identity] = {name: totals([by[identity, f] for f in group])
            for name, group in [('all', foes), ('archive', [f for f in foes if f not in transfer]),
                                ('excluded', [f for f in foes if f in transfer])]}
    peers = read(followup/'refinement_crossplay.json')
    runner = read(root/'runner_probe/completion.json') if (root/'runner_probe/completion.json').exists() else None
    runner_confirm = read(root/'runner_confirm/completion.json') if (root/'runner_confirm/completion.json').exists() else None
    result = dict(status=done['status'], chosen=chosen, models=models, verdict=done['verdict'],
        opponent_order=foes, per_opponent=[dict(foe=f, excluded=f in transfer,
            original=totals([by['cem_original', f]]), selected=totals([by[chosen, f]])) for f in foes],
        refinement_crossplay=[{k:r[k] for k in ('own','foe','summary')} for r in peers],
        runner_diagnostic=runner, runner_confirmation=runner_confirm,
        scope='Frozen development choice; same 40 ICs in both seats against 16 fixed local opponents. '
              'Three refinements share one learned parent. No real student submissions tested.')
    out.mkdir(parents=True, exist_ok=True)
    (out/'final_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    v=done['verdict']
    old,new=models['cem_original']['all'],models[chosen]['all']
    gain=(new['win_rate']-old['win_rate'])*100
    lines=[
        '# 다양한 정책을 상대하는 자동 학습 결과', '',
        f"최종 시험 전에 개발 기준으로 고른 정책은 **{chosen}**다. 같은 16상대·각 {new['n']:,}경기에서 "
        f"기존 CEM 승률 **{old['win_rate']:.1%} → 새 정책 {new['win_rate']:.1%}** ({gain:+.1f}%p)였다.",
        f"원시 결과는 기존 {wdl(old)}, 새 정책 {wdl(new)}다. 사전 평가 관문은 **{done['status']}**다.", '',
        '## 실제로 실행한 자동 반복', '',
        '6차례 후보 학습→기존 전체 상대와 대결→후보 보존을 실행했고 챔피언은 3번 교체됐다. '
        '교체되지 못한 후보도 다음 학습 상대에 추가했다. 이후 모든 후보를 동일 13상대로 재선발해 round_05를 부모로 고르고, '
        '난수 seed 2200/2201/2202로 각각 추가 학습했다. 최종 정책은 시험을 열기 전에 고정했다.', '',
        '현재 방식은 공개 관측에 반응하는 **17개 제어 계수를 CEM으로 탐색하는 학습**이다. '
        'DQN/PPO 신경망의 수렴을 달성한 결과는 아니다. 원래 정책과 6개 후보, 3개 미세조정 정책을 모두 보존했다.', '',
        '## 같은 최종 조건에서의 비교', '',
        '| 정책 | 전체 승률 | 보관 상대 13종 | 학습 제외 상대 3종 | 전체 승/무/패 |',
        '|---|---:|---:|---:|---|']
    for identity, groups in models.items():
        label=identity+(' **(사전 선택)**' if identity==chosen else '')
        lines.append(f"| {label} | {groups['all']['win_rate']:.1%} | {groups['archive']['win_rate']:.1%} | "
                     f"{groups['excluded']['win_rate']:.1%} | {wdl(groups['all'])} |")
    lo,hi=v['paired_ic_bootstrap_95_ci']
    lines += ['',f"추가 학습 3회의 평균 승률 개선은 {v['mean_win_gain']*100:+.1f}%p, "
              f"초기조건 단위 bootstrap 95% 구간은 [{lo*100:+.1f}, {hi*100:+.1f}]%p다. "
              '이는 선택 정책 하나의 신뢰구간이 아니라 3개 정책의 평균 개선에 대한 구간이다.', '',
              '모든 상대에 공통인 40개 초기조건×양 좌석=상대당 80경기다. '
              '총 경기 수만큼 독립 표본이 있는 것은 아니다. 구간은 고정된 정책·상대 집합에 조건부이며 '
              '독립적인 최초 학습의 변동성을 나타내지 않는다. 세 추가 학습은 같은 부모와 상대 집합을 공유한다.', '',
              '### 사전 선택 정책의 상대별 결과', '',
              '| 상대 | 기존 CEM 승/무/패 | 새 정책 승/무/패 | 새 정책 승률 |',
              '|---|---|---|---:|']
    for r in result['per_opponent']:
        label=r['foe']+(' (학습 제외)' if r['excluded'] else '')
        lines.append(f"| {label} | {wdl(r['original'])} | {wdl(r['selected'])} | {r['selected']['win_rate']:.1%} |")
    lines += ['', '![같은 최종 시험에서 기존 정책과 새 정책 비교](results/final_comparison.png)', '',
              '### 추가 학습 정책끼리의 대결', '',
              '| 정책 A | 정책 B | A 관점 승/무/패 |', '|---|---|---|']
    for r in peers:
        lines.append(f"| {r['own']} | {r['foe']} | {wdl(r['summary'])} |")
    lines += ['', '이 대결은 최종 정책을 고른 뒤의 진단이다. 시험에서 더 좋아 보이는 모델로 선택을 바꾸지 않았다.', '',
              '## 도주형 상대 진단', '']
    if runner:
        lines += [f"선택 정책은 Evader에 {wdl(runner['parent'])}, 8개 수동 추격 설정 중 선택된 "
                  f"`{runner['best']['id']}`는 {wdl(runner['best_summary'])}였다. "
                  '별도의 공개 개발 조건 6개×양 좌석에서 확인한 진단이며, 이 8개는 새로 학습한 모델이 아니다.',
                  f"평균 최소 거리는 선택 정책 {runner['parent']['min_range']:,}m, "
                  f"진단에서 선택된 설정 {runner['best_summary']['min_range']:,}m였다. "
                  '이 유한한 실험으로 격추 가능/불가능을 일반적으로 증명할 수는 없다.', '']
    else:
        lines += ['추가 진단은 아직 완료되지 않았다. 결과가 나올 때 이 보고서를 갱신한다.', '']
    if runner_confirm:
        lo_runner,hi_runner=runner_confirm['paired_ic_win_gain_95_ci']
        lines += [f"선택된 수동 추격 설정을 새로운 개발 조건40개×양 좌석에서 재확인했다. 기존 정책은 "
                  f"{wdl(runner_confirm['parent'])}, 추격 설정은 {wdl(runner_confirm['probe'])}였다. "
                  f"초기조건 단위 승률 차이95% 구간은 [{100*lo_runner:+.1f}, {100*hi_runner:+.1f}]%p다.",
                  '이는 도주형 상대에 대한 개선 가능성을 확인한 결과다. 다른 상대에 대한 성능을 확인한 새 학습 정책이 아니므로 '
                  '기존 최종 모델을 교체하지 않았다. 다음 작업은 관측에 따라 추격을 선택하는 정책을 전체 상대 집합에서 학습·비교하는 것이다.', '']
    lines += ['## 해석과 다음 사용', '',
              '- 공식 FairFight 초기조건·JSBSim 물리·20Hz·무장·9행동·승패 판정을 유지했다. 대결 상대를 바꿔 평가했다.',
              '- 무승부 0.5는 개발 목적함수의 규칙이다. 공식 대회의 무승부 배점이라고 가정하지 않았다. '
              '최악 상대 점수 0.5는 최악 승률 50%를 뜻하지 않는다.',
              '- Lead·Circler·DDQN seed 0은 이번 리그 학습에서 제외한 로컬 상대다. 실제 다른 학생 모델을 이겼다는 증거는 아직 없다.',
              '- 연속 상태의 모든 상황을 경험할 수는 없다. 이번 검증은 공식 시작 조건과 고정된 상대 집합에 한정된다.',
              '- 다음 확장은 실제 참가자 정책을 보관 상대에 추가하고, 약점 상대를 반영해 다음 모델을 학습하는 것이다. '
              '예전 정책들도 계속 유지해 특정 상대만 이기는 전략으로 퇴행하는지 확인한다.',
              '- 시험 band 40000000은 이제 사용했다. 이 결과를 바탕으로 후속 학습하면 새 미래 시험 band를 따로 잡아야 한다.', '',
              '## 실행 파일과 재현', '',
              '최종 정책: `experiments/league/bundle/models/final/`. 원래 정책: `bundle/models/cem_original/`. '
              '묶음의 `manifest.json`에서 파일 해시와 평가 상태를 확인한다. [묶음 안내](bundle/README.md)에 실행 방법이 있다.', '',
              '```powershell',
              'python -m tools.league_duel --a experiments/league/bundle/models/final --b path/to/other_policy --out runs/next_opponent --band 33020000 --n 40',
              '```', '',
              '원시 평가·설정·선택 기록은 `runs/league_20261006/followup_pool/`, 이동용 근거는 '
              '`bundle/evidence/final/`에 보존한다. GitHub 보관 범위와 다음 작업은 저장소 루트의 '
              '`START_HERE.md`, `docs/LEAGUE_STATE.json`을 따른다.', '']
    (out.parent/'result.md').write_text('\n'.join(lines),encoding='utf-8')
    if plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        labels={'cem_original':'Original CEM', 'ddqn_s1':'DDQN seed 1','ddqn_s0':'DDQN seed 0',
                'ace':'Ace','pursuit':'Pursuit','evader':'Evader','fixed_left':'Fixed left',
                'fixed_right':'Fixed right','lead':'Lead','circler':'Circler'}
        names=[labels.get(f,f.replace('_',' '))+(' *' if f in transfer else '') for f in foes]
        fig,axes=plt.subplots(1,2,figsize=(13,9),layout='constrained',sharey=True)
        y=np.arange(len(foes))
        for ax,kind,title in zip(axes,('win_rate','draw_rate'),('Win rate','Draw rate')):
            for identity,offset,color,label in [('cem_original',-.18,'#9aa5b1','Original CEM'),(chosen,.18,'#117c83',chosen+' (frozen choice)')]:
                vals=[(totals([by[identity,f]])['win_rate'] if kind=='win_rate'
                       else totals([by[identity,f]])['draws']/80)*100 for f in foes]
                ax.barh(y+offset,vals,height=.34,color=color,label=label)
                for yi,val in zip(y+offset,vals):
                    ax.text(val+1,yi,f'{val:.0f}',va='center',fontsize=8,color='#23313d')
            ax.set(xlim=(0,112),xticks=[0,25,50,75,100],xlabel='Percent',title=title,yticks=y,yticklabels=names)
            ax.axhline(12.5,color='#99a4b0',lw=1)
            ax.grid(axis='x',alpha=.15)
            ax.set_axisbelow(True)
        axes[0].invert_yaxis()
        handles,legend_labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,legend_labels,loc='outside lower center',ncol=2,frameon=False,fontsize=10)
        fig.suptitle('Final evaluation | Same 40 initial conditions x both seats per opponent\n'
                     '* Excluded from league training; local opponents, not student submissions',fontsize=13)
        fig.savefig(out/'final_comparison.png',dpi=170)
        plt.close(fig)
    print(json.dumps(dict(status=done['status'],chosen=chosen,original=old,new=new),ensure_ascii=False))


if __name__=='__main__':
    p=ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('runs/league_20261006'))
    p.add_argument('--out',type=Path,default=Path('experiments/league/results'))
    p.add_argument('--plot',action='store_true')
    a=p.parse_args()
    run(a.run,a.out,a.plot)
