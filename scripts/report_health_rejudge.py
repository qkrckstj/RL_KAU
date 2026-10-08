"""Summarize completed historical health rejudgment without pooling panels."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'docs/health_rejudge_round4_20261008'


def report():
    data = json.loads((OUT/'all_records/summary.json').read_text('utf-8'))
    latest = json.loads((OUT/'latest_panels.json').read_text('utf-8'))
    with (OUT/'all_records/file_inventory.csv').open(encoding='utf-8-sig', newline='') as src, \
         (OUT/'file_win_rates.csv').open('w', encoding='utf-8-sig', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames+['old_win_rate', 'new_win_rate', 'rate_status'])
        writer.writeheader()
        for row in reader:
            n = int(row['records']); invalid = int(row['invalid'])
            row.update(old_win_rate=int(row['old_win'])/n,
                       new_win_rate='' if invalid else int(row['win'])/n,
                       rate_status='invalid_records_require_review' if invalid else 'record_file_only_not_independent_experiment')
            writer.writerow(row)
    totals = data['inventory_totals']
    lines = ['# 종료 체력 기준 재판정 결과 — 2026-10-08', '',
      '**기준: 양쪽 종료 체력을 소수점 네 자리로 반올림하고 높은 쪽 승리, 같은 값은 무승부.**', '',
      '## 확인 범위', '',
      f"- 로컬 JSON {data['scanned_json_files']:,}개를 검사했다. 소스 사본·복원용 checkout·replay_snapshot은 제외했다.",
      f"- 경기 기록이 있는 파일 {totals.get('files_with_records',0):,}개, 발견한 기록 {totals.get('records',0):,}건을 재판정했다.",
      f"- 파일 읽기/해석 오류 {len(data['errors'])}개, 유효하지 않은 체력 기록 {totals.get('invalid',0)}건.",
      '- 파일별 과거/새 승률은 `file_win_rates.csv`에 있다. 같은 경기의 복제·재생·개발/시험 집계가 섞일 수 있어 전체 기록을 하나의 승률로 합치지 않는다.',
      '- JSON 원시 체력이 없는 기록, 다른 컴퓨터에만 있는 기록, 제외한 복제 폴더는 전체 실험의 정확한 재채점이 완료됐다고 주장하지 않는다.', '',
      '## 최신 동일 조건 평가', '',
      '| 패널 | 모델 | 경기 수 | 이전 승률 | 새 승/무/패 | 새 승률 |',
      '|---|---|---:|---:|---:|---:|']
    for row in latest['rows']:
        panel = '112M' if 'inactive' in row['panel'] else '113M'
        name = 'PPO7400' if row['role']=='repeat' else 'CEM6800' if row['role']=='cem' else 'compact7800' if panel=='112M' else 'rotated7801'
        lines.append(f"|{panel}|{name}|{row['records']}|{100*row['old_win_rate']:.2f}%|{row.get('win',0)} / {row.get('draw',0)} / {row.get('loss',0)}|{100*row['win_rate']:.2f}%|")
    lines += ['', '112M와113M는 상대·초기조건이 다르다. PPO의 두 행동 난수와 CEM의 한 결정적 행동을 독립 학습 반복으로 세지 않는다. 동일7400의 repeat/baseline 별칭은 한 번만 표시했다.', '',
      '## 다음 판단', '',
      '- compact7800은112M에서7400보다2.86%p 낮아 추가 학습 우선순위가 낮다.',
      '- rotated7801은113M에서7400보다3.91%p 높다. 유망 후보로 남기되 구판정 신뢰구간을 재사용하거나 확정 승격하지 않는다.',
      '- 같은113M의CEM6800은70.31%다. 다음 공통 조건 비교에는 CEM도 포함해 최고 PPO와 전체 최선 정책을 구분한다.',
      '- 새 기준용 학습 환경·평가·보상 통합은 아직 구현/실행하지 않았다. 학습은 정지 상태다.',
      '- 반올림 동률 재생은 사용자 지시에 따라 중단했다. 그 부분 재생의 고정밀 값으로 동률을 뒤집지 않는다.', '',
      '계획서: [적응형 자동 학습 계획](../ADAPTIVE_TRAINING_PLAN_HEALTH_V1.md). GitHub에 자동 업로드하지 않는다.', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(scanned=data['scanned_json_files'], files=totals.get('files_with_records'), records=totals.get('records'), errors=len(data['errors']), invalid=totals.get('invalid',0)), ensure_ascii=False))


if __name__ == '__main__':
    report()
