"""Original training environment with observation-only threat gate accounting."""
from experiments.aim_async.terminal_preference_worker import TerminalPreferenceEnv
from experiments.aim_async.threat_metrics import ThreatMetricsEnv,METRIC_NAMES as WEZ_METRIC_NAMES
from experiments.aim_async.wez_or_damaged_metrics import WezOrDamagedMetricsEnv,METRIC_NAMES
from experiments.aim_async.early_range_metrics import EarlyRangeMetricsEnv,METRIC_NAMES as EARLY_METRIC_NAMES
from experiments.aim_async.range_metrics import RangeMetricsEnv,METRIC_NAMES as RANGE_METRIC_NAMES
from experiments.aim_async.range_numpy import experiment_range
from experiments.league.tail_curriculum import CurriculumArchive
from experiments.rear_async.worker import Restartable


def make_worker(plan,index,restart=None):
    threshold=experiment_range(plan)
    import torch
    from stable_baselines3.common.monitor import Monitor
    from tools.league_matches import Actor
    torch.set_num_threads(1)
    archive=CurriculumArchive(plan['teacher'],plan['opponents'],plan['probabilities'],
        plan['seed']*100+index,plan['ic_band'],index,plan['workers'],
        'red' if index%2==0 else 'blue',coverage_period=plan['coverage_period'],curriculum=plan['curriculum'])
    for spec in plan['opponents']:archive.cache[spec['id']]=Actor(spec)
    wrapped=TerminalPreferenceEnv(archive,plan['ppo']['gamma'],plan['aim_coefficients'],
        plan['relative_potential_coefficient'],plan['control_guidance_coefficient'],
        win_bonus=plan['terminal_win_bonus'],draw_penalty=plan['terminal_draw_penalty'])
    env=Restartable(Monitor(RangeMetricsEnv(EarlyRangeMetricsEnv(WezOrDamagedMetricsEnv(ThreatMetricsEnv(wrapped))),range_threshold_m=threshold),info_keywords=(
        'verdict_rule','result','terminal_reason','damage_dealt','damage_received',
        'connected_fire_seconds','lock_breaks','first_damage_time','first_received_time',
        'first_shot_advantage','relative_shaping_total','control_guidance_total',
        'terminal_preference_total',*WEZ_METRIC_NAMES,*METRIC_NAMES,*EARLY_METRIC_NAMES,*RANGE_METRIC_NAMES)))
    if restart is not None:env.restore_worker(restart)
    return env
