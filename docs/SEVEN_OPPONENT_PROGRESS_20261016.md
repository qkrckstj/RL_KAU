# 7개 상대 학습·평가 전체 경과 — 2026-10-16

현재 인계 상태는 **paused**다. 이번 게시를 위해 학습을 재개하지 않는다. 완료한 신경망 PPO 학습과 이후 직접 조종·코드 제어 실험을 구분한다. 원본 7개는 교체하지 않았고 추가 합격 모델도 없다. **v1은 아직 없다.**

## 변경하지 않는 평가 기준

상대별 8개 공통 초기조건(3401500000–3401500007), 양 좌석(red/blue), 상대 행동 난수 4900/4901: 7×8×2×2=224경기. 상대전적은 한 번이라도 패배하면 패배, 패배 없이 최소 1승이면 승리, 전부 무승부면 패배다. 개별 경기 판정은 종료 체력 소수점 네 자리 반올림 비교(`user_terminal_health_round4_v2`)다. 물리·무장·초기조건 생성기는 보존한다.

7개 모두 screen 통과한 **동일한 고정 후보**만 별도 미사용 16개 초기조건·양 좌석·동일 2행동 난수(448경기)로 재검증한다. 조건·모델·풀 identity를 고정하고 전부 통과해야 풀에 추가한다. 재검증 조건은 사용 전 기존 reservation/소비 기록과 대조해야 한다. 사용한 screen은 적응적 개발 조건이며 unseen이 아니다. 유한 평가의 전승을 모든 상황에서의 승리 보장으로 표현하지 않는다.

## CPU PPO 학습 방식

기존 CPU PPO의 actor/critic/Adam을 보존하여 이어서 학습하고 NumPy 추론 제출물을 내보냈다. 학습 단계와 평가 단계는 분리했다. 최근 설정의 학습은 8개 물리 프로세스/32개 가상 환경을 사용했다. 각 상대는 주기적으로 모두 포함하고, 이전 결과에 따른 nonwin/loss 가중치로 배분을 조절했다. 초기 122개 계획은 설계 참고자료이며 그대로 복제한 것이 아니다.

변경한 축은 LR·학습량·엔트로피·피해 비용·상대 배분, GAE/참조 정책 정규화, 할인율, 무승부 손실·승리 보상, rollout 길이, 위협/거리/피격 후 조건부 prior 등이다. gamma는 후반 분기에서 .9995, GAE lambda .999, 승리 추가 보상 1.0, 무승부 추가 패널티 .25를 시험했다. 각 행의 settings에 실제 설정과 원본 config SHA-256이 있다. 다른 seed/부모/순차 조정이 섞인 결과를 단일 인자의 인과 효과로 주장하지 않는다.

닫힌 production **92후보 / 458,752,000결정 / 332,964완료 학습 에피소드 / 51287.269324초 학습**. Production 평가는 30,688경기(그중 baseline 10,080), 5112.856237초. 별도 진단/중간 평가를 포함한 기존 standard 평가 누계는 **34,272경기**다. 두 수는 중복 합산하지 않는다. Smoke·직접 조종·게시 검증은 별도다.

보존한 신경망 대표는 s61500(최선 strict 3/7,14승202무8패; 45800·46201·47236 통과)와 마지막 완료 s75501(0/7,29승176무19패)이다. 3/7 동률 후보들이 있으며, 게임 승수와 strict 상대 통과 수는 다른 지표다. 가중치·optimizer·worker/RNG 스냅샷을 함께 제공하지만 옛 프로세스의 정확한 재개는 보장하지 않는다.

## 완료 후보 92개: 매 회 학습량·시간·상대별 결과

셀은 승/무/패이며 ✓는 해당 상대 strict 통과다. 모든 행은 224 screen 경기다. 링크된 result는 원본 바이트이며, settings는 큰 frozen-source 표를 제외한 검사 전용 메타데이터다. 부모·추가 설정·평가 시간은 각 result/settings에서 확인한다.

|실행 / 회차|학습 결정|학습 경기|학습 초|평가 초|통과|s44701|s45800|s47236|s47238|s46201|s46401|s46500|
|---|---:|---:|---:|---:|---:|---|---|---|---|---|---|---|
|[main_20261015 / 0](../artifacts/20261016/evidence/production/seven_league_main_20261015/iteration_000/result.json)|1310720|932|142.744|38.757|0/7|1/28/3|3/24/5|9/17/6|8/18/6|4/24/4|4/23/5|3/27/2|
|[main_20261015 / 1](../artifacts/20261016/evidence/production/seven_league_main_20261015/iteration_001/result.json)|1310720|921|137.539|38.462|1/7|5/26/1|2/28/2|8/17/7|10/16/6|8/23/1|8/24/0 ✓|1/28/3|
|[main_20261015 / 2](../artifacts/20261016/evidence/production/seven_league_main_20261015/iteration_002/result.json)|1310720|927|148.265|37.934|0/7|2/28/2|1/30/1|5/20/7|5/19/8|2/26/4|5/26/1|4/26/2|
|[main_20261015 / 3](../artifacts/20261016/evidence/production/seven_league_main_20261015/iteration_003/result.json)|1310720|923|141.419|38.948|0/7|3/26/3|5/25/2|7/21/4|7/22/3|5/25/2|3/25/4|2/27/3|
|[lowerlr_20261015 / 0](../artifacts/20261016/evidence/production/seven_league_lowerlr_20261015/iteration_000/result.json)|1310720|936|145.816|38.396|0/7|3/25/4|3/23/6|7/19/6|6/21/5|8/22/2|4/24/4|1/28/3|
|[lowerlr_20261015 / 1](../artifacts/20261016/evidence/production/seven_league_lowerlr_20261015/iteration_001/result.json)|1310720|910|141.848|38.184|1/7|4/26/2|3/28/1|9/18/5|10/18/4|2/23/7|6/22/4|3/29/0 ✓|
|[longer_20261015 / 0](../artifacts/20261016/evidence/production/seven_league_longer_20261015/iteration_000/result.json)|5242880|3724|553.335|38.211|0/7|2/27/3|4/27/1|8/21/3|8/21/3|5/23/4|9/22/1|2/29/1|
|[longer_20261015 / 1](../artifacts/20261016/evidence/production/seven_league_longer_20261015/iteration_001/result.json)|5242880|3720|559.613|38.916|0/7|5/26/1|5/24/3|9/18/5|10/17/5|4/24/4|5/25/2|1/30/1|
|[lossweight_20261015 / 0](../artifacts/20261016/evidence/production/seven_league_lossweight_20261015/iteration_000/result.json)|5242880|3688|548.215|38.781|0/7|8/19/5|4/23/5|8/21/3|7/22/3|6/24/2|5/24/3|1/26/5|
|[lossweight_20261015 / 1](../artifacts/20261016/evidence/production/seven_league_lossweight_20261015/iteration_001/result.json)|5242880|3784|548.071|37.518|1/7|3/26/3|2/27/3|7/20/5|9/19/4|2/24/6|6/24/2|5/27/0 ✓|
|[received075_20261015 / 0](../artifacts/20261016/evidence/production/seven_league_received075_20261015/iteration_000/result.json)|5242880|3736|549.070|37.241|0/7|3/25/4|1/27/4|5/23/4|4/24/4|2/28/2|3/28/1|5/26/1|
|[received075_20261015 / 1](../artifacts/20261016/evidence/production/seven_league_received075_20261015/iteration_001/result.json)|5242880|3703|550.530|38.235|0/7|4/25/3|4/25/3|4/23/5|5/22/5|3/26/3|3/27/2|4/26/2|
|[entropy010_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_entropy010_20261016/iteration_000/result.json)|5242880|3727|550.537|37.131|0/7|1/29/2|2/28/2|5/24/3|5/22/5|4/22/6|2/27/3|4/25/3|
|[entropy010_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_entropy010_20261016/iteration_001/result.json)|5242880|3745|547.782|36.931|1/7|3/28/1|0/29/3|4/24/4|4/27/1|1/31/0 ✓|1/29/2|3/27/2|
|[entropy_continue_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_entropy_continue_20261016/iteration_000/result.json)|5242880|3759|550.737|36.384|0/7|2/24/6|2/27/3|1/30/1|1/29/2|4/26/2|4/25/3|3/23/6|
|[entropy_continue_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_entropy_continue_20261016/iteration_001/result.json)|5242880|3780|547.951|37.439|3/7|1/26/5|3/29/0 ✓|2/29/1|3/28/1|3/29/0 ✓|3/26/3|1/31/0 ✓|
|[threepass_continue_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_threepass_continue_20261016/iteration_000/result.json)|5242880|3706|556.047|40.945|0/7|6/17/9|4/23/5|1/22/9|2/24/6|4/25/3|5/21/6|8/22/2|
|[threepass_continue_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_threepass_continue_20261016/iteration_001/result.json)|5242880|3696|560.285|43.355|0/7|6/20/6|10/13/9|4/20/8|6/18/8|4/25/3|6/18/8|4/22/6|
|[threepass_lowerlr_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_threepass_lowerlr_20261016/iteration_000/result.json)|5242880|3799|563.561|43.437|0/7|9/17/6|4/20/8|7/17/8|7/18/7|3/19/10|2/23/7|8/23/1|
|[threepass_lowerlr_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_threepass_lowerlr_20261016/iteration_001/result.json)|5242880|3726|556.656|43.182|0/7|8/16/8|5/17/10|6/19/7|3/22/7|9/19/4|7/21/4|3/22/7|
|[entropy003_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_entropy003_20261016/iteration_000/result.json)|5242880|3851|558.808|37.361|0/7|1/27/4|3/24/5|5/25/2|7/22/3|3/27/2|2/25/5|5/23/4|
|[entropy003_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_entropy003_20261016/iteration_001/result.json)|5242880|3861|561.911|35.965|0/7|1/30/1|1/28/3|2/29/1|4/27/1|2/27/3|1/26/5|1/30/1|
|[entropy000_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_entropy000_20261016/iteration_000/result.json)|5242880|3886|558.587|35.816|1/7|1/29/2|3/26/3|2/26/4|2/26/4|0/31/1|4/27/1|2/30/0 ✓|
|[entropy000_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_entropy000_20261016/iteration_001/result.json)|5242880|3823|560.240|36.466|1/7|4/28/0 ✓|2/29/1|0/29/3|1/27/4|4/25/3|1/27/4|2/27/3|
|[lossweight16_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_lossweight16_20261016/iteration_000/result.json)|5242880|3813|557.814|36.218|1/7|2/28/2|3/27/2|4/28/0 ✓|5/25/2|0/30/2|3/28/1|0/30/2|
|[lossweight16_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_lossweight16_20261016/iteration_001/result.json)|5242880|3863|558.536|36.627|1/7|0/30/2|0/29/3|5/27/0 ✓|4/27/1|3/26/3|0/31/1|1/30/1|
|[coverage4_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_coverage4_20261016/iteration_000/result.json)|5242880|3803|555.289|36.384|0/7|1/29/2|1/28/3|3/27/2|2/28/2|4/26/2|2/28/2|3/27/2|
|[coverage4_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_coverage4_20261016/iteration_001/result.json)|5242880|3785|565.212|36.635|0/7|0/29/3|1/30/1|3/28/1|4/26/2|2/28/2|1/29/2|3/28/1|
|[received100_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_received100_20261016/iteration_000/result.json)|5242880|3826|555.705|37.538|1/7|3/25/4|1/24/7|5/23/4|6/22/4|5/27/0 ✓|3/25/4|3/24/5|
|[received100_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_received100_20261016/iteration_001/result.json)|5242880|3782|561.254|37.616|0/7|3/24/5|2/26/4|3/27/2|3/27/2|4/26/2|4/23/5|1/30/1|
|[weight16_lowerlr_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_weight16_lowerlr_20261016/iteration_000/result.json)|5242880|3868|564.478|36.164|0/7|1/30/1|1/28/3|2/28/2|2/29/1|2/29/1|0/31/1|5/26/1|
|[weight16_lowerlr_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_weight16_lowerlr_20261016/iteration_001/result.json)|5242880|3829|567.359|36.535|0/7|2/28/2|0/30/2|3/27/2|3/28/1|1/28/3|1/29/2|1/30/1|
|[gae995_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_gae995_20261016/iteration_000/result.json)|5242880|3802|562.250|36.548|0/7|1/27/4|2/29/1|2/25/5|2/25/5|1/29/2|1/29/2|5/25/2|
|[gae995_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_gae995_20261016/iteration_001/result.json)|5242880|3861|567.850|36.686|0/7|3/27/2|0/30/2|4/26/2|4/26/2|1/30/1|1/28/3|0/30/2|
|[reference010_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_reference010_20261016/iteration_000/result.json)|5242880|3858|603.309|36.198|0/7|3/27/2|0/32/0|2/28/2|2/28/2|2/29/1|4/26/2|3/26/3|
|[reference010_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_reference010_20261016/iteration_001/result.json)|5242880|3853|606.806|36.601|0/7|3/23/6|1/29/2|2/27/3|2/27/3|2/27/3|3/27/2|4/27/1|
|[reference100_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_reference100_20261016/iteration_000/result.json)|5242880|3860|602.233|36.691|3/7|1/28/3|2/30/0 ✓|3/29/0 ✓|2/29/1|2/30/0 ✓|2/27/3|2/29/1|
|[reference100_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_reference100_20261016/iteration_001/result.json)|5242880|3861|608.496|36.091|0/7|1/28/3|1/29/2|1/29/2|1/29/2|2/29/1|3/23/6|1/29/2|
|[reference_continue_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_reference_continue_20261016/iteration_000/result.json)|5242880|3869|618.676|36.158|2/7|3/27/2|4/28/0 ✓|1/29/2|3/27/2|3/29/0 ✓|4/27/1|2/27/3|
|[reference_continue_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_reference_continue_20261016/iteration_001/result.json)|5242880|3830|596.038|36.919|0/7|1/27/4|1/30/1|1/29/2|1/29/2|1/30/1|1/26/5|0/30/2|
|[reference_refresh_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_reference_refresh_20261016/iteration_000/result.json)|5242880|3853|604.675|36.462|0/7|1/28/3|1/29/2|3/27/2|4/26/2|3/27/2|8/23/1|0/30/2|
|[reference_refresh_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_reference_refresh_20261016/iteration_001/result.json)|5242880|3857|592.460|36.762|1/7|2/27/3|1/31/0 ✓|1/29/2|1/28/3|0/31/1|3/28/1|0/30/2|
|[refresh_lowerlr_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_refresh_lowerlr_20261016/iteration_000/result.json)|5242880|3858|605.699|36.712|3/7|1/25/6|2/30/0 ✓|5/27/0 ✓|3/27/2|3/29/0 ✓|1/28/3|1/29/2|
|[refresh_lowerlr_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_refresh_lowerlr_20261016/iteration_001/result.json)|5242880|3854|600.978|36.643|0/7|1/27/4|3/28/1|2/28/2|2/28/2|3/26/3|2/28/2|2/27/3|
|[refresh_lowerlr2_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_refresh_lowerlr2_20261016/iteration_000/result.json)|5242880|3841|607.832|36.640|2/7|3/23/6|1/31/0 ✓|3/28/1|2/27/3|2/30/0 ✓|2/28/2|1/29/2|
|[refresh_lowerlr2_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_refresh_lowerlr2_20261016/iteration_001/result.json)|5242880|3855|602.556|37.090|2/7|1/25/6|1/31/0 ✓|4/27/1|5/25/2|2/30/0 ✓|2/26/4|1/28/3|
|[gae095_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_gae095_20261016/iteration_000/result.json)|5242880|3864|611.337|36.287|1/7|2/24/6|1/30/1|4/28/0 ✓|3/28/1|0/32/0|3/26/3|1/30/1|
|[gae095_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_gae095_20261016/iteration_001/result.json)|5242880|3869|614.826|36.246|2/7|2/26/4|1/30/1|3/29/0 ✓|4/27/1|1/30/1|4/28/0 ✓|1/28/3|
|[entropy001_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_entropy001_20261016/iteration_000/result.json)|5242880|3853|592.818|36.753|1/7|1/28/3|2/30/0 ✓|4/27/1|2/29/1|3/28/1|4/27/1|3/28/1|
|[entropy001_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_entropy001_20261016/iteration_001/result.json)|5242880|3873|609.325|37.079|1/7|3/26/3|2/28/2|2/28/2|1/29/2|3/29/0 ✓|3/27/2|3/28/1|
|[alt64001_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_alt64001_20261016/iteration_000/result.json)|5242880|3853|596.805|36.825|1/7|0/28/4|1/31/0 ✓|4/27/1|2/29/1|2/29/1|5/23/4|1/29/2|
|[alt64001_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_alt64001_20261016/iteration_001/result.json)|5242880|3870|596.873|36.940|2/7|0/26/6|2/30/0 ✓|4/28/0 ✓|2/29/1|2/28/2|3/25/4|0/29/3|
|[reference030_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_reference030_20261016/iteration_000/result.json)|5242880|3839|593.189|36.947|2/7|1/27/4|0/31/1|3/29/0 ✓|3/29/0 ✓|2/28/2|2/28/2|1/28/3|
|[reference030_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_reference030_20261016/iteration_001/result.json)|5242880|3847|593.002|37.199|0/7|2/26/4|3/28/1|2/29/1|2/28/2|3/28/1|5/26/1|2/29/1|
|[alt65500_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_alt65500_20261016/iteration_000/result.json)|5242880|3863|596.142|36.555|2/7|3/23/6|2/28/2|4/28/0 ✓|3/28/1|1/31/0 ✓|2/27/3|2/28/2|
|[alt65500_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_alt65500_20261016/iteration_001/result.json)|5242880|3862|591.285|36.988|0/7|0/27/5|1/28/3|3/28/1|2/28/2|4/25/3|3/27/2|3/28/1|
|[damage025_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_damage025_20261016/iteration_000/result.json)|5242880|3877|586.828|37.271|1/7|2/26/4|1/30/1|2/28/2|1/28/3|3/28/1|4/28/0 ✓|2/29/1|
|[damage025_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_damage025_20261016/iteration_001/result.json)|5242880|3861|605.221|37.274|0/7|2/26/4|1/30/1|2/28/2|2/27/3|4/27/1|2/26/4|1/29/2|
|[relative050_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_relative050_20261016/iteration_000/result.json)|5242880|3882|626.330|36.599|2/7|1/27/4|2/29/1|4/26/2|3/28/1|2/30/0 ✓|2/27/3|1/31/0 ✓|
|[relative050_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_relative050_20261016/iteration_001/result.json)|5242880|3846|626.526|36.843|0/7|0/29/3|1/30/1|4/27/1|2/29/1|3/28/1|3/27/2|2/28/2|
|[guidance000_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_guidance000_20261016/iteration_000/result.json)|5242880|3873|639.902|36.351|2/7|2/27/3|2/30/0 ✓|3/28/1|1/29/2|3/29/0 ✓|4/26/2|1/30/1|
|[guidance000_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_guidance000_20261016/iteration_001/result.json)|5242880|3863|620.484|36.708|2/7|1/27/4|2/29/1|4/27/1|2/29/1|3/29/0 ✓|5/26/1|1/31/0 ✓|
|[alt67501_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_alt67501_20261016/iteration_000/result.json)|5242880|3861|628.385|36.627|1/7|1/27/4|1/29/2|3/27/2|3/26/3|4/28/0 ✓|2/27/3|0/31/1|
|[alt67501_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_alt67501_20261016/iteration_001/result.json)|5242880|3839|605.463|36.946|1/7|1/27/4|1/30/1|2/29/1|2/29/1|4/28/0 ✓|3/27/2|1/28/3|
|[alt58001_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_alt58001_20261016/iteration_000/result.json)|5242880|3817|624.108|36.747|0/7|2/29/1|2/29/1|2/27/3|1/28/3|5/25/2|2/25/5|2/27/3|
|[alt58001_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_alt58001_20261016/iteration_001/result.json)|5242880|3814|627.038|36.128|1/7|3/29/0 ✓|2/29/1|1/28/3|0/30/2|4/27/1|2/28/2|3/27/2|
|[continue68501_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_continue68501_20261016/iteration_000/result.json)|5242880|3805|623.538|36.642|2/7|3/29/0 ✓|2/30/0 ✓|0/29/3|1/29/2|5/25/2|1/29/2|4/26/2|
|[continue68501_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_continue68501_20261016/iteration_001/result.json)|5242880|3824|625.939|36.109|1/7|2/30/0 ✓|3/28/1|2/27/3|2/27/3|2/29/1|2/26/4|4/25/3|
|[gamma9995_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_gamma9995_20261016/iteration_000/result.json)|5242880|3790|613.053|36.453|1/7|3/29/0 ✓|2/29/1|2/28/2|1/29/2|1/30/1|1/29/2|4/24/4|
|[gamma9995_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_gamma9995_20261016/iteration_001/result.json)|5242880|3799|600.168|36.515|2/7|4/28/0 ✓|2/30/0 ✓|0/29/3|2/27/3|4/27/1|0/28/4|3/27/2|
|[draw025_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_draw025_20261016/iteration_000/result.json)|5242880|3799|592.472|36.918|2/7|4/28/0 ✓|2/30/0 ✓|0/30/2|1/29/2|5/25/2|1/27/4|3/27/2|
|[draw025_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_draw025_20261016/iteration_001/result.json)|5242880|3809|578.845|36.190|2/7|1/31/0 ✓|0/31/1|2/26/4|1/28/3|2/30/0 ✓|0/30/2|3/24/5|
|[winbonus100_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_winbonus100_20261016/iteration_000/result.json)|5242880|3793|593.585|36.798|0/7|2/29/1|1/30/1|2/28/2|1/28/3|5/23/4|1/27/4|0/27/5|
|[winbonus100_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_winbonus100_20261016/iteration_001/result.json)|5242880|3820|587.787|36.681|2/7|4/28/0 ✓|2/30/0 ✓|1/29/2|1/28/3|4/25/3|0/29/3|2/27/3|
|[lambda999_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_lambda999_20261016/iteration_000/result.json)|5242880|3807|591.125|35.996|1/7|2/29/1|2/30/0 ✓|0/27/5|0/27/5|5/26/1|0/30/2|1/27/4|
|[lambda999_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_lambda999_20261016/iteration_001/result.json)|5242880|3805|592.772|36.741|1/7|3/28/1|2/30/0 ✓|1/28/3|1/27/4|5/25/2|1/27/4|1/29/2|
|[update375_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_update375_20261016/iteration_000/result.json)|5242880|3802|584.781|36.409|1/7|3/29/0 ✓|1/29/2|2/29/1|1/30/1|1/26/5|0/29/3|3/27/2|
|[update375_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_update375_20261016/iteration_001/result.json)|5242880|3804|597.364|36.278|0/7|1/28/3|1/30/1|2/29/1|1/30/1|5/22/5|0/29/3|1/28/3|
|[rollout2048_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_rollout2048_20261016/iteration_000/result.json)|5242880|3776|585.856|36.707|0/7|3/28/1|1/29/2|0/31/1|0/31/1|2/25/5|0/29/3|3/27/2|
|[rollout2048_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_rollout2048_20261016/iteration_001/result.json)|5242880|3814|596.107|36.502|2/7|3/29/0 ✓|1/31/0 ✓|0/30/2|0/30/2|4/27/1|0/29/3|2/27/3|
|[longref030_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_longref030_20261016/iteration_000/result.json)|5242880|3811|601.314|36.573|2/7|1/30/1|1/30/1|1/31/0 ✓|1/31/0 ✓|2/27/3|0/29/3|4/27/1|
|[longref030_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_longref030_20261016/iteration_001/result.json)|5242880|3782|583.555|36.459|2/7|1/30/1|1/31/0 ✓|3/26/3|1/28/3|3/29/0 ✓|1/27/4|4/26/2|
|[threat_prior_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_threat_prior_20261016/iteration_000/result.json)|5242880|3791|598.066|37.722|0/7|2/27/3|0/28/4|3/25/4|3/24/5|3/24/5|2/27/3|3/25/4|
|[threat_prior_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_threat_prior_20261016/iteration_001/result.json)|5242880|3774|598.120|37.631|1/7|0/30/2|1/31/0 ✓|2/28/2|3/27/2|1/27/4|1/28/3|2/28/2|
|[threat_refzero_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_threat_refzero_20261016/iteration_000/result.json)|5242880|3748|581.239|37.477|0/7|1/27/4|1/28/3|5/24/3|4/25/3|1/29/2|0/29/3|2/27/3|
|[threat_refzero_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_threat_refzero_20261016/iteration_001/result.json)|5242880|3799|563.893|37.567|0/7|1/28/3|2/29/1|1/29/2|1/29/2|1/29/2|2/29/1|3/24/5|
|[wez_or_damaged_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_wez_or_damaged_20261016/iteration_000/result.json)|5242880|3708|580.987|38.640|0/7|2/29/1|0/28/4|4/25/3|4/26/2|4/26/2|2/27/3|2/28/2|
|[wez_or_damaged_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_wez_or_damaged_20261016/iteration_001/result.json)|5242880|3660|577.589|38.563|0/7|2/27/3|3/27/2|3/27/2|4/26/2|4/26/2|3/26/3|5/22/5|
|[terminal_loss200_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_terminal_loss200_20261016/iteration_000/result.json)|5242880|3686|577.751|37.534|0/7|2/27/3|3/26/3|5/24/3|6/23/3|4/24/4|4/23/5|4/27/1|
|[terminal_loss200_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_terminal_loss200_20261016/iteration_001/result.json)|5242880|3703|578.979|38.486|0/7|1/29/2|2/25/5|3/26/3|3/26/3|5/24/3|4/23/5|3/28/1|
|[early_range4000_20261016 / 0](../artifacts/20261016/evidence/production/seven_league_early_range4000_20261016/iteration_000/result.json)|5242880|3683|595.981|42.658|0/7|1/29/2|2/23/7|4/23/5|4/24/4|4/23/5|2/27/3|6/25/1|
|[early_range4000_20261016 / 1](../artifacts/20261016/evidence/production/seven_league_early_range4000_20261016/iteration_001/result.json)|5242880|3702|595.442|42.012|0/7|3/27/2|2/27/3|4/25/3|5/24/3|2/24/6|8/23/1|5/26/1|

### 분기별 설정 기록

|실행|LR|gamma|GAE|Entropy|Received|Reference|Win bonus|Draw penalty|Rollout|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[main_20261015](../artifacts/20261016/evidence/production/seven_league_main_20261015/settings.json)|3e-05|0.999|0.99|0.003|0.5|source default; see settings|source default; see settings|source default; see settings|640|
|[lowerlr_20261015](../artifacts/20261016/evidence/production/seven_league_lowerlr_20261015/settings.json)|1.5e-05|0.999|0.99|0.003|0.5|source default; see settings|source default; see settings|source default; see settings|640|
|[longer_20261015](../artifacts/20261016/evidence/production/seven_league_longer_20261015/settings.json)|1.5e-05|0.999|0.99|0.003|0.5|source default; see settings|source default; see settings|source default; see settings|640|
|[lossweight_20261015](../artifacts/20261016/evidence/production/seven_league_lossweight_20261015/settings.json)|1.5e-05|0.999|0.99|0.003|0.5|source default; see settings|source default; see settings|source default; see settings|640|
|[received075_20261015](../artifacts/20261016/evidence/production/seven_league_received075_20261015/settings.json)|1.5e-05|0.999|0.99|0.003|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[entropy010_20261016](../artifacts/20261016/evidence/production/seven_league_entropy010_20261016/settings.json)|1.5e-05|0.999|0.99|0.01|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[entropy_continue_20261016](../artifacts/20261016/evidence/production/seven_league_entropy_continue_20261016/settings.json)|1.5e-05|0.999|0.99|0.01|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[threepass_continue_20261016](../artifacts/20261016/evidence/production/seven_league_threepass_continue_20261016/settings.json)|1.5e-05|0.999|0.99|0.01|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[threepass_lowerlr_20261016](../artifacts/20261016/evidence/production/seven_league_threepass_lowerlr_20261016/settings.json)|7.5e-06|0.999|0.99|0.01|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[entropy003_20261016](../artifacts/20261016/evidence/production/seven_league_entropy003_20261016/settings.json)|7.5e-06|0.999|0.99|0.003|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[entropy000_20261016](../artifacts/20261016/evidence/production/seven_league_entropy000_20261016/settings.json)|7.5e-06|0.999|0.99|0.0|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[lossweight16_20261016](../artifacts/20261016/evidence/production/seven_league_lossweight16_20261016/settings.json)|7.5e-06|0.999|0.99|0.0|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[coverage4_20261016](../artifacts/20261016/evidence/production/seven_league_coverage4_20261016/settings.json)|7.5e-06|0.999|0.99|0.0|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[received100_20261016](../artifacts/20261016/evidence/production/seven_league_received100_20261016/settings.json)|7.5e-06|0.999|0.99|0.0|1.0|source default; see settings|source default; see settings|source default; see settings|640|
|[weight16_lowerlr_20261016](../artifacts/20261016/evidence/production/seven_league_weight16_lowerlr_20261016/settings.json)|3.75e-06|0.999|0.99|0.0|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[gae995_20261016](../artifacts/20261016/evidence/production/seven_league_gae995_20261016/settings.json)|3.75e-06|0.999|0.995|0.0|0.75|source default; see settings|source default; see settings|source default; see settings|640|
|[reference010_20261016](../artifacts/20261016/evidence/production/seven_league_reference010_20261016/settings.json)|3.75e-06|0.999|0.99|0.0|0.75|0.1|source default; see settings|source default; see settings|640|
|[reference100_20261016](../artifacts/20261016/evidence/production/seven_league_reference100_20261016/settings.json)|3.75e-06|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[reference_continue_20261016](../artifacts/20261016/evidence/production/seven_league_reference_continue_20261016/settings.json)|3.75e-06|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[reference_refresh_20261016](../artifacts/20261016/evidence/production/seven_league_reference_refresh_20261016/settings.json)|3.75e-06|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[refresh_lowerlr_20261016](../artifacts/20261016/evidence/production/seven_league_refresh_lowerlr_20261016/settings.json)|1.875e-06|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[refresh_lowerlr2_20261016](../artifacts/20261016/evidence/production/seven_league_refresh_lowerlr2_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[gae095_20261016](../artifacts/20261016/evidence/production/seven_league_gae095_20261016/settings.json)|9.375e-07|0.999|0.95|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[entropy001_20261016](../artifacts/20261016/evidence/production/seven_league_entropy001_20261016/settings.json)|9.375e-07|0.999|0.99|0.001|0.75|1.0|source default; see settings|source default; see settings|640|
|[alt64001_20261016](../artifacts/20261016/evidence/production/seven_league_alt64001_20261016/settings.json)|9.375e-07|0.999|0.95|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[reference030_20261016](../artifacts/20261016/evidence/production/seven_league_reference030_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|0.3|source default; see settings|source default; see settings|640|
|[alt65500_20261016](../artifacts/20261016/evidence/production/seven_league_alt65500_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|0.3|source default; see settings|source default; see settings|640|
|[damage025_20261016](../artifacts/20261016/evidence/production/seven_league_damage025_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[relative050_20261016](../artifacts/20261016/evidence/production/seven_league_relative050_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[guidance000_20261016](../artifacts/20261016/evidence/production/seven_league_guidance000_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[alt67501_20261016](../artifacts/20261016/evidence/production/seven_league_alt67501_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[alt58001_20261016](../artifacts/20261016/evidence/production/seven_league_alt58001_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[continue68501_20261016](../artifacts/20261016/evidence/production/seven_league_continue68501_20261016/settings.json)|9.375e-07|0.999|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[gamma9995_20261016](../artifacts/20261016/evidence/production/seven_league_gamma9995_20261016/settings.json)|9.375e-07|0.9995|0.99|0.0|0.75|1.0|source default; see settings|source default; see settings|640|
|[draw025_20261016](../artifacts/20261016/evidence/production/seven_league_draw025_20261016/settings.json)|9.375e-07|0.9995|0.99|0.0|0.75|1.0|0.0|0.25|640|
|[winbonus100_20261016](../artifacts/20261016/evidence/production/seven_league_winbonus100_20261016/settings.json)|9.375e-07|0.9995|0.99|0.0|0.75|1.0|1.0|0.25|640|
|[lambda999_20261016](../artifacts/20261016/evidence/production/seven_league_lambda999_20261016/settings.json)|9.375e-07|0.9995|0.999|0.0|0.75|1.0|1.0|0.25|640|
|[update375_20261016](../artifacts/20261016/evidence/production/seven_league_update375_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|1.0|1.0|0.25|640|
|[rollout2048_20261016](../artifacts/20261016/evidence/production/seven_league_rollout2048_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|1.0|1.0|0.25|2048|
|[longref030_20261016](../artifacts/20261016/evidence/production/seven_league_longref030_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|0.3|1.0|0.25|2048|
|[threat_prior_20261016](../artifacts/20261016/evidence/production/seven_league_threat_prior_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|0.3|1.0|0.25|2048|
|[threat_refzero_20261016](../artifacts/20261016/evidence/production/seven_league_threat_refzero_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|0.0|1.0|0.25|2048|
|[wez_or_damaged_20261016](../artifacts/20261016/evidence/production/seven_league_wez_or_damaged_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|0.0|1.0|0.25|2048|
|[terminal_loss200_20261016](../artifacts/20261016/evidence/production/seven_league_terminal_loss200_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|0.0|1.0|0.25|2048|
|[early_range4000_20261016](../artifacts/20261016/evidence/production/seven_league_early_range4000_20261016/settings.json)|3.75e-06|0.9995|0.999|0.0|0.75|0.0|1.0|0.25|2048|

## 무승부 원인과 원본 7개 리그전

초기 baseline은 33승170무21패이며, 170무 모두 두 체력이 반올림 후 0이었다. 따라서 당시 다수 무승부가 동시 소모였다는 관측은 맞다. 이를 모든 이후 경기의 원인이라고 일반화하지 않는다. [진단 보고서](../artifacts/20261016/reports/seven_mutual_zero_diagnosis_20261016/SUMMARY.md), [논문 조사](LITERATURE_2D_RL_MUTUAL_ZERO_20261016.md).

원본 7개끼리 기존 동일 조건 결과를 재집계했다: 21쌍×32=672개 고유 경기, 모델당 192경기/6상대. 44701은 46201에, 46500은 45800에 strict 승리, 나머지 matchup은 양측 모두 실패했다. 누구도 6개 전부 통과하지 않았다. [전체 상대표](../artifacts/20261016/reports/seven_original_round_robin_20261016/SUMMARY.md)와 [672경기 기록](../artifacts/20261016/evidence/seven_original_round_robin_20261016/selected_games.json).

## 직접 조종에서 관측 기반 제어 후보로

직접 조종은 실제 시뮬레이터 상태를 보고 에이전트가 명령과 유지 구간을 고른 개별 경기다. 그 행동을 일반화한 controller의 224경기는 코드가 매 관측에서 행동을 산출한다. 이는 에이전트가 224경기를 동시에 손으로 조종한 것이 아니며, **새 신경망 학습 결정은 0**이다. 수동 에피소드의 성공과 전체 screen 통과는 별개다. 후방뿐 아니라 측면을 포함하여 “나는 때리고 상대는 못 때리는” 구간을 평가한다.

구형 시간표 replay 후보 manual001은 7승30무187패/0상대 통과로 실패했다. 이후 공개 관측 39개/행동 9개의 상태기계로 바꿨다. wide/approach→초기 merge→우선회·속도 유지→signed gun lead 추적→일방 공격→실제 체력 우위와 안전한 이탈 기하가 있을 때 heading hold. 상대 ID·seed·privileged state로 행동을 정하지 않는다. 시간은 reset 감지용이며 시간표 기동이 아니다.

|후보|변경|평가 경기|승/무/패|통과|평가 초|판단|
|---|---|---:|---|---:|---:|---|
|[feedback_000](../artifacts/20261016/direct/feedback_000/REPORT.md)|시연에서 관측 기반 상태기계 작성|224|160/6/58|0/7|47.991280|후속 변경 전 원본 보존|
|[feedback_001](../artifacts/20261016/direct/feedback_001/REPORT.md)|signed gun-lead gate 교정|224|189/4/31|0/7|50.121004|후속 변경 전 원본 보존|
|[feedback_002](../artifacts/20261016/direct/feedback_002/REPORT.md)|이탈 gate를 실제 상대 기하로 변경|224|203/4/17|1/7|52.165296|현재 보존|
|[feedback_003](../artifacts/20261016/direct/feedback_003/REPORT.md)|brake speed 482→520만 변경|224|202/1/21|0/7|51.666549|회귀로 기각|

**현재 작업 기준은 feedback_002**다. feedback_003은 총승수가 1 줄고 strict 통과가 0으로 떨어져 기각했다. 두 후보의 policy.py 바이트는 같고 파라미터만 달랐다. 002의 개별 상대 결과:

|상대|승|무|패|strict|
|---|---:|---:|---:|---|
|s44701_step_002621440|30|0|2|loss|
|s45800_step_002621440|28|0|4|loss|
|s46201_step_005242880|26|1|5|loss|
|s46401_step_005242880|28|1|3|loss|
|s46500_step_005242880|29|1|2|loss|
|s47236_step_002621440|31|0|1|loss|
|s47238_step_010485760|31|1|0|win|

수동 manual003은 47236/red에서 HP .505/0 승리, 약 2.5초 일방 공격(.505 피해/0 피격)을 관측했다. manual004는 47236/blue에서 .505/0 승리, 약 2.6초 일방 공격이었다. manual005는 45800/red에서 .604/.043 종료 체력 승리, 약 3.05초 일방 공격 후 이탈했다. [실제 manual003 경기 영상](../artifacts/20261016/video/manual003_full_match.mp4), [측면 공격 느린 영상](../artifacts/20261016/video/manual003_side_attack_slow.mp4).

마지막 manual006은 46201/red/seed3401500000/action4900에서 41.4초/828tick 동일 prefix 뒤 추가 감속·좌반전을 시험했지만 74.3초에 0/.472로 패배했다. 총 1,486tick/13명령, 상호작용·추론 포함 221.224634초였다. 단순 감속 또는 brake520은 해법으로 채택하지 않았다.

직접 제어 진단 누계 **4,021 native 경기 실행**, 수동 에피소드 7개. 평가·진단·재생을 포함한 실행 수이며 독립 unseen 경기 수가 아니다. 초기 넓은 grid는 과거 기록으로만 남긴다. 사용자의 수정 이후 평가 크기를 계속 늘리는 탐색은 하지 않으며 새 후보도 동일 224 screen을 따른다. 초기 grid·회피·setpoint·opening 실험 경과도 [전체 보고서 폴더](../artifacts/20261016/reports/direct_pilot_20261016/)에 보존했다. 이번 게시용 검증은 이 역사적 누계에 합치지 않는다.

## 다음 컴퓨터에서 이어갈 정확한 출발점

feedback002의 brake_speed=482, merge_range=240을 유지한 원본에서 출발한다. 다음 **미시험 가설**은 초기 merge 우선회를 더 일찍 시작하도록 merge_range만 240→600으로 바꾸는 것이다. 기각된 003의 brake520을 상속하지 않는다. [검토용 다음 작업 설명](../artifacts/20261016/reports/direct_pilot_20261016/feedback_003/NEXT_MERGE_BRIEF.md). 새 폴더/출력·부모 identity·정확한 한 변수 diff·패키지 loader parity를 먼저 검증하고, 승인된 재개 뒤에만 224 screen을 실행한다. 가설이 성공했다는 뜻은 아니다.

v1은 screen 7/7와 별도 미사용 confirmation 전부를 통과한 최초 고정 모델에 부여한다. 이후 실제 개선된 모델에 v2,v3…를 부여하고 부모·정책·가중치·설정·해시·상대별 결과·시간을 기록한다. 지금 imitation/curriculum 학습을 시작하지 않는다. 통과 후 초반 피격 감소를 시도해도 strict 승리를 유지해야 하며 도망만 가서 무승부인 후보는 실패다.

## 원본 문서와 증거

- [연속 작업 일지](SEVEN_MODEL_LOCAL_LEAGUE_20261015.md): 이전 진행 중 표기는 각 당시 상태다. 현재 상태는 이 인계 문서와 LEAGUE_STATE의 publication_handoff를 따른다.
- [122개 기존 가중치·설정 변경 목록](HISTORICAL_WEIGHT_AND_SETTING_CHANGES_20261015.md): 이전 게시 원본 보존.
- [Production ledger 원본](../artifacts/20261016/evidence/campaign.json), [92회 result/settings](../artifacts/20261016/evidence/production/).
- [직접 제어 원본 보고서](../artifacts/20261016/reports/direct_pilot_20261016/), [고정 후보·224 raw cells·audit](../artifacts/20261016/direct/).
- [이식/실행/재개 안내](CPU_CONTINUATION_HANDOFF_20261016.md).

옛 문서/JSON의 절대 경로·PID·실행 중 표기는 provenance다. 전체 중간 학습 아카이브를 올렸다는 뜻이 아니며, 각 결과/설정과 진행 내용은 보존하되 선택하지 않은 중간 가중치·대량 raw grid는 기존 컴퓨터 로컬에 남긴다.
