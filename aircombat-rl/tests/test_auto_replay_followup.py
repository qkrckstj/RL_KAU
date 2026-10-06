from tools.auto_replay_followup import retention


def test_retention_branch_is_based_on_validation_only():
    def h(wins):
        return [dict(summary=dict(kills=w)) for w in wins]
    assert retention(h([9,9,9,9,9,9]))["retained"]
    assert not retention(h([9,0,0,0,0,0]))["retained"]
    assert not retention(h([9,0,0,0,0,9]))["retained"]
    assert not retention(h([0,0,0,0,0,0]))["retained"]
