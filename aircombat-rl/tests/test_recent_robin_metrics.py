from tools.league_recent_round_robin import analyze


def test_reciprocal_counts_and_equal_role_weighting():
    roles={'a':[{'id':'a0'},{'id':'a1'}],'b':[{'id':'b0'}],'c':[{'id':'c0'},{'id':'c1'}]}
    rows=[]
    for a,b,w,d,l in [('a','b',2,0,0),('a','c',0,2,0),('b','c',0,0,2)]:
        for own in roles[a]:
            for foe in roles[b]:rows.append(dict(own=own['id'],foe=foe['id'],episodes=[{},{}],summary=dict(wins=w,draws=d,losses=l)))
    result=analyze(dict(roles=roles,n=2,scope='test'),rows)
    assert result['matrix']['a']['b']['wins']==result['matrix']['b']['a']['losses']==4
    assert result['matrix']['a']['c']['draws']==result['matrix']['c']['a']['draws']==8
    assert result['aggregates']['a']['equal_role_score']==.75
    assert result['aggregates']['c']['equal_role_score']==.75
    assert result['aggregates']['b']['equal_role_score']==0
