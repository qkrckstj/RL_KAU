from tools import league_extend_teacher_train as training


def test_extension_must_beat_teacher_and_initial_without_tail_regression(monkeypatch):
    monkeypatch.setattr(training,'reference_rank',[.85,.6,.25,-1])
    initial=[.8,.5,.25,-2]
    assert not training.extend([.82,.6,.25,-1],[.84,.6,.25,-1],initial)
    assert not training.extend([.82,.6,.25,-1],[.87,.59,.25,-1],initial)
    assert not training.extend([.82,.6,.25,-1],[.87,.6,0.,-1],initial)
    assert training.extend([.86,.6,.25,-1],[.86,.6,.25,-1],initial)


def test_initial_execution_itself_cannot_trigger_extension(monkeypatch):
    monkeypatch.setattr(training,'reference_rank',[.85,.6,.25,-1])
    initial=[.9,.7,.5,0]
    assert not training.extend(initial,initial,initial)
