from stageml.benchmark_stats import summarize_ms, speedup


def test_summarize_ms_reports_runs_and_std():
    stats = summarize_ms([1.0, 2.0, 3.0], warmups=2)
    assert stats["runs"] == 3
    assert stats["warmups"] == 2
    assert abs(stats["mean_ms"] - 2.0) < 1e-9
    assert abs(stats["std_ms"] - 1.0) < 1e-9
    assert stats["p50_ms"] == 2.0


def test_speedup_uses_selected_metric():
    base = {"p50_ms": 10.0}
    opt = {"p50_ms": 2.0}
    assert speedup(base, opt) == 5.0
