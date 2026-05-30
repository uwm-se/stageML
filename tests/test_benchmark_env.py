from stageml.benchmark_env import assert_same_environment, attach_environment, capture_environment


def test_same_environment_assertion_accepts_rows_from_one_process():
    env = capture_environment("cpu")
    result = {
        "a": attach_environment({"status": "ok", "p50_ms": 1.0}, env),
        "b": attach_environment({"status": "ok", "p50_ms": 2.0}, env),
    }
    assert_same_environment(result, ["a", "b"])


def test_same_environment_assertion_rejects_confounded_rows():
    env1 = capture_environment("cpu")
    env2 = dict(env1)
    env2["torch_version"] = "different"
    result = {
        "a": attach_environment({"status": "ok"}, env1),
        "b": attach_environment({"status": "ok"}, env2),
    }
    try:
        assert_same_environment(result, ["a", "b"])
    except RuntimeError as exc:
        assert "confounded benchmark" in str(exc)
    else:
        raise AssertionError("expected environment mismatch")
