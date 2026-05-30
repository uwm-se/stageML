
import torch
import torch.nn.functional as F
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stageml.annotations import stage0, stage1, compile_staged, BindingTime
from stageml.tracer       import propagate_stages, staging_summary
import torch.fx as fx



def test_stage_objects_exist():
    assert stage0.level == 0
    assert stage1.level == 1

def test_lattice_join():
    """S ⊔ S = S,  S ⊔ D = D,  D ⊔ D = D"""
    assert stage0.join(stage0) == stage0
    assert stage0.join(stage1) == stage1
    assert stage1.join(stage0) == stage1
    assert stage1.join(stage1) == stage1

def test_staging_env_built_correctly():
    @compile_staged
    def f(x: stage1, W: stage0):
        return W @ x

    assert f._gamma["x"] == stage1
    assert f._gamma["W"] == stage0

def test_unannotated_defaults_to_dynamic():
    @compile_staged
    def f(x, W: stage0):
        return W @ x

    # x has no annotation → defaults to stage1
    assert f._gamma["x"] == stage1

def test_analyze_runs():
    @compile_staged
    def f(x: stage1, W: stage0):
        return W @ x
    f.analyze()   # should not raise



def _make_simple_graph():
    @compile_staged
    def f(x: stage1, W: stage0):
        return W @ x

    gm = fx.symbolic_trace(f)
    return gm, f._gamma

def test_placeholder_annotation():
    gm, gamma = _make_simple_graph()
    annotations = propagate_stages(gm.graph, gamma)
    nodes = {n.name: n for n in gm.graph.nodes}
    assert annotations[nodes["x"]] == stage1
    assert annotations[nodes["w"]] == stage0

def test_all_static_inputs_give_static_output():
    """If all inputs are stage0, the output should be stage0."""
    @compile_staged
    def f(W1: stage0, W2: stage0):
        return W1 @ W2

    gm = fx.symbolic_trace(f)
    annotations = propagate_stages(gm.graph, f._gamma)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    assert annotations[output_node] == stage0

def test_any_dynamic_input_gives_dynamic_output():
    """If any input is stage1, the output should be stage1."""
    @compile_staged
    def f(x: stage1, W: stage0):
        return W @ x

    gm = fx.symbolic_trace(f)
    annotations = propagate_stages(gm.graph, f._gamma)
    output_node = [n for n in gm.graph.nodes if n.op == "output"][0]
    assert annotations[output_node] == stage1

def test_staging_summary_counts():
    gm, gamma = _make_simple_graph()
    annotations = propagate_stages(gm.graph, gamma)
    summary = staging_summary(annotations)
    assert summary["total_ops"] == len(list(gm.graph.nodes))
    assert summary["static_ops"] + summary["dynamic_ops"] == summary["total_ops"]
    assert 0 <= summary["static_pct"] <= 100

def test_soundness_invariant():
 
    @compile_staged
    def mlp(x: stage1, W1: stage0, W2: stage0):
        h = torch.relu(W1 @ x)
        return W2 @ h

    gm = fx.symbolic_trace(mlp)
    annotations = propagate_stages(gm.graph, mlp._gamma)

    for node in gm.graph.nodes:
        if annotations.get(node) == stage0:
            for arg in node.args:
                if isinstance(arg, fx.Node):
                    assert annotations.get(arg) == stage0, \
                        f"SOUNDNESS VIOLATED: stage-0 node '{node.name}' " \
                        f"has stage-1 operand '{arg.name}'"

    print("\n  Soundness invariant: PASS — no stage-0 node has a stage-1 operand")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
