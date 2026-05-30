"""
StageML tested on Microsoft's official LoRA implementation.
Source: https://github.com/microsoft/LoRA
 
Microsoft's loralib.Linear computes:
  result = F.linear(x, W, bias) + (dropout(x) @ A.T @ B.T) * scaling
 
The x @ A.T part mixes dynamic input with static weight, so that chain
is stage-1. But we can also test the Conv2d variant which computes:
  weight + (lora_B @ lora_A).view(weight.shape) * scaling
where lora_B @ lora_A is entirely stage-0.
 
We also build a model that USES loralib layers, which is the real
deployment scenario.
"""
 
import torch
import torch.nn as nn
import time
 
try:
    import loralib as lora
except ImportError:
    raise ImportError("Install loralib: pip install loralib")
 
from stageml.tracer import trace_and_annotate
from stageml.evaluator import specialize
from stageml.annotations import stage0, stage1
 
 
def benchmark_latency(fn, x, warmup=200, iterations=2000):
    for _ in range(warmup):
        with torch.no_grad():
            fn(x)
    start = time.perf_counter()
    for _ in range(iterations):
        with torch.no_grad():
            fn(x)
    elapsed = (time.perf_counter() - start) / iterations * 1000
    return elapsed
 
 
class LoRAModel(nn.Module):
    """
    A simple model using Microsoft's loralib.Linear layers.
    This is how LoRA is actually used in practice: you replace
    nn.Linear layers with lora.Linear layers in an existing model.
    """
    def __init__(self, dim=256, r=16):
        super().__init__()
        self.layer1 = lora.Linear(dim, dim, r=r)
        self.layer2 = lora.Linear(dim, dim, r=r)
        self.relu = nn.ReLU()
 
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x
 
 
class ManualLoRAMerge(nn.Module):
    """
    Same model but with manual weight merging in forward().
    This is the pattern where StageML shines: the merged weight
    W + (B @ A) * scaling is computed every forward call but is
    entirely static.
    """
    def __init__(self, dim=256, r=16):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim, dim))
        self.A1 = nn.Parameter(torch.randn(r, dim) * 0.01)
        self.B1 = nn.Parameter(torch.randn(dim, r) * 0.01)
        self.W2 = nn.Parameter(torch.randn(dim, dim))
        self.A2 = nn.Parameter(torch.randn(r, dim) * 0.01)
        self.B2 = nn.Parameter(torch.randn(dim, r) * 0.01)
        self.scaling = 1.0 / r
 
    def forward(self, x):
        # Both merged weights are stage-0 computation chains
        merged1 = self.W1 + (self.B1 @ self.A1) * self.scaling
        merged2 = self.W2 + (self.B2 @ self.A2) * self.scaling
        x = torch.relu(x @ merged1.t())
        x = x @ merged2.t()
        return x
 
 
def run_test(name, model, input_shape, description=""):
    print("=" * 70)
    print(f"  {name}")
    if description:
        print(f"  {description}")
    print(f"  Input: {input_shape}")
    print("=" * 70)
 
    model.eval()
    x = torch.randn(*input_shape)
    stage_env = {'x': 'stage1'}
 
    try:
        gm, gamma = trace_and_annotate(model, stage_env)
    except Exception as e:
        print(f"  TRACING FAILED: {e}")
        import traceback
        traceback.print_exc()
        print()
        return
 
    total = len(list(gm.graph.nodes))
    static_ops = sum(1 for n in gm.graph.nodes if gamma.get(n) == stage0)
    compute_before = sum(1 for n in gm.graph.nodes if n.op == 'call_function')
    static_compute = [n for n in gm.graph.nodes
                      if n.op == 'call_function' and gamma.get(n) == stage0]
 
    print(f"  Total ops        : {total}")
    print(f"  Stage-0 (static) : {static_ops} ({static_ops/total*100:.1f}%)")
    print(f"  Compute ops      : {compute_before}")
 
    if static_compute:
        print(f"  Static compute ops (FOLDABLE):")
        for n in static_compute:
            target_name = n.target.__name__ if hasattr(n.target, '__name__') else str(n.target)
            print(f"    * {n.name} ({target_name})")
 
    try:
        gm_residual = specialize(gm, gamma)
        compute_after = sum(1 for n in gm_residual.graph.nodes if n.op == 'call_function')
 
        with torch.no_grad():
            original_out = model(x)
            residual_out = gm_residual(x)
 
        max_diff = (original_out - residual_out).abs().max().item()
 
        print(f"\n  Compute ops eliminated: {compute_before - compute_after}")
        print(f"  Correctness: {'PASS' if max_diff < 1e-4 else 'FAIL'} (diff: {max_diff:.2e})")
 
        with torch.no_grad():
            eager_ms = benchmark_latency(model, x)
            residual_ms = benchmark_latency(gm_residual, x)
        speedup = eager_ms / residual_ms if residual_ms > 0 else 0
 
        print(f"  Eager    : {eager_ms:.4f} ms/call")
        print(f"  Residual : {residual_ms:.4f} ms/call")
        print(f"  Speedup  : {speedup:.2f}x")
 
    except Exception as e:
        print(f"  SPECIALIZATION FAILED: {e}")
        import traceback
        traceback.print_exc()
 
    print()
 
 
def main():
    print()
    print("=" * 70)
    print("  StageML on Microsoft's Official LoRA (loralib)")
    print("  Source: https://github.com/microsoft/LoRA")
    print("=" * 70)
    print()
 
    # Test 1: Microsoft's loralib.Linear directly
    run_test(
        "Microsoft loralib.Linear (2-layer model)",
        LoRAModel(dim=256, r=16),
        (1, 256),
        "Uses loralib.Linear — the standard LoRA deployment pattern."
    )
 
    # Test 2: Manual merge pattern (the pattern StageML is best at)
    run_test(
        "Manual LoRA Merge (W + B@A pattern)",
        ManualLoRAMerge(dim=256, r=16),
        (1, 256),
        "Explicit W + (B @ A) * scaling merge in forward(). This is what"
        " StageML folds at compile time."
    )
 
    print("=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print("  Microsoft's loralib computes x @ A first (dynamic @ static),")
    print("  so the chain is stage-1 from the start.")
    print()
    print("  The manual merge pattern computes B @ A first (static @ static),")
    print("  creating a stage-0 chain that StageML can fold.")
    print()
    print("  This shows that code structure affects what StageML can optimise.")
    print("  The semantically identical computation has different staging")
    print("  properties depending on evaluation order.")
    print()
 
 
if __name__ == "__main__":
    main()