
import torch
import torch.nn.functional as F
import time
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stageml import stage0, stage1, compile_staged, compile_model



@compile_staged
def tiny_mlp(
    x:  stage1,   # input — changes every call
    W1: stage0,   # weight layer 1 — fixed after training
    b1: stage0,   # bias layer 1   — fixed after training
    W2: stage0,   # weight layer 2 — fixed after training
    b2: stage0,   # bias layer 2   — fixed after training
) -> torch.Tensor:
    h = F.relu(F.linear(x, W1, b1))
    return F.linear(h, W2, b2)


def run_benchmark():
    print("\n" + "="*60)
    print("StageML Benchmark 1: TinyMLP")
    print("="*60)

    torch.manual_seed(42)
    W1 = torch.randn(128, 64)
    b1 = torch.randn(128)
    W2 = torch.randn(10, 128)
    b2 = torch.randn(10)

    x  = torch.randn(1, 64)

    static_vals = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    print("\n[1] Running StageML compiler pipeline...")
    residual_fn, report = compile_model(
        tiny_mlp,
        example_input=x,
        static_vals=static_vals,
        verbose=True,
    )

    N = 1000
    print(f"[2] Latency benchmark ({N} iterations)...")

    # Original
    t0 = time.perf_counter()
    for _ in range(N):
        _ = tiny_mlp(x, W1, b1, W2, b2)
    t_original = (time.perf_counter() - t0) / N * 1000

    # torch.compile
    compiled = torch.compile(lambda x: tiny_mlp(x, W1, b1, W2, b2))
    _ = compiled(x)  # warmup
    t0 = time.perf_counter()
    for _ in range(N):
        _ = compiled(x)
    t_compiled = (time.perf_counter() - t0) / N * 1000

    print(f"\n  PyTorch eager   : {t_original:.4f} ms/call")
    print(f"  torch.compile   : {t_compiled:.4f} ms/call")
    print(f"\n  Op count reduction: {report.static_pct:.1f}% of ops eliminated")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_benchmark()
