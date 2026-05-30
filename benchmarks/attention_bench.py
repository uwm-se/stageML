

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from stageml.runtime    import compile_model
from stageml.tracer     import trace_and_annotate
from stageml.evaluator  import specialize


class SingleHeadAttention(nn.Module):

    def __init__(self, embed_dim: int = 64, seq_len: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len   = seq_len
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)
        return self.W_o(attn_out)


def benchmark_latency(
    fn,
    x: torch.Tensor,
    warmup: int = 100,
    iterations: int = 1000,
) -> float:
    for _ in range(warmup):
        fn(x)
    start = time.perf_counter()
    for _ in range(iterations):
        fn(x)
    return (time.perf_counter() - start) / iterations * 1000


def main():
    print("=" * 60)
    print("StageML Benchmark 2: Single-Head Attention")
    print("=" * 60)

    model = SingleHeadAttention(embed_dim=64, seq_len=32)
    model.eval()
    x = torch.randn(1, 32, 64)

    stage_env = {"x": "stage1"}

    print("\n[1] Running StageML compiler pipeline...")
    compile_model(model, example_input=x, stage_env=stage_env, verbose=True)

    print("\n[2] Latency benchmark (1000 iterations)...")
    with torch.no_grad():
        eager_ms = benchmark_latency(model, x)
        print(f"  PyTorch eager   : {eager_ms:.4f} ms/call")

        compiled_model = torch.compile(model)
        compiled_model(x)  # warmup compile
        compile_ms = benchmark_latency(compiled_model, x)
        print(f"  torch.compile   : {compile_ms:.4f} ms/call")

    print("\n[3] Numerical correctness...")
    gm, gamma = trace_and_annotate(model, stage_env)
    gm_residual = specialize(gm, gamma)
    with torch.no_grad():
        original_out = model(x)
        residual_out = gm_residual(x)
        max_diff = (original_out - residual_out).abs().max().item()
        print(f"  Max abs difference: {max_diff:.2e}")
        print(f"  Correctness: {'PASS' if max_diff < 1e-5 else 'FAIL'}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
