from __future__ import annotations

"""OpenAI-compatible vLLM HTTP benchmark for multi tenant LoRA traffic.

This benchmark measures the live-server metrics systems reviewers expect:
TTFT, TPOT, total latency, run count, and standard deviation. It is intentionally
separate from the hidden-state benchmark because it crosses the HTTP boundary.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any

from stageml.benchmark_stats import summarize_ms, write_json_csv


def zipf_probs(n: int, alpha: float) -> list[float]:
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n)]
    total = sum(weights)
    return [x / total for x in weights]


def sample_tenants(n: int, tenants: int, distribution: str, alpha: float, seed: int) -> list[int]:
    rng = random.Random(seed)
    if distribution == "uniform":
        return [rng.randrange(tenants) for _ in range(n)]
    if distribution == "zipf":
        return rng.choices(list(range(tenants)), weights=zipf_probs(tenants, alpha), k=n)
    if distribution == "bursty":
        out: list[int] = []
        while len(out) < n:
            t = rng.choices(list(range(tenants)), weights=zipf_probs(tenants, alpha), k=1)[0]
            out.extend([t] * 8)
        return out[:n]
    raise ValueError(f"unknown distribution {distribution}")


def parse_models(value: str, tenants: int) -> list[str]:
    models = [x.strip() for x in value.split(",") if x.strip()]
    if not models:
        raise ValueError("at least one model or adapter name is required")
    return [models[i % len(models)] for i in range(tenants)]


async def one_request(session: Any, *, url: str, model: str, prompt: str, max_tokens: int, tenant_id: int, timeout: int) -> dict[str, float | int | str]:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "extra_body": {"tenant_id": tenant_id},
    }
    t0 = time.perf_counter()
    first = None
    chunks = 0
    async with session.post(url, json=payload, timeout=timeout) as resp:
        if resp.status >= 400:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:500]}")
        async for raw in resp.content:
            now = time.perf_counter()
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line or line == "data: [DONE]":
                continue
            if first is None:
                first = now
            chunks += 1
    end = time.perf_counter()
    ttft_ms = ((first or end) - t0) * 1000.0
    total_ms = (end - t0) * 1000.0
    tpot_ms = max(0.0, (total_ms - ttft_ms) / max(chunks - 1, 1))
    return {"tenant_id": tenant_id, "ttft_ms": ttft_ms, "total_ms": total_ms, "tpot_ms": tpot_ms, "chunks": chunks, "model": model}


async def run_async(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import aiohttp
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("aiohttp is required. Install with pip install aiohttp") from exc

    tenants = sample_tenants(args.requests, args.tenants, args.tenant_distribution, args.zipf_alpha, args.seed)
    tenant_models = parse_models(args.models, args.tenants)
    prompts = [line.strip() for line in Path(args.prompts).read_text().splitlines() if line.strip()] if args.prompts else []
    if not prompts:
        prompts = ["Explain why compiler specialization can improve repeated inference workloads."]
    sem = asyncio.Semaphore(args.concurrency)
    rows: list[dict[str, Any]] = []

    async with aiohttp.ClientSession() as session:
        async def worker(i: int, tenant: int) -> None:
            async with sem:
                prompt = prompts[i % len(prompts)]
                model = tenant_models[tenant]
                rows.append(await one_request(session, url=args.url, model=model, prompt=prompt, max_tokens=args.max_tokens, tenant_id=tenant, timeout=args.timeout))

        await asyncio.gather(*(worker(i, t) for i, t in enumerate(tenants)))

    ttft = summarize_ms([float(x["ttft_ms"]) for x in rows], warmups=0)
    tpot = summarize_ms([float(x["tpot_ms"]) for x in rows], warmups=0)
    total = summarize_ms([float(x["total_ms"]) for x in rows], warmups=0)
    return {
        "benchmark": "vllm_http_multitenant_bench",
        "claim_boundary": "Live OpenAI-compatible HTTP benchmark. Use only when StageML router is integrated into the vLLM server under test.",
        "url": args.url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "tenants": args.tenants,
        "tenant_distribution": args.tenant_distribution,
        "zipf_alpha": args.zipf_alpha,
        "max_tokens": args.max_tokens,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "total_latency_ms": total,
        "raw_requests": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    p.add_argument("--models", required=True, help="comma-separated base or LoRA model names accepted by the vLLM server")
    p.add_argument("--prompts", default="")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=100)
    p.add_argument("--tenants", type=int, default=10)
    p.add_argument("--tenant-distribution", choices=["uniform", "zipf", "bursty"], default="zipf")
    p.add_argument("--zipf-alpha", type=float, default=1.2)
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="paper_outputs/vllm_http_multitenant.json")
    args = p.parse_args()
    result = asyncio.run(run_async(args))
    out, csv = write_json_csv(result, args.out, systems=["ttft_ms", "tpot_ms", "total_latency_ms"])
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"wrote {out}")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
