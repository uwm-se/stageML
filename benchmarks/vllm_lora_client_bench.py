import argparse
import csv
import json
import os
import statistics
import time
from pathlib import Path

import requests
from transformers import AutoTokenizer


def read_trace(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt") or obj.get("text") or str(obj)
            rows.append(prompt)
    return rows


def find_adapters(adapter_dir: Path):
    adapters = []
    for p in adapter_dir.iterdir():
        if p.is_dir() and (p / "adapter_config.json").exists():
            adapters.append(p)
    return sorted(adapters)


def truncate_prompt(tokenizer, prompt: str, max_prompt_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if len(ids) <= max_prompt_tokens:
        return prompt
    ids = ids[:max_prompt_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def post_json(url, payload, timeout):
    r = requests.post(url, json=payload, timeout=timeout)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--num-requests", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-model-len", type=int, default=int(os.environ.get("VLLM_MAX_MODEL_LEN", "192")))
    ap.add_argument("--warmup", type=int, default=int(os.environ.get("VLLM_WARMUP", "2")))
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--load-adapters", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    trace = Path(args.trace)
    adapter_dir = Path(args.adapter_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_name = os.environ.get("VLLM_MODEL") or os.environ.get("MODEL_PATH") or "mistralai/Mixtral-8x7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    prompts = read_trace(trace)
    if not prompts:
        raise RuntimeError(f"No prompts found in {trace}")

    adapters = find_adapters(adapter_dir)
    adapter_name = None

    if args.load_adapters:
        for adapter in adapters:
            adapter_name = adapter.name
            payload = {
                "lora_name": adapter.name,
                "lora_path": str(adapter.resolve()),
            }
            r = post_json(f"{base_url}/v1/load_lora_adapter", payload, args.timeout)
            print(f"loaded {adapter.name}: {r.text}")
            if r.status_code >= 400:
                raise RuntimeError(f"Failed to load LoRA adapter {adapter}: {r.text}")
            break

    if adapter_name is None:
        models = requests.get(f"{base_url}/v1/models", timeout=args.timeout).json()
        adapter_name = models["data"][0]["id"]

    max_prompt_tokens = max(1, args.max_model_len - args.max_tokens - 8)

    needed = args.num_requests + args.warmup
    repeated_prompts = []
    while len(repeated_prompts) < needed:
        repeated_prompts.extend(prompts)

    clean_prompts = [
        truncate_prompt(tokenizer, p, max_prompt_tokens)
        for p in repeated_prompts[:needed]
    ]

    def one_request(prompt):
        payload = {
            "model": adapter_name,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
        }
        t0 = time.perf_counter()
        r = post_json(f"{base_url}/v1/completions", payload, args.timeout)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        return r, dt_ms

    for p in clean_prompts[: args.warmup]:
        r, dt = one_request(p)
        print(f"warmup status={r.status_code} ms={dt:.3f}")
        if r.status_code >= 400:
            print(r.text[:500])

    times = []
    errors = 0

    measured_prompts = clean_prompts[args.warmup : args.warmup + args.num_requests]
    if len(measured_prompts) < args.num_requests:
        measured_prompts = clean_prompts[: args.num_requests]

    for p in measured_prompts:
        r, dt = one_request(p)
        if r.status_code >= 400:
            errors += 1
            print(f"error {r.status_code}: {r.text[:500]}")
        else:
            times.append(dt)

    measured = len(times)
    if measured:
        p50 = statistics.median(times)
        p95 = sorted(times)[max(0, int(0.95 * len(times)) - 1)]
        mean = statistics.mean(times)
        rps = measured / (sum(times) / 1000.0)
        tps = (measured * args.max_tokens) / (sum(times) / 1000.0)
    else:
        p50 = p95 = mean = rps = tps = 0.0

    row = {
        "system": "vllm_lora",
        "num_requests": args.num_requests,
        "measured_requests": measured,
        "errors": errors,
        "adapters": len(adapters),
        "requests_per_sec": rps,
        "tokens_per_sec": tps,
        "p50_ms": p50,
        "p95_ms": p95,
        "mean_ms": mean,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "model": model_name,
        "adapter": adapter_name,
    }

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    print(f"wrote={out}")
    print(row)


if __name__ == "__main__":
    main()
