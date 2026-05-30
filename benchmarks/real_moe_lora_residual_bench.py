from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from stageml.moe_ir import MoEAdapterMeta
from stageml.moe_lora_layers import DynamicMoELoRALayer, MaterializedMoELoRALayer, MoEAdapterSpec, normalize_routing_weights
from stageml.quant_absint import QuantizationConfig, lora_delta, analyze_residualization
from stageml.residual_planner import AdapterQuantInfo, PlannerConfig, choose_residual_plan
from stageml.moe_plan_export import write_plan


def _load_optional_transformers():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Install real benchmark dependencies with pip install -r requirements_real_benchmarks.txt") from exc
    return AutoModelForCausalLM, AutoTokenizer


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    files = list(path.glob("*.safetensors")) + list(path.glob("*.bin")) + list(path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no adapter checkpoint files found in {path}")
    f = files[0]
    if f.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError("Install safetensors to load adapter checkpoints") from exc
        return load_file(str(f))
    obj = torch.load(str(f), map_location="cpu")
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"unsupported checkpoint object in {f}")
    return obj


def _read_prompts(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    if not rows:
        raise ValueError("prompt trace is empty")
    for row in rows:
        if "prompt" not in row:
            raise ValueError("each prompt trace row must contain a prompt field")
    return rows


def _expert_index_from_name(name: str) -> int | None:
    m = re.search(r"experts[\.\_](\d+)", name)
    if m:
        return int(m.group(1))
    return None



def _tensor_from_module_weight(module_or_weight) -> torch.Tensor | None:
    """
    Return a real 2D torch tensor from a normal Linear weight or a bitsandbytes
    4 bit Linear weight. This is for benchmarking and correctness checks, not
    for production serving.
    """
    weight = getattr(module_or_weight, "weight", module_or_weight)

    if weight is None:
        return None

    try:
        if hasattr(weight, "dequantize"):
            t = weight.dequantize()
        else:
            t = weight
    except Exception:
        try:
            t = weight.detach()
        except Exception:
            return None

    if not isinstance(t, torch.Tensor):
        return None

    if t.ndim != 2:
        return None

    return t.detach().to(torch.float32).cpu()




def extract_expert_weight(
    model: torch.nn.Module,
    *,
    out_features: int | None = None,
    in_features: int | None = None,
    max_experts: int | None = None,
) -> tuple[torch.Tensor, str]:
    """
    Memory safe extraction for fused MixtralExperts.

    The observed H100 model layout is:

        model.layers.N.mlp.experts.gate_up_proj  [8, 28672, 4096]
        model.layers.N.mlp.experts.down_proj     [8, 4096, 14336]

    gate_up_proj packs w1 and w3:

        w1 = gate_up_proj[:, :14336, :]
        w3 = gate_up_proj[:, 14336:, :]

    This function selects w1 and copies only the requested expert slice to CPU
    before converting to float32. This avoids allocating an extra 3.5 GB fp32
    copy on the GPU.
    """
    for mod_name, module in model.named_modules():
        if not mod_name.endswith(".mlp.experts"):
            continue

        gate_up = getattr(module, "gate_up_proj", None)
        if gate_up is None:
            continue

        if not isinstance(gate_up, torch.Tensor):
            gate_up = getattr(gate_up, "weight", gate_up)

        if not isinstance(gate_up, torch.Tensor):
            continue

        if gate_up.ndim != 3:
            continue

        if gate_up.shape[1] % 2 != 0:
            continue

        num_total_experts = int(gate_up.shape[0])
        hidden = int(gate_up.shape[2])
        intermediate = int(gate_up.shape[1] // 2)

        n = num_total_experts
        if max_experts is not None:
            n = min(n, int(max_experts))

        # IMPORTANT:
        # Slice first, move bf16 slice to CPU, then convert to fp32 on CPU.
        # Do not call .to(torch.float32) while the tensor is still on GPU.
        w1_cpu = gate_up[:n, :intermediate, :].detach().cpu().to(torch.float32).contiguous()

        print("found_fused_mixtral_gate_up_proj", mod_name, tuple(gate_up.shape))
        print("selected_projection", "w1")
        print("selected_expert_indices", list(range(n)))
        print("selected_expert_weight_shape", tuple(w1_cpu.shape))

        return w1_cpu, f"{mod_name}.w1.weight"

    raise RuntimeError("could not find fused MixtralExperts gate_up_proj tensor")

def _normalize_lora_key(key: str) -> str:
    key = key.replace(".default.", ".")
    key = key.replace("base_model.model.", "")
    return key


def load_expert_lora_adapter(adapter_dir: Path, *, num_experts: int, in_features: int, out_features: int) -> MoEAdapterSpec:
    sd = {_normalize_lora_key(k): v for k, v in _load_state_dict(adapter_dir).items() if isinstance(v, torch.Tensor)}
    pairs: dict[int, dict[str, torch.Tensor]] = {i: {} for i in range(num_experts)}
    preferred_projection = os.environ.get("LORA_PROJECTION", "w1").strip().lower()
    for key, tensor in sd.items():
        expert = _expert_index_from_name(key)
        if expert is None or expert >= num_experts:
            continue
        lk = key.lower()

        # Prefer the projection matching the extracted fused base weight.
        # Default is w1.
        if preferred_projection and f".{preferred_projection}." not in lk:
            continue

        if "lora_a" in lk and tensor.ndim == 2 and tensor.shape[1] == in_features:
            pairs[expert]["A"] = tensor.detach().cpu().to(torch.float32)
        if "lora_b" in lk and tensor.ndim == 2 and tensor.shape[0] == out_features:
            pairs[expert]["B"] = tensor.detach().cpu().to(torch.float32)
    ranks = [v["A"].shape[0] for v in pairs.values() if "A" in v and "B" in v]
    if not ranks:
        raise RuntimeError(f"could not find expert LoRA A and B tensors in {adapter_dir}")
    rank = max(set(ranks), key=ranks.count)
    A = torch.zeros(num_experts, rank, in_features, dtype=torch.float32)
    B = torch.zeros(num_experts, out_features, rank, dtype=torch.float32)
    found = 0
    for e, pair in pairs.items():
        if "A" not in pair or "B" not in pair:
            continue
        a = pair["A"]
        b = pair["B"]
        if a.shape[0] != rank or b.shape[1] != rank:
            continue
        A[e] = a
        B[e] = b
        found += 1
    if found == 0:
        raise RuntimeError(f"found adapter tensors in {adapter_dir} but no tensors matched rank and shape")
    return MoEAdapterSpec(name=adapter_dir.name, A=A, B=B, scaling=1.0)



def _tensor_to_cpu_float(t: torch.Tensor) -> torch.Tensor:
    """
    Move tensor to CPU first, then convert to float32.
    This avoids extra fp32 allocation on the GPU.
    """
    return t.detach().cpu().to(torch.float32).contiguous()





def _find_gate_weight(model: torch.nn.Module, num_experts: int, hidden_dim: int):
    """
    Find a Mixtral router/gate weight.

    Full Mixtral router shape is [8, hidden_dim]. Smoke runs may use only
    the first MAX_EXPERTS experts, for example 4, so this accepts router
    weights with at least num_experts rows and slices them.
    """
    candidates = []

    def consider(name: str, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            return

        shape = tuple(tensor.shape)
        orientation = None

        if shape[1] == hidden_dim and shape[0] >= num_experts and shape[0] <= 64:
            orientation = "out_in"
        elif shape[0] == hidden_dim and shape[1] >= num_experts and shape[1] <= 64:
            orientation = "in_out"

        if orientation is None:
            return

        lname = name.lower()
        score = 0

        if "gate" in lname:
            score += 100
        if "router" in lname:
            score += 100
        if ".mlp." in lname:
            score += 50
        if "layers.0" in lname:
            score += 20

        if "experts" in lname:
            score -= 100
        if "lora" in lname:
            score -= 100
        if "embed" in lname:
            score -= 100
        if "lm_head" in lname:
            score -= 100

        candidates.append((score, name, tensor, orientation))

    for name, param in model.named_parameters():
        consider(name, param)

    for name, tensor in model.state_dict().items():
        consider(name, tensor)

    if not candidates:
        sample = []
        for name, tensor in model.state_dict().items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                s = tuple(tensor.shape)
                if hidden_dim in s or num_experts in s:
                    sample.append(f"{name} shape={s}")
                    if len(sample) >= 80:
                        break
        raise RuntimeError(
            "could not find a router/gate weight. Candidate 2D tensors:\n"
            + "\n".join(sample)
        )

    candidates.sort(key=lambda x: x[0], reverse=True)
    score, name, tensor, orientation = candidates[0]

    gate = tensor.detach().cpu().to(torch.float32).contiguous()

    if orientation == "out_in":
        gate = gate[:num_experts, :]
    elif orientation == "in_out":
        gate = gate[:, :num_experts]

    print("selected_gate_weight", name)
    print("selected_gate_weight_original_shape", tuple(tensor.shape))
    print("selected_gate_weight_used_shape", tuple(gate.shape))
    print("selected_gate_orientation", orientation)

    return gate, name, orientation



def collect_real_hidden_states(model, tokenizer, rows, *, layer_index: int, device: str, max_tokens: int) -> torch.Tensor:
    """
    Collect real hidden states from real prompts using microbatches.

    This avoids H100 OOM when the full 4 bit Mixtral model already occupies
    most GPU memory.
    """
    texts = []
    for r in rows:
        if isinstance(r, dict):
            texts.append(r.get("prompt") or r.get("text") or str(r))
        else:
            texts.append(str(r))

    batch_size = int(os.environ.get("HIDDEN_BATCH_SIZE", "1"))
    chunks = []

    for i in range(0, len(texts), batch_size):
        sub = texts[i:i + batch_size]

        batch = tokenizer(
            sub,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_tokens,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = model(**batch, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states[layer_index].detach().cpu().to(torch.float32)
        mask = batch["attention_mask"].detach().cpu().bool()
        chunks.append(hs[mask].contiguous())

        del out, hs, mask, batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0).contiguous()

def make_routing(model, hidden: torch.Tensor, num_experts: int, top_k: int, device: str) -> tuple[torch.Tensor, torch.Tensor, str]:
    """
    Compute real MoE routing from the discovered router/gate weight.

    Routing is computed on CPU to avoid extra H100 memory pressure after the
    full 4 bit Mixtral model is loaded.
    """
    hidden_cpu = hidden.detach().cpu().to(torch.float32).contiguous()
    hidden_dim = int(hidden_cpu.shape[-1])

    gate_weight, gate_name, orientation = _find_gate_weight(model, num_experts, hidden_dim)

    with torch.no_grad():
        if orientation == "out_in":
            logits = hidden_cpu @ gate_weight.t()
        elif orientation == "in_out":
            logits = hidden_cpu @ gate_weight
        else:
            raise RuntimeError(f"unknown gate orientation {orientation}")

        weights, ids = torch.topk(torch.softmax(logits, dim=-1), k=top_k, dim=-1)

    print("routing_gate_name", gate_name)
    print("routing_top_k", top_k)
    print("routing_ids_shape", tuple(ids.shape))

    return ids.long(), normalize_routing_weights(ids, weights), str(gate_name)

def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round((pct / 100.0) * (len(xs) - 1)))))
    return float(xs[idx])


def time_layer(layer, x, expert_ids, routing_weights, adapter_ids, repeats: int) -> dict[str, float]:
    times = []
    y = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        y = layer(x, expert_ids, routing_weights, adapter_ids)
        if x.device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    assert y is not None
    mean = sum(times) / len(times)
    std = (sum((t - mean) ** 2 for t in times) / (len(times) - 1)) ** 0.5 if len(times) > 1 else 0.0
    return {
        "runs": len(times),
        "p50_ms": percentile(times, 50) * 1000.0,
        "p95_ms": percentile(times, 95) * 1000.0,
        "mean_ms": mean * 1000.0,
        "std_ms": std * 1000.0,
        "min_ms": min(times) * 1000.0,
        "max_ms": max(times) * 1000.0,
    }


def _gpu_metadata() -> dict[str, Any]:
    meta = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        meta.update({
            "gpu_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "gpu_count": torch.cuda.device_count(),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        })
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, timeout=10)
        meta["nvidia_smi"] = out.strip()
    except Exception:
        pass
    return meta


def _load_shared_lora_adapter(adapter_dir: Path, *, num_experts: int, in_features: int, out_features: int) -> MoEAdapterSpec:
    sd = {_normalize_lora_key(k): v for k, v in _load_state_dict(adapter_dir).items() if isinstance(v, torch.Tensor)}
    a_candidates = []
    b_candidates = []
    for key, tensor in sd.items():
        lk = key.lower()
        if "lora_a" in lk and tensor.ndim == 2 and tensor.shape[1] == in_features:
            a_candidates.append((key, tensor.detach().cpu().to(torch.float32)))
        if "lora_b" in lk and tensor.ndim == 2 and tensor.shape[0] == out_features:
            b_candidates.append((key, tensor.detach().cpu().to(torch.float32)))
    if not a_candidates or not b_candidates:
        raise RuntimeError(f"no compatible shared LoRA tensors found in {adapter_dir}")
    A0 = a_candidates[0][1]
    rank = int(A0.shape[0])
    B0 = None
    for _, b in b_candidates:
        if b.shape[1] == rank:
            B0 = b
            break
    if B0 is None:
        raise RuntimeError(f"found shared LoRA tensors in {adapter_dir} but ranks did not match")
    A = A0.unsqueeze(0).repeat(num_experts, 1, 1).contiguous()
    B = B0.unsqueeze(0).repeat(num_experts, 1, 1).contiguous()
    return MoEAdapterSpec(name=adapter_dir.name, A=A, B=B, scaling=1.0)


def _load_adapter_with_optional_fallback(adapter_dir: Path, *, num_experts: int, in_features: int, out_features: int, allow_shared_fallback: bool) -> tuple[MoEAdapterSpec, str]:
    try:
        return load_expert_lora_adapter(adapter_dir, num_experts=num_experts, in_features=in_features, out_features=out_features), "expert_specific"
    except Exception:
        if not allow_shared_fallback:
            raise
        return _load_shared_lora_adapter(adapter_dir, num_experts=num_experts, in_features=in_features, out_features=out_features), "shared_reused_across_experts"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter-dirs", nargs="+", required=True)
    ap.add_argument("--prompts-jsonl", required=True)
    ap.add_argument("--out-dir", default="out/real_moe_lora")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--layer-index", type=int, default=1)
    ap.add_argument("--max-prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--max-experts", type=int, default=8)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--memory-budget-mb", type=float, default=256.0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--load-in-4bit", action="store_true", help="Load the Hugging Face MoE model with bitsandbytes 4 bit quantization for Colab sized GPUs")
    ap.add_argument("--device-map", default=None, help="Optional Hugging Face device map. Use auto with load in 4 bit on Colab")
    ap.add_argument("--allow-shared-lora-fallback", action="store_true", help="If an adapter is not expert specific, reuse the same LoRA tensors across experts for smoke testing. Do not use this for the main expert specific paper claim.")
    args = ap.parse_args()

    def normalize_device_map(value):
        if value is None:
            return None
        v = str(value).strip()
        if v == "" or v.lower() == "none":
            return None
        if v.lower() in {"cuda", "cuda:0", "single", "single_gpu", "gpu", "0"}:
            return {"": 0}
        return v

    normalized_device_map = normalize_device_map(args.device_map)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    AutoModelForCausalLM, AutoTokenizer = _load_optional_transformers()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:
            raise RuntimeError("Install bitsandbytes and a recent transformers version to use --load-in-4bit") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = normalized_device_map if normalized_device_map is not None else {"": 0}
    else:
        model_kwargs["torch_dtype"] = torch.float16 if args.device == "cuda" else torch.float32
        model_kwargs["device_map"] = normalized_device_map
    print("model_kwargs_device_map", model_kwargs.get("device_map"))
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    print("hf_device_map", getattr(model, "hf_device_map", "none"))
    if not args.load_in_4bit and args.device_map is None:
        model.to(args.device)
    model.eval()

    rows = _read_prompts(Path(args.prompts_jsonl), args.max_prompts)
    if any("adapter" not in row for row in rows):
        raise ValueError("each prompt row must contain an adapter field matching one adapter directory name")
    hidden = collect_real_hidden_states(model, tokenizer, rows, layer_index=args.layer_index, device=args.device, max_tokens=args.max_tokens)
    expert_weight, expert_name = extract_expert_weight(model, max_experts=args.max_experts)
    num_experts, out_features, in_features = expert_weight.shape
    adapter_load_modes = {}
    adapters = []
    for p in args.adapter_dirs:
        spec, mode = _load_adapter_with_optional_fallback(
            Path(p),
            num_experts=num_experts,
            in_features=in_features,
            out_features=out_features,
            allow_shared_fallback=args.allow_shared_lora_fallback,
        )
        adapters.append(spec)
        adapter_load_modes[spec.name] = mode
    names = {a.name for a in adapters}
    adapter_ids = []
    for row in rows:
        if row["adapter"] not in names:
            raise ValueError(f"prompt adapter {row['adapter']} is not one of {sorted(names)}")
    flat_prompts = []
    encoded = tokenizer([r["prompt"] for r in rows], return_tensors="pt", padding=True, truncation=True, max_length=args.max_tokens)
    for i, row in enumerate(rows):
        count = int(encoded["attention_mask"][i].sum().item())
        flat_prompts.extend([row["adapter"]] * count)
    adapter_ids = flat_prompts[: hidden.shape[0]]
    expert_ids, routing_weights, gate_name = make_routing(model, hidden, num_experts, args.top_k, args.device)

    expert_weight = expert_weight.to(torch.float32)
    hidden = hidden[:, :in_features].contiguous()
    dyn = DynamicMoELoRALayer(expert_weight, adapters)
    mat = MaterializedMoELoRALayer(expert_weight, adapters)
    y_dyn = dyn(hidden, expert_ids, routing_weights, adapter_ids)
    y_mat = mat(hidden, expert_ids, routing_weights, adapter_ids)
    max_error = float((y_dyn - y_mat).abs().max().item())

    dyn_time = time_layer(dyn, hidden, expert_ids, routing_weights, adapter_ids, args.repeats)
    mat_time = time_layer(mat, hidden, expert_ids, routing_weights, adapter_ids, args.repeats)

    counts = {name: adapter_ids.count(name) for name in names}
    metas = [MoEAdapterMeta(a.name, a.rank, a.num_experts, a.in_features, a.out_features, dtype="fp16", request_count=counts[a.name]) for a in adapters]
    qinfo = {}
    qconfig = QuantizationConfig(bits=4, per_channel=True, channel_dim=0)
    for adapter in adapters:
        eps = []
        for e in range(num_experts):
            delta = lora_delta(adapter.A[e], adapter.B[e], adapter.scaling)
            bound = analyze_residualization(expert_weight[e], delta, theta=args.theta, config=qconfig)
            eps.append(bound.epsilon_weight_fro)
        qinfo[adapter.name] = AdapterQuantInfo(epsilon=max(eps), safe=max(eps) <= args.theta)
    plan = choose_residual_plan(metas, config=PlannerConfig(memory_budget_mb=args.memory_budget_mb, theta=args.theta), quant_info=qinfo)
    write_plan(plan, out_dir / "residual_plan.json")

    result = {
        "model": args.model,
        "expert_weight_name": expert_name,
        "gate_name": gate_name,
        "num_tokens": int(hidden.shape[0]),
        "num_experts": int(num_experts),
        "top_k": int(args.top_k),
        "adapters": sorted(names),
        "adapter_counts": counts,
        "max_abs_error_dynamic_vs_materialized": max_error,
        "dynamic": dyn_time,
        "materialized": mat_time,
        "plan_summary": plan.by_kind(),
        "plan_path": str(out_dir / "residual_plan.json"),
        "memory_budget_mb": float(args.memory_budget_mb),
        "theta": float(args.theta),
        "adapter_load_modes": adapter_load_modes,
        "adapter_metas": [
            {
                "name": m.name,
                "rank": m.rank,
                "num_experts": m.num_experts,
                "in_features": m.in_features,
                "out_features": m.out_features,
                "dtype": m.dtype,
                "request_count": m.request_count,
                "factor_numel": m.factor_numel,
                "materialized_numel": m.materialized_numel,
            }
            for m in metas
        ],
        "quantization": {
            name: {"epsilon": qi.epsilon, "safe": qi.safe} for name, qi in qinfo.items()
        },
        "environment": _gpu_metadata(),
    }
    (out_dir / "real_moe_lora_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
