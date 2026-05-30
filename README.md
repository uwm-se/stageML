# StageML

StageML is a Python research prototype for specializing machine learning inference workloads. It focuses on LoRA and MoE-style adapter serving, where part of the computation can be prepared ahead of time and the remaining request-time computation is emitted as a residual program.

The repository includes the StageML Python package, MLIR lowering utilities, LoRA and MoE residualization code, benchmark scripts, configuration examples, and Lean proof artifacts.

## Requirements

StageML requires Python 3.11 or newer.

Core Python dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

For H100, vLLM, PEFT, and real-model experiments, use the H100 requirements file in a separate CUDA environment.

```bash
pip install -r requirements-h100.txt
```

For artifact-style experiments with Hugging Face models and Triton support, use:

```bash
pip install -r requirements_artifact.txt
```

Optional system dependencies for IREE and related systems experiments are listed in:

```text
requirements_systems_optional.txt
```

## Installation

Clone the repository and install the package in editable mode.

```bash
git clone <repository-url>
cd stageML_github

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Check that the package imports correctly.

```bash
python -c "import stageml; print(stageml.__all__)"
```

## Repository Layout

```text
stageML_github/
├── stageml/                  Core StageML package
├── benchmarks/               Benchmark and experiment scripts
├── tests/                    Unit and correctness tests
├── configs/                  Example model, adapter, and kernel manifests
├── csrc/                     Native extension source
├── Proofs/                   Lean proof files
├── requirements.txt          Core Python dependencies
├── requirements-h100.txt     H100 and vLLM experiment dependencies
├── requirements_artifact.txt Artifact evaluation dependencies
├── setup.py                  Python package setup
├── lakefile.lean             Lean project configuration
└── lean-toolchain            Lean toolchain version
```

## Main Components

### Core staging interface

The basic public interface is exposed from `stageml/__init__.py`.

```python
from stageml import stage0, stage1, compile_staged, BindingTime
```

The main staging files are:

```text
stageml/annotations.py
stageml/tracer.py
stageml/evaluator.py
stageml/runtime.py
stageml/rewrite.py
```

### Residual planning

Residualization and cost-based plan selection are implemented in:

```text
stageml/residual_planner.py
stageml/planner_validation.py
stageml/policy_optimizer.py
```

### LoRA and adapter serving

LoRA and adapter-bank logic are implemented in:

```text
stageml/adapter_bank.py
stageml/adapter_cache.py
stageml/peft_bridge.py
```

### MoE and multi-tenant serving

MoE staging and residual plans are implemented in:

```text
stageml/moe_stages.py
stageml/moe_ir.py
stageml/moe_lora_layers.py
stageml/moe_plan_export.py
stageml/vllm_stage_router.py
```

### MLIR and backend support

MLIR lowering and backend utilities are implemented in:

```text
stageml/mlir_lower.py
stageml/real_mlir_lower.py
stageml/canonical_mlir_lower.py
stageml/moe_mlir_lower.py
stageml/torch_mlir_backend.py
stageml/iree_residual_mlir.py
stageml/baremetal_backend.py
```

### GPU and generated-kernel support

GPU-specific and generated-kernel code is implemented in:

```text
stageml/triton_generator.py
stageml/sm90_native_backend.py
stageml/tools_compile_sm90_native.py
stageml/tools_compile_iree.py
stageml/tools_emit_iree_residual.py
```

### Quantization analysis

Quantization safety checks and experiments are implemented in:

```text
stageml/quant_absint.py
stageml/quant_experiments.py
```

## Running Tests

Run the full test suite.

```bash
pytest
```

Run tests with coverage.

```bash
pytest --cov=stageml --cov-report=term-missing
```

Generate an HTML coverage report.

```bash
pytest --cov=stageml --cov-report=html
```

The HTML report is written to:

```text
htmlcov/index.html
```

Useful targeted test commands:

```bash
pytest tests/test_annotations.py
pytest tests/test_residual_planner.py
pytest tests/test_rewrite_lora.py
pytest tests/test_moe_lora_correctness.py
pytest tests/test_quant_absint.py
pytest tests/test_canonical_mlir_lower.py
pytest tests/test_vllm_stage_router.py
```

## Running Benchmarks

Most benchmark scripts are under `benchmarks/`. Run them from the repository root.

### LoRA baseline benchmark

```bash
python benchmarks/lora_baselines_bench.py
```

### LoRA sweep

```bash
python benchmarks/lora_sweep.py
```

### GPU LoRA and MoE benchmark

```bash
python benchmarks/gpu_lora_moe_bench.py
```

Example with explicit parameters:

```bash
python benchmarks/gpu_lora_moe_bench.py \
  --dim 4096 \
  --rank 16 \
  --batch 1 \
  --dtype float16 \
  --warmup 30 \
  --iterations 100 \
  --out out/gpu_lora_moe_results.csv
```

### LLaMA-scale LoRA block benchmark

```bash
python benchmarks/llama_scale_lora_block_bench.py
```

Example:

```bash
python benchmarks/llama_scale_lora_block_bench.py \
  --hidden-size 4096 \
  --rank 16 \
  --batch 1 \
  --seq-len 128 \
  --dtype float16
```

### Torch compile comparison

```bash
python benchmarks/torch_compile_comparison.py
```

### Rewrite ablation benchmark

```bash
python benchmarks/rewrite_ablation_bench.py
```

### Multi-adapter benchmark

```bash
python benchmarks/multi_adapter_bank_bench.py
```

### Quantization benchmarks

```bash
python benchmarks/quantization_theta_sweep.py
python benchmarks/quantization_middle_band_bench.py
python benchmarks/quantized_weight_bench.py
```

### MLIR demos

```bash
python benchmarks/canonical_mlir_demo.py
python benchmarks/torch_mlir_demo.py
```

### Triton residual kernel checks

```bash
python benchmarks/triton_lora_equivalence_check.py
python benchmarks/triton_generated_residual_bench.py
```

## Reproducing the Main Experiment Bundle

The repository includes a runner for the main benchmark group.

```bash
python benchmarks/paper_full_runner.py --out-dir out/paper_run
```

Common options:

```bash
python benchmarks/paper_full_runner.py \
  --dim 4096 \
  --rank 16 \
  --batch 1 \
  --seq-lens 1 8 32 128 \
  --dtype float16 \
  --warmup 30 \
  --iterations 100 \
  --out-dir out/paper_run
```

The runner writes logs and summaries under:

```text
out/paper_run/
```

The command manifest is written to:

```text
out/paper_run/paper_run_manifest.txt
```

The combined CSV summary is written to:

```text
out/paper_run/combined_summary.csv
```

## Real-Model and H100 Experiments

Real-model experiments use Hugging Face, PEFT, vLLM, and CUDA dependencies. Install the H100 dependencies before running these scripts.

```bash
pip install -r requirements-h100.txt
```

Example scripts:

```bash
python benchmarks/peft_lora_layer_bench.py
python benchmarks/peft_stageml_full_model_bench.py
python benchmarks/peft_real_model_generate_bench.py
python benchmarks/real_peft_multi_adapter_bench.py
python benchmarks/vllm_http_multitenant_bench.py
python benchmarks/accepted_fusion_h100_bench.py
```

The repository includes example manifests in `configs/`.

```text
configs/model_adapter_manifest.example.json
configs/multimodel_manifest.h100.json
configs/stageml_kernel_manifest.example.json
```

The H100 manifest contains local paths such as `/data/stageml_h100_run/...`. Update those paths before running on another machine.

## MLIR, IREE, and Native Backend Tools

StageML includes utilities for emitting and compiling residual MLIR.

```bash
python -m stageml.tools_emit_iree_residual
python -m stageml.tools_compile_iree
python -m stageml.tools_compile_sm90_native
```

IREE support requires an external IREE installation. Verify that the compiler is available before using the IREE backend.

```bash
iree-compile --help
```

## Lean Proofs

Lean proof files are stored in `Proofs/`.

```text
Proofs/Stage.lean
Proofs/Residualization.lean
Proofs/Soundness.lean
Proofs/TensorShape.lean
```

Build the Lean project with:

```bash
lake build
```

## Native Extension Source

The repository contains one native source file:

```text
csrc/stageml_residual_op.cpp
```

This file is used for native residual operator experiments. Build behavior depends on the local CUDA, PyTorch, and compiler environment.

## Useful Development Commands

Run tests:

```bash
pytest
```

Run coverage:

```bash
pytest --cov=stageml --cov-report=term-missing
```

Run a small benchmark:

```bash
python benchmarks/lora_baselines_bench.py
```

Run the main benchmark bundle:

```bash
python benchmarks/paper_full_runner.py --out-dir out/paper_run
```

Build Lean proofs:

```bash
lake build
```

## Notes

Some experiments are hardware-dependent. CPU-only machines can run most unit tests and some small synthetic benchmarks, but H100, vLLM, Triton, IREE, and SM90 experiments require the matching GPU and system software.

Some benchmark scripts expect model files, adapter directories, prompt traces, or manifest files. Use the example files in `configs/` as templates and update local paths before running those experiments.
