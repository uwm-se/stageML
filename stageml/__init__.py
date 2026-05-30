"""
StageML — Multi-Stage Programming DSL for ML Inference
with an MLIR Compiler Backend.

The package initializer is intentionally lightweight. Some optional runtime
modules depend on PyTorch, PEFT, vLLM, or CUDA. Backend-only tools such as the
IREE compiler wrapper must still import stageml on machines or virtual
environments where those optional runtime dependencies are not installed.
"""

from stageml.annotations import stage0, stage1, compile_staged, BindingTime

__all__ = [
    "stage0",
    "stage1",
    "compile_staged",
    "BindingTime",
]

try:
    from stageml.runtime import compile_model, StagingReport
    __all__.extend(["compile_model", "StagingReport"])
except Exception:
    # PyTorch is optional for backend-only tools such as IREE compilation.
    pass

try:
    from stageml.rewrite import optimize_evaluation_order, RewriteStats
    __all__.extend(["optimize_evaluation_order", "RewriteStats"])
except Exception:
    pass

try:
    from stageml.real_mlir_lower import lower_to_parseable_mlir, write_parseable_mlir
    __all__.extend(["lower_to_parseable_mlir", "write_parseable_mlir"])
except Exception:
    # real_mlir_lower imports torch.fx, so it is unavailable in the lightweight
    # IREE compiler venv unless torch is also installed.
    pass

try:
    from stageml.canonical_mlir_lower import (
        lower_to_canonical_mlir,
        write_canonical_mlir,
        verify_canonical_mlir_with_mlir_opt,
    )
    __all__.extend([
        "lower_to_canonical_mlir",
        "write_canonical_mlir",
        "verify_canonical_mlir_with_mlir_opt",
    ])
except Exception:
    pass

# PEFT bridge helpers for full-model StageML replacement benchmarks.
try:
    from .peft_bridge import replace_lora_layers_with_stageml, ReplacementStats
    __all__.extend(["replace_lora_layers_with_stageml", "ReplacementStats"])
except Exception:
    pass

try:
    from stageml.moe_stages import MoEStage, join as moe_join
    from stageml.moe_lora_layers import DynamicMoELoRALayer, MaterializedMoELoRALayer, MoEAdapterSpec
    from stageml.quant_absint import QuantizationConfig, analyze_residualization, safe_to_residualize
    from stageml.residual_planner import PlannerConfig, choose_residual_plan
    __all__.extend([
        "MoEStage",
        "moe_join",
        "DynamicMoELoRALayer",
        "MaterializedMoELoRALayer",
        "MoEAdapterSpec",
        "QuantizationConfig",
        "analyze_residualization",
        "safe_to_residualize",
        "PlannerConfig",
        "choose_residual_plan",
    ])
except Exception:
    pass
