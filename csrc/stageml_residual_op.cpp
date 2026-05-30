#include <torch/extension.h>

// This is a deliberately small registration skeleton. The production version
// should launch the IREE/Triton/PTX artifact selected by the StageML manifest.
// Until that launcher is implemented, this op throws so benchmark code cannot
// accidentally report Python fallback numbers as bare-metal numbers.

torch::Tensor residual_moe(
    torch::Tensor hidden,
    torch::Tensor gate_up,
    torch::Tensor down,
    torch::Tensor topk_weights,
    torch::Tensor topk_ids) {
  TORCH_CHECK(false, "StageML C++ residual_moe launcher is a skeleton. Set STAGEML_CUSTOM_OP_LIB only after wiring the PTX/IREE launcher.");
}

TORCH_LIBRARY(stageml, m) {
  m.def("residual_moe(Tensor hidden, Tensor gate_up, Tensor down, Tensor topk_weights, Tensor topk_ids) -> Tensor");
}

TORCH_LIBRARY_IMPL(stageml, CUDA, m) {
  m.impl("residual_moe", &residual_moe);
}

TORCH_LIBRARY_IMPL(stageml, CPU, m) {
  m.impl("residual_moe", &residual_moe);
}
