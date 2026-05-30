import Lake
open Lake DSL

package StageML where


lean_lib StageML where
  roots := #[
    `Proofs.Stage,
    `Proofs.TensorShape,
    `Proofs.Residualization,
    `Proofs.Soundness
  ]
