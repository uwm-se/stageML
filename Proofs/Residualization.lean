set_option autoImplicit false

namespace StageML

structure Mat where
  rows : Nat
  cols : Nat

axiom matmul : Mat -> Mat -> Mat
axiom matAdd : Mat -> Mat -> Mat
axiom scalarMul : Nat -> Mat -> Mat

axiom matmul_add_right :
  forall (x w d : Mat),
    matmul x (matAdd w d) =
    matAdd (matmul x w) (matmul x d)

axiom scalar_push :
  forall (alpha : Nat) (x a b : Mat),
    matmul x (scalarMul alpha (matmul a b)) =
    scalarMul alpha (matmul (matmul x a) b)

theorem exact_lora_residualization
    (x w a b : Mat)
    (alpha : Nat) :
    matmul x (matAdd w (scalarMul alpha (matmul a b))) =
    matAdd
      (matmul x w)
      (scalarMul alpha (matmul (matmul x a) b)) := by
  calc
    matmul x (matAdd w (scalarMul alpha (matmul a b)))
        = matAdd
            (matmul x w)
            (matmul x (scalarMul alpha (matmul a b))) := by
              rw [matmul_add_right]
    _   = matAdd
            (matmul x w)
            (scalarMul alpha (matmul (matmul x a) b)) := by
              rw [scalar_push]

end StageML
