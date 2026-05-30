namespace StageML

structure Shape where
  rows : Nat
  cols : Nat
  deriving DecidableEq, Repr

inductive WellMatmul : Shape -> Shape -> Shape -> Prop where
  | mk {m k n : Nat} :
      WellMatmul { rows := m, cols := k } { rows := k, cols := n } { rows := m, cols := n }

inductive WellAdd : Shape -> Shape -> Shape -> Prop where
  | mk {m n : Nat} :
      WellAdd { rows := m, cols := n } { rows := m, cols := n } { rows := m, cols := n }

theorem matmul_shape_preserved {m k n : Nat} :
    WellMatmul { rows := m, cols := k } { rows := k, cols := n } { rows := m, cols := n } := by
  exact WellMatmul.mk

theorem add_shape_preserved {m n : Nat} :
    WellAdd { rows := m, cols := n } { rows := m, cols := n } { rows := m, cols := n } := by
  exact WellAdd.mk

-- If two matrix multiplications are well shaped, the LoRA update x A B is well shaped.
theorem lora_update_shape {batch input rank output : Nat} :
    WellMatmul
      { rows := batch, cols := input }
      { rows := input, cols := rank }
      { rows := batch, cols := rank }
    /\
    WellMatmul
      { rows := batch, cols := rank }
      { rows := rank, cols := output }
      { rows := batch, cols := output } := by
  constructor
  · exact WellMatmul.mk
  · exact WellMatmul.mk

theorem residual_weight_shape {input rank output : Nat} :
    WellMatmul { rows := input, cols := rank } { rows := rank, cols := output } { rows := input, cols := output } := by
  exact WellMatmul.mk

end StageML
