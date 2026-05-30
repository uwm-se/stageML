namespace StageML

inductive Stage where
  | base
  | adapter
  | tenant
  | request
  | routing
  | token
  deriving DecidableEq, Repr

open Stage

def rank : Stage -> Nat
  | base => 0
  | adapter => 1
  | tenant => 2
  | request => 3
  | routing => 4
  | token => 5

def leStage (a b : Stage) : Prop := rank a <= rank b

def join : Stage -> Stage -> Stage
  | base, s => s
  | s, base => s
  | adapter, s => if rank adapter <= rank s then s else adapter
  | s, adapter => if rank s <= rank adapter then adapter else s
  | tenant, s => if rank tenant <= rank s then s else tenant
  | s, tenant => if rank s <= rank tenant then tenant else s
  | request, s => if rank request <= rank s then s else request
  | s, request => if rank s <= rank request then request else s
  | routing, s => if rank routing <= rank s then s else routing
  | s, routing => if rank s <= rank routing then routing else s
  | token, _ => token

theorem join_base_left (s : Stage) : join base s = s := by
  cases s <;> rfl

theorem join_base_right (s : Stage) : join s base = s := by
  cases s <;> rfl

theorem join_token_left (s : Stage) : join token s = token := by
  cases s <;> rfl

theorem join_token_right (s : Stage) : join s token = token := by
  cases s <;> simp [join, rank]

theorem join_idem (s : Stage) : join s s = s := by
  cases s <;> simp [join, rank]

theorem adapter_before_request : leStage adapter request := by
  unfold leStage rank
  decide

theorem request_before_token : leStage request token := by
  unfold leStage rank
  decide

theorem base_before_all (s : Stage) : leStage base s := by
  cases s <;> unfold leStage rank <;> decide

theorem all_before_token (s : Stage) : leStage s token := by
  cases s <;> unfold leStage rank <;> decide

-- This is the binding time soundness fact used by the paper at the lattice level.
theorem folded_at_adapter_has_no_token_stage (s : Stage) :
    leStage s adapter -> s ≠ token := by
  intro h
  cases s <;> simp [leStage, rank] at h ⊢

end StageML
