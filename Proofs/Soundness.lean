-- Proofs/Soundness.lean
-- StageML Formal Proof: Binding-Time Soundness and Specialization Preservation
----
-- It formalizes:
--   1. The two-point binding-time lattice S <= D.
--   2. Join-based stage propagation.
--   3. A small expression IR with static variables, dynamic variables, constants, and addition.
--   4. A binding-time analysis for that IR.
--   5. A partial evaluator specializer.
--   6. Semantic preservation of specialization.


namespace StageML

inductive BindingTime : Type
  | S : BindingTime   -- static, compile-time
  | D : BindingTime   -- dynamic, runtime
  deriving DecidableEq, Repr

namespace BindingTime

def le : BindingTime -> BindingTime -> Prop
  | .S, _  => True
  | .D, .D => True
  | .D, .S => False

def join : BindingTime -> BindingTime -> BindingTime
  | .S, .S => .S
  | .S, .D => .D
  | .D, .S => .D
  | .D, .D => .D

theorem join_comm (a b : BindingTime) :
    join a b = join b a := by
  cases a <;> cases b <;> rfl

theorem join_assoc (a b c : BindingTime) :
    join (join a b) c = join a (join b c) := by
  cases a <;> cases b <;> cases c <;> rfl

theorem join_idempotent (a : BindingTime) :
    join a a = a := by
  cases a <;> rfl

theorem join_D_right (a : BindingTime) :
    join a .D = .D := by
  cases a <;> rfl

theorem join_D_left (a : BindingTime) :
    join .D a = .D := by
  cases a <;> rfl

theorem join_S_iff (a b : BindingTime) :
    join a b = .S <-> a = .S /\ b = .S := by
  cases a <;> cases b <;> simp [join]

end BindingTime

open BindingTime

-- A minimal StageML expression IR.
--
-- const n    : known literal
-- sVar i     : static input variable, known at specialization time
-- dVar i     : dynamic input variable, known only at runtime
-- add e1 e2  : arithmetic node
--
-- This is intentionally small. It is enough to prove the central staging
-- property without mixing the proof with implementation details of pandas,
-- MoE kernels, or vLLM.

inductive Expr : Type
  | const : Nat -> Expr
  | sVar  : Nat -> Expr
  | dVar  : Nat -> Expr
  | add   : Expr -> Expr -> Expr
  deriving Repr, DecidableEq

open Expr

abbrev Env := Nat -> Nat

-- Denotational semantics for the small IR.

def eval (senv denv : Env) : Expr -> Nat
  | const n   => n
  | sVar i    => senv i
  | dVar i    => denv i
  | add a b   => eval senv denv a + eval senv denv b

-- Binding-time analysis.
--
-- Static literals and static variables are S.
-- Dynamic variables are D.
-- Addition takes the join of the operand stages.

def bta : Expr -> BindingTime
  | const _ => .S
  | sVar _  => .S
  | dVar _  => .D
  | add a b => BindingTime.join (bta a) (bta b)

-- Local soundness property used by the graph-level intuition:
-- if an addition node is static, both operands are static.
-- Equivalently, no stage-0 node has a stage-1 operand.

theorem add_node_staging_soundness (a b : Expr) :
    bta (add a b) = .S -> bta a = .S /\ bta b = .S := by
  intro h
  exact (BindingTime.join_S_iff (bta a) (bta b)).mp h

-- Contrapositive forms: if either operand is dynamic, the addition node is dynamic.

theorem any_dynamic_gives_dynamic_left (a b : Expr) :
    bta a = .D -> bta (add a b) = .D := by
  intro h
  simp [bta, h, BindingTime.join_D_left]

theorem any_dynamic_gives_dynamic_right (a b : Expr) :
    bta b = .D -> bta (add a b) = .D := by
  intro h
  simp [bta, h, BindingTime.join_D_right]

-- Monotonicity for one operand: replacing a static operand by a dynamic operand
-- cannot make the result more static.

theorem propagation_monotone_left (b : BindingTime) :
    BindingTime.le (BindingTime.join .S b) (BindingTime.join .D b) := by
  cases b <;> simp [BindingTime.join, BindingTime.le]

theorem propagation_monotone_right (a : BindingTime) :
    BindingTime.le (BindingTime.join a .S) (BindingTime.join a .D) := by
  cases a <;> simp [BindingTime.join, BindingTime.le]

-- Partial evaluator helper.
-- It folds addition only when both residual operands are constants.

def foldAdd : Expr -> Expr -> Expr
  | const x, const y => const (x + y)
  | a, b             => add a b

theorem eval_foldAdd (senv denv : Env) (a b : Expr) :
    eval senv denv (foldAdd a b) = eval senv denv a + eval senv denv b := by
  cases a <;> cases b <;> simp [foldAdd, eval]

-- Partial evaluator.
--
-- Static variables are replaced by constants from the static environment.
-- Dynamic variables are kept.
-- If both children of an addition specialize to constants, the addition is
-- folded. Otherwise, the residual addition is preserved.

def specialize (senv : Env) : Expr -> Expr
  | const n => const n
  | sVar i  => const (senv i)
  | dVar i  => dVar i
  | add a b => foldAdd (specialize senv a) (specialize senv b)

-- Specialized programs contain no static variables.
-- This is useful because the residual program should depend only on dynamic
-- inputs and constants.

def NoStaticVars : Expr -> Prop
  | const _ => True
  | sVar _  => False
  | dVar _  => True
  | add a b => NoStaticVars a /\ NoStaticVars b

theorem noStatic_foldAdd {a b : Expr} :
    NoStaticVars a -> NoStaticVars b -> NoStaticVars (foldAdd a b) := by
  intro ha hb
  cases a <;> cases b <;>
    simp [foldAdd, NoStaticVars] at ha hb ⊢
  case const.add =>
    exact hb
  case dVar.add =>
    exact hb
  case add.const =>
    exact ha
  case add.dVar =>
    exact ha
  case add.add =>
    exact And.intro ha hb

theorem specialize_has_no_static_vars (senv : Env) :
    forall e : Expr, NoStaticVars (specialize senv e) := by
  intro e
  induction e with
  | const n =>
      simp [specialize, NoStaticVars]
  | sVar i =>
      simp [specialize, NoStaticVars]
  | dVar i =>
      simp [specialize, NoStaticVars]
  | add a b iha ihb =>
      simp [specialize]
      exact noStatic_foldAdd iha ihb

-- Semantic preservation.
--
-- Evaluating the specialized residual program under any static environment
-- gives the same result as evaluating the original program under the static
-- environment used at specialization time.
--
-- This is the formal version of:
--
--   eval(original, static_vals, dynamic_vals)
--     =
--   eval(residual, dynamic_vals)

theorem specialize_preserves_eval (senv senv' denv : Env) :
    forall e : Expr, eval senv' denv (specialize senv e) = eval senv denv e := by
  intro e
  induction e with
  | const n =>
      simp [specialize, eval]
  | sVar i =>
      simp [specialize, eval]
  | dVar i =>
      simp [specialize, eval]
  | add a b iha ihb =>
      simp [specialize, eval_foldAdd, eval, iha, ihb]

-- Static expressions specialize to a single constant equal to their compile-time value.
--
-- This theorem connects BTA soundness with partial evaluation:
-- if the analysis says an expression is static, the specializer fully evaluates it.

theorem static_expression_specializes_to_const (senv denv : Env) :
    forall e : Expr, bta e = .S -> specialize senv e = const (eval senv denv e) := by
  intro e
  induction e with
  | const n =>
      intro _
      simp [specialize, eval]
  | sVar i =>
      intro _
      simp [specialize, eval]
  | dVar i =>
      intro h
      cases h
  | add a b iha ihb =>
      intro h
      have hab : bta a = .S /\ bta b = .S :=
        add_node_staging_soundness a b h
      have ha := iha hab.left
      have hb := ihb hab.right
      simp [specialize, ha, hb, foldAdd, eval]



def emptyEnv : Env := fun _ => 0

theorem residual_semantic_preservation (senv denv : Env) (e : Expr) :
    eval emptyEnv denv (specialize senv e) = eval senv denv e := by
  exact specialize_preserves_eval senv emptyEnv denv e

end StageML
