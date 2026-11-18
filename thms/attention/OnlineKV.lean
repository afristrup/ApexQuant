import Mathlib

open Finset
open scoped BigOperators

namespace ApexQuant.Attention

/-- A key reconstructed from one nested pass. -/
def passKey {E P : Type*} [AddCommMonoid E] [Module ℝ E] (coeff : P → ℝ) (key : P → E) (p : P) : E :=
  coeff p • key p

/-- The online key assembled from the available pass prefixes. -/
def onlineKey {E P : Type*} [AddCommMonoid E] [Module ℝ E] (passes : Finset P)
    (coeff : P → ℝ) (key : P → E) : E :=
  passes.sum (passKey coeff key)

/-- Inner-product attention score decomposition over online KV passes. -/
theorem online_inner_product_decomposition
    {E P : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    {passes : Finset P} {q : E} {coeff : P → ℝ} {key : P → E} :
    inner ℝ q (onlineKey passes coeff key) =
      passes.sum (fun p => coeff p * inner ℝ q (key p)) := by
  unfold onlineKey passKey
  rw [inner_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [real_inner_smul_right]

/-- Concrete rotated-query form. Each pass stores a code vector in a rotated
coordinate system and contributes the inverse-rotated key prefix. -/
theorem online_rotated_query_decomposition
    {E P : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    {passes : Finset P} {q : E} {coeff : P → ℝ}
    {rotation : P → (E ≃ₗᵢ[ℝ] E)} {code : P → E} :
    inner ℝ q
        (onlineKey passes coeff (fun p => (rotation p).symm (code p))) =
      passes.sum (fun p => coeff p * inner ℝ (rotation p q) (code p)) := by
  unfold onlineKey passKey
  rw [inner_sum]
  apply Finset.sum_congr rfl
  intro p _
  rw [real_inner_smul_right]
  have hinner := (rotation p).inner_map_map q ((rotation p).symm (code p))
  rw [hinner.symm]
  simp

/-- If the available pass prefixes are nested, adding another pass changes the
score by exactly its weighted rotated-query inner product. -/
theorem online_prefix_increment
    {E P : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    {passes : Finset P} {p : P} {q : E} {coeff : P → ℝ} {key : P → E}
    [DecidableEq P] (hp : p ∉ passes) :
    inner ℝ q (onlineKey (insert p passes) coeff key) -
        inner ℝ q (onlineKey passes coeff key) =
      coeff p * inner ℝ q (key p) := by
  unfold onlineKey passKey
  rw [Finset.sum_insert hp, inner_add_right]
  rw [real_inner_smul_right]
  ring


/-- Upgrading one rotated key prefix increments its attention logit without
recomputing earlier passes. -/
theorem online_rotated_prefix_increment
    {E P : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    {passes : Finset P} {p : P} {q : E} {coeff : P → ℝ}
    {rotation : P → (E ≃ₗᵢ[ℝ] E)} {code : P → E}
    [DecidableEq P] (hp : p ∉ passes) :
    inner ℝ q (onlineKey (insert p passes) coeff
        (fun p => (rotation p).symm (code p))) -
        inner ℝ q (onlineKey passes coeff
          (fun p => (rotation p).symm (code p))) =
      coeff p * inner ℝ (rotation p q) (code p) := by
  rw [online_prefix_increment hp]
  have hinner := (rotation p).inner_map_map q ((rotation p).symm (code p))
  rw [hinner.symm]
  simp

/-- The attention-logit error is bounded by query norm times key residual norm. -/
theorem attention_score_error_le
    {E P : Type*} [SeminormedAddCommGroup E] [InnerProductSpace ℝ E]
    {passes : Finset P} {q k : E} {coeff : P → ℝ} {key : P → E} :
    |inner ℝ q k - inner ℝ q (onlineKey passes coeff key)| ≤
      ‖q‖ * ‖k - onlineKey passes coeff key‖ := by
  rw [← inner_sub_right]
  exact abs_real_inner_le_norm q (k - onlineKey passes coeff key)

end ApexQuant.Attention
