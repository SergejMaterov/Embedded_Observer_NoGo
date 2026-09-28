# Changelog

## 2.0.0 (September 2026)

**Paper** (`paper/Embedded_Observer_NoGo_revised.docx`): rewritten. Main changes relative to earlier versions:

- Lemma B (fixed-preparation formulation, one numerical example) replaced by Theorems 4.1–4.2 with complete proofs;
  new quantitative result: `(1+1/d_O) E_W <= D_W^2 <= 2(1+1/d_O) E_W`.
- Non-reconstructibility reformulated as Proposition 3.1 (valid for every global state, with the dimension of the fibre).
- Added Proposition 4.7 (worst-case bound) and Corollary 4.4 (short-time autonomy bound for weak coupling).
- Removed: monotonic-growth/irreversibility claims, sections with dangling cross-references, experimental proposals,
  interpretation-specific overclaims.
- New: exact core autonomy for local circuits (Proposition 7.3).

**Code**: the checks were refactored into `embedded_observer_checks.py` (functions `check_c1` … `check_c8`, all helpers at
module level) and a `pytest` suite (`tests/`). The printed tables of the script are unchanged and reproduce Appendix A.
