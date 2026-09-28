# Embedded_Observer_NoGo

[![CI](https://github.com/SergejMaterov/Embedded_Observer_NoGo/actions/workflows/ci.yml/badge.svg)](https://github.com/SergejMaterov/Embedded_Observer_NoGo/actions/workflows/ci.yml)

Paper and numerical checks for **The Embedded-Observer No-Go Theorem: Non-Reconstructibility, Non-Autonomy, and Their
Quantitative Form for Tensor-Factor Observers in Finite-Dimensional Closed Quantum Systems** (Sergej Materov, revised
version, September 2026).

An observer is modelled as a proper tensor factor `O` of a finite-dimensional closed quantum system
`H = H_O ⊗ H_R` with unitary dynamics. The paper proves two independent obstructions, each with a quantitative form:

1. **Kinematic.** The reduced state `ρ_O` leaves open a manifold of global pure states of real dimension
   `2·r·d_R − r² − 1` (`r = rank ρ_O`) and fixes the state of the complement only up to its spectrum.
   The amount of correlation is accessible (`I(O:R) = 2·S(ρ_O)`); its content is not.
2. **Dynamical.** `O` obeys a fixed unitary law, independent of the state of `R`, **iff** the coupling is a product
   unitary. In robust form, with `E` the (average) autonomy defect and `D` the normalised Hilbert–Schmidt distance
   from the nearest product unitary,

   ```
   (1 + 1/d_O) · E  ≤  D²  ≤  2 (1 + 1/d_O) · E        (Theorem 4.2, both constants sharp)
   ```

   plus a worst-case bound (Proposition 4.7), a short-time autonomy bound for weak Hamiltonian coupling
   (Corollary 4.4), a bound on the growth of the reduced entropy per step (Proposition 5.3), and an exact causal cone
   with exact autonomy of the part of `O` beyond the cone of `R` for local circuits (Section 7).

## Repository layout

```
paper/                                 the paper (.docx is the reference version; .pdf is a LibreOffice rendering)
embedded_observer_checks.py            all numerical checks C1–C8 of Appendix A; prints the tables of the paper
tests/test_paper_claims.py             pytest suite: one test per proved statement, fresh seeds
results/checks_output.txt              output of the script that is reported in Appendix A
requirements.txt, requirements-dev.txt
.github/workflows/ci.yml               CI: tests + full checks on Python 3.10–3.12
```

## Quick start

```bash
python -m pip install -r requirements-dev.txt
python embedded_observer_checks.py      # ≈ 30 s; asserts on failure, prints the tables of Appendix A
python -m pytest                        # ≈ 5 s
```

`make install`, `make test`, `make checks` do the same. The script uses seeded random numbers; last-digit differences at
the 1e-16 level across numpy/BLAS versions are expected, the reported inequalities are not affected.

## Which statement is checked where

| Paper | Statement | Check |
|---|---|---|
| Prop. 3.1(b) | fibre dimension `2 r d_R − r² − 1` | `check_c7`, `test_proposition_3_1_fibre_dimension` |
| Cor. 3.3 | `I(O:R) = 2 S(ρ_O)` | `test_corollary_3_3_…` |
| Thm. 4.1, Rem. 4.5 | exact case; CNOT (fixed preparation) and SWAP examples | `check_c3`, `test_theorem_4_1_…`, `test_swap_…`, `test_cnot_…` |
| Thm. 4.2 | closed form of `E_W` and the two-sided bound | `check_c1`, `check_c2`, `test_theorem_4_2_…` |
| Cor. 4.4 | short-time autonomy bound | `check_c4`, `test_corollary_4_4_weak_coupling` |
| Prop. 4.7 | worst-case bound | `check_c8`, `test_proposition_4_7_worst_case` |
| Prop. 5.3 | entropy bound | `check_c5`, `test_proposition_5_3_entropy_bound` |
| Thm. 7.2, Prop. 7.3 | exact causal cone, exact core autonomy | `check_c6`, `test_theorem_7_2_…`, `test_proposition_7_3_…` |

## Status and caveats

- Numerical checks are sanity checks of statements that are **proved in the paper**; they are not proofs.
- The results are elementary. Table 1 of the paper says which parts are known (unitary equivalence of purifications,
  the light cone of a finite-depth circuit) and which are not found in this form in the literature (the quantitative
  equivalence of Theorem 4.2). Independent checking of Theorem 4.2 is welcome; please open an issue.
- Scope: finite-dimensional, closed, pure global state, observer = proper tensor factor. No claim is made about
  continuum quantum field theory, about any interpretation of quantum mechanics, or about experiments.
- `ε(U)` is an *average-input* quantity; worst-case statements are made explicitly (Proposition 4.7).

## Citation

See `CITATION.cff`. If you use the paper, please cite the version in `paper/`.

## License

Code: MIT (see `LICENSE`). The text of the paper in `paper/` is © 2026 Sergej Materov and is provided for reading and
verification.
