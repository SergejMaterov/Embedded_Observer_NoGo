# Computational Appendix — The Embedded-Observer No-Go Theorem

This repository reproduces every numerical claim in:

> S. Materov, *The Embedded-Observer No-Go Theorem: Causal Order, Geometrically Induced Reduction,
> and the Impossibility of Complete Self-Description in Any Closed Unitary Quantum System*, 2026.

The paper's central results are proved analytically (Lemmas A, B; Theorem 3.1; Theorem 4.2). This
repository does not "prove" anything by simulation — proofs are proofs — but it makes every
non-obvious claim concretely checkable, and it exists because two of the four checks below turned up
a genuine subtlety (§3.4's stock/flow separation) and a genuine methodological trap (§4's "vacuous
test", see below) that are easy to get wrong even when the underlying mathematics is correct.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── run_all.sh                      <- reproduces every number in the paper, in order
├── LICENSE
├── src/
│   ├── lemma_a_purification.py     <- §3.1: purification non-uniqueness (Hughston-Jozsa-Wootters)
│   ├── lemma_b_autonomy.py         <- §3.2: no autonomous unitary under active coupling
│   ├── stock_flow_separation.py    <- §3.4: stock (a) and flow (b) are logically distinct
│   └── exact_light_cone.py         <- §4.2-4.3: the Exact Light Cone theorem
└── results/
    └── log.txt                     <- full output of run_all.sh
```

## What each script proves, and what it doesn't

- **`lemma_a_purification.py`** — exact, deterministic linear algebra: constructs two distinct global
  qubit-pair states, related by a generic unitary on the R-factor alone, and confirms they reduce to
  bit-for-bit identical ρ_O. This is a direct instance of the Hughston-Jozsa-Wootters theorem, not a
  statistical claim — no sampling, no seed-dependence in the qualitative conclusion.

- **`lemma_b_autonomy.py`** — same O-preparation, two different R-preparations, evolved through the
  same generic entangling gate, giving two different (both mixed) ρ_O. Confirms no single fixed
  O-only unitary can reproduce both outcomes from one input.

- **`stock_flow_separation.py`** — the paper's central subtlety. An entangling step is followed by an
  **exactly product** step. Confirms ρ_O(2) = u_O ρ_O(1) u_O† to numerical precision (≈1e-10) even
  though ρ_O(1) is already mixed (purity ≈0.62) — i.e. condition (b) [autonomy] can hold for a single
  transition even after condition (a) [reconstructibility] has already, and permanently, failed.

- **`exact_light_cone.py`** — the paper's main quantitative result. Builds a discrete local circuit on
  a 10-qubit chain from **generic (Haar-random) gates** — deliberately not a fine-tuned gate like an
  exact-angle ZZ-rotation, which can produce a misleading algebraic coincidence where the commutator
  stays zero for reasons unrelated to locality (see "A trap we hit ourselves" below). Confirms the
  commutator is exactly zero (machine precision, ~1e-15) up to the predicted step, then jumps to O(1)
  starting exactly at the predicted step.

None of these scripts are Monte Carlo estimates of a statistical tendency (contrast with, e.g., a
repository estimating a percentile from repeated random draws) — each is a single, deterministic
numerical instance of an exact analytical claim, and the assertions in each script will hold for
(almost) any seed, not just the ones checked in here. Seeds are fixed only for reproducibility of the
exact printed numbers, not because the qualitative result depends on a lucky draw.

## A trap we hit ourselves, twice, while developing this

**Trap 1 — the vacuous commutator test.** An early version of the light-cone check computed
`||[U^-n A U^n, U^-n B U^n]||` — i.e. it conjugated *both* operators by the same U^n. This is
unitarily invariant: `||U^-n [A,B] U^n|| = ||[A,B]||` for *every* n, so the test trivially reproduces
the n=0 value forever and tells you nothing about locality. `exact_light_cone.py` evolves *only* A,
comparing it against a static B, which is the physically meaningful question ("does information
injected at vertex 0 reach vertex d after n steps"). If you are adapting this code, keep that
asymmetry — it is easy to accidentally symmetrize it back into the vacuous form.

**Trap 2 — a fine-tuned gate hides the real phenomenon.** An early version used a single fixed
ZZ-type rotation at exactly θ=π/4 on every edge. This gate happens to anticommute with X in a way
that made certain Pauli-string commutators vanish identically regardless of causal separation — a
genuine but misleading algebraic special case, not evidence about locality one way or the other.
Switching to a **generic (Haar-random) gate per edge** removed the coincidence and revealed the true
predicted behaviour (exact zero, then a clean jump to O(1)). If you test this yourself with a
"nice" analytic gate, check whether your chosen gate has a special symmetry before trusting a
null result.

## Quickstart

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

Runtime: a few seconds total. Each script can also be run directly, e.g.
`python3 src/exact_light_cone.py --n_qubits 12 --seed 3` to try other chain lengths or seeds.

## Citing

If this repository is cited independently of the paper, please cite the paper itself and reference
this repository as its computational supplement.

### License
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

**CC BY-NC 4.0** — Creative Commons Attribution-NonCommercial 4.0 International.

You are free to use, share, and adapt this code for **non-commercial purposes**
provided you give appropriate credit. Commercial use requires written permission
from the author.

© 2026 Sergej Materov <sergejmaterov2@gmail.com> ORCID: 0009-0001-3398-9906
