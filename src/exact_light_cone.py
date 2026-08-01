"""
Verification of Theorem 4.2 (§4.2-4.3): the Exact Light Cone for discrete
local circuits.

Claim: for a circuit built from strictly local, disjoint-support gates
applied in K sequential colour-class sublayers per step, the commutator
between a Heisenberg-evolved operator A(n) (initially supported at vertex 0)
and a static operator B (supported at vertex d, graph distance d away) is
EXACTLY zero for n < d/K, and only permitted to become nonzero from
n = ceil(d/K) onward -- not merely exponentially suppressed, but exactly
zero, because the underlying evolution is a finite product of strictly
local gates rather than the exponential of a local Hamiltonian.

We verify this on a path graph (chain) of N qubits with a GENERIC
(Haar-random, but fixed once) two-qubit gate on every edge -- deliberately
avoiding any fine-tuned gate (e.g. an exact ZZ-rotation at a special angle)
that could produce a misleading algebraic coincidence.

IMPORTANT METHODOLOGICAL NOTE: the physically meaningful test evolves ONLY
one of the two operators (A forward via the Heisenberg picture; B held
static). Testing the commutator of two operators BOTH conjugated by the
same U^n is unitarily invariant and trivially reproduces the n=0 value at
every n -- a check that looks like a light-cone test but is vacuous. See
README for details.
"""

import argparse
import numpy as np
from scipy.stats import unitary_group


def embed_two_qubit(gate4: np.ndarray, i: int, n: int) -> np.ndarray:
    """Embed a 4x4 gate acting on adjacent qubits (i, i+1) into n-qubit space."""
    left = np.eye(2 ** i, dtype=complex)
    right = np.eye(2 ** (n - i - 2), dtype=complex)
    return np.kron(np.kron(left, gate4), right)


def op_on_qubit(op: np.ndarray, q: int, n: int) -> np.ndarray:
    I2 = np.eye(2, dtype=complex)
    mats = [I2] * n
    mats[q] = op
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def build_step_unitary(n: int, seed: int):
    """Build one local-circuit step on an n-qubit path graph: K=2 colour
    classes (even edges, odd edges), each edge carrying a fixed, generic
    (Haar-random) two-qubit gate."""
    rng = np.random.default_rng(seed)
    edges_even = list(range(0, n - 1, 2))
    edges_odd = list(range(1, n - 1, 2))
    gate_for_edge = {i: unitary_group.rvs(4, random_state=rng) for i in (edges_even + edges_odd)}

    dim = 2 ** n

    def apply_sublayer(edge_starts):
        U = np.eye(dim, dtype=complex)
        for i in edge_starts:
            U = embed_two_qubit(gate_for_edge[i], i, n) @ U
        return U

    U_step = apply_sublayer(edges_odd) @ apply_sublayer(edges_even)
    K = 2  # two colour-class sublayers per step for a path graph (max degree 2)
    return U_step, K


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_qubits", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    N = args.n_qubits
    U_step, K = build_step_unitary(N, args.seed)

    unitary_ok = np.allclose(U_step.conj().T @ U_step, np.eye(2 ** N), atol=1e-9)
    print(f"Circuit: {N}-qubit path graph, K={K} colour-class sublayers/step, generic gates.")
    print(f"Unitarity check: {unitary_ok}\n")
    assert unitary_ok, "FAILED: step operator is not unitary"

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    A0 = op_on_qubit(X, 0, N)
    B_target = op_on_qubit(X, N - 1, N)  # STATIC -- not evolved

    d = N - 1
    threshold = d / K
    print(f"Graph distance d(0,{N-1}) = {d}. Predicted EXACT zero for n < {threshold} "
          f"(i.e. n <= {int(np.ceil(threshold)) - 1}), nonzero permitted from n = {int(np.ceil(threshold))}.\n")

    results = []
    for n in range(0, d + 1):
        Un = np.linalg.matrix_power(U_step, n)
        A_n = Un.conj().T @ A0 @ Un   # Heisenberg-evolve ONLY A
        comm_norm = np.linalg.norm(A_n @ B_target - B_target @ A_n, ord=2)
        results.append((n, comm_norm))
        flag = "~0 (machine precision)" if comm_norm < 1e-9 else ""
        print(f"n={n:2d}: ||[A(n), B_static]|| = {comm_norm:.6e}  {flag}")

    n_pred_zero_max = int(np.ceil(threshold)) - 1
    all_zero_before = all(v < 1e-9 for n, v in results if n <= n_pred_zero_max)
    first_nonzero_at_predicted = results[n_pred_zero_max + 1][1] > 1e-3

    print()
    assert all_zero_before, f"FAILED: expected exact zero for n<={n_pred_zero_max}"
    assert first_nonzero_at_predicted, f"FAILED: expected nonzero at n={n_pred_zero_max+1}"
    print(f"PASSED: commutator is exactly zero (machine precision) for n<={n_pred_zero_max}, "
          f"and becomes O(1) starting exactly at the predicted n={n_pred_zero_max+1}.")


if __name__ == "__main__":
    main()
