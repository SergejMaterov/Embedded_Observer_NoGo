"""
Verification of Lemma A (§3.1 of the paper): purification non-uniqueness
(Hughston-Jozsa-Wootters, 1993).

Claim: if rho_O is not pure, there exist infinitely many distinct global
pure states on H_O (x) H_R -- related by arbitrary unitaries acting on H_R
alone -- that all reduce to the exact same rho_O. An observer confined to
O therefore cannot determine the global state from rho_O alone.

This script constructs two such global states explicitly (using two
different orthonormal bases on H_R, related by a generic random unitary)
and confirms they produce bit-for-bit identical rho_O while remaining
distinct as global states.
"""

import argparse
import numpy as np
from scipy.stats import unitary_group


def reduced_rho_O(psi: np.ndarray) -> np.ndarray:
    """psi is a length-4 statevector for a 2x2 (O,R) qubit pair;
    returns the 2x2 reduced density matrix on O via partial trace over R."""
    m = psi.reshape(2, 2)
    return m @ m.conj().T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=float, default=0.7, help="eigenvalue of rho_O (mixedness)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    p = args.p
    rho_O_target = np.diag([p, 1 - p]).astype(complex)

    # Purification 1: the "Schmidt-basis-aligned" purification
    psi1 = np.sqrt(p) * np.kron([1, 0], [1, 0]) + np.sqrt(1 - p) * np.kron([0, 1], [0, 1])
    psi1 = psi1.astype(complex)

    # Purification 2: apply a generic unitary to the R-factor ONLY
    U_R = unitary_group.rvs(2, random_state=args.seed)
    psi1_reshaped = psi1.reshape(2, 2)
    psi2 = (psi1_reshaped @ U_R.T).reshape(4)

    rhoO_1 = reduced_rho_O(psi1)
    rhoO_2 = reduced_rho_O(psi2)

    same_rho = np.allclose(rhoO_1, rhoO_2, atol=1e-10)
    same_psi = np.allclose(psi1, psi2, atol=1e-10)

    print(f"Target rho_O (p={p}):\n{np.round(rho_O_target, 4)}\n")
    print(f"rho_O from purification 1:\n{np.round(rhoO_1, 4)}\n")
    print(f"rho_O from purification 2:\n{np.round(rhoO_2, 4)}\n")
    print(f"Same reduced state (rho_O_1 == rho_O_2)? {same_rho}")
    print(f"Same global state  (psi1   == psi2)?     {same_psi}")

    assert same_rho, "FAILED: reduced states should be identical"
    assert not same_psi, "FAILED: global states should be genuinely distinct"
    print("\nPASSED: two distinct global states share an identical reduced rho_O.")
    print("An observer confined to O cannot distinguish them from local data alone.")


if __name__ == "__main__":
    main()
