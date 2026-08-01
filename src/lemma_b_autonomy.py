"""
Verification of Lemma B (§3.2 of the paper): no autonomous unitary law once
the O-R coupling is active (entangling).

Claim: if the O-R coupling over [t, t+dt] is entangling, there is in
general no fixed unitary W on H_O alone such that
    rho_O(t+dt) = W rho_O(t) W^dagger
holds for every possible state of R.

This script fixes O's initial state identically across two runs, varies
only R's initial state, and evolves both through the SAME generic
entangling gate. If O's dynamics were truly autonomous, both resulting
rho_O should be pure and identical (since a fixed unitary applied to a
fixed pure input always gives a fixed pure output). We show this fails.
"""

import argparse
import numpy as np
from scipy.stats import unitary_group


def evolve_rho_O(G: np.ndarray, psi0_O: np.ndarray, psi0_R: np.ndarray) -> np.ndarray:
    psi = np.kron(psi0_O, psi0_R)
    psi_n = G @ psi
    m = psi_n.reshape(2, 2)
    return m @ m.conj().T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()

    G = unitary_group.rvs(4, random_state=args.seed)  # generic entangling gate

    psi0_O = np.array([1, 0], dtype=complex)  # O always starts in |0>
    psi0_R_a = np.array([1, 0], dtype=complex)
    psi0_R_b = np.array([0, 1], dtype=complex)

    rhoO_a = evolve_rho_O(G, psi0_O, psi0_R_a)
    rhoO_b = evolve_rho_O(G, psi0_O, psi0_R_b)

    purity_a = np.trace(rhoO_a @ rhoO_a).real
    purity_b = np.trace(rhoO_b @ rhoO_b).real
    identical = np.allclose(rhoO_a, rhoO_b, atol=1e-10)

    print("Same O-initial-state (|0>), two different R-initial-states:")
    print(f"rho_O (R started |0>):\n{np.round(rhoO_a, 4)}")
    print(f"rho_O (R started |1>):\n{np.round(rhoO_b, 4)}\n")
    print(f"Purity (R=|0>): {purity_a:.6f}   Purity (R=|1>): {purity_b:.6f}")
    print(f"Identical outputs? {identical}")
    print()
    print("If O's dynamics were autonomous/unitary, a single fixed W acting on")
    print("the SAME initial rho_O=|0><0| must give the SAME (and pure) output")
    print("regardless of R -- it does not.")

    assert purity_a < 0.999 and purity_b < 0.999, "FAILED: outputs should be mixed"
    assert not identical, "FAILED: outputs should differ depending on R's state"
    print("\nPASSED: no fixed O-only unitary law reproduces both outcomes.")


if __name__ == "__main__":
    main()
