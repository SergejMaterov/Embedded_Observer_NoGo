"""
Verification of §3.4's counter-example: (a) [reduction failure, "stock"] and
(b) [autonomy failure, "flow"] are logically distinct, not one fact restated.

Protocol: an entangling step first mixes rho_O (permanently breaking (a) --
the global state can no longer be reconstructed from rho_O alone, Lemma A).
A SECOND, exactly product step is then applied. Because that specific
transition is non-entangling, Lemma B's condition for autonomous unitary
evolution is met FOR THAT STEP ALONE, despite the input already being mixed:
    rho_O(2) = u_O rho_O(1) u_O^dagger    exactly.
"""

import argparse
import numpy as np
from scipy.stats import unitary_group


def rho_O_from_psi(psi: np.ndarray) -> np.ndarray:
    m = psi.reshape(2, 2)
    return m @ m.conj().T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed_entangling", type=int, default=3)
    ap.add_argument("--seed_uO", type=int, default=5)
    ap.add_argument("--seed_uR", type=int, default=6)
    args = ap.parse_args()

    G_entangling = unitary_group.rvs(4, random_state=args.seed_entangling)
    u_O = unitary_group.rvs(2, random_state=args.seed_uO)
    u_R = unitary_group.rvs(2, random_state=args.seed_uR)
    G_product = np.kron(u_O, u_R)  # EXACTLY product coupling for step 2

    psi0 = np.kron([1, 0], [1, 0]).astype(complex)

    # Step 1: entangling -> mixes rho_O, permanently breaking (a)
    psi1 = G_entangling @ psi0
    rho_O_1 = rho_O_from_psi(psi1)
    purity1 = np.trace(rho_O_1 @ rho_O_1).real

    # Step 2: exactly product -> test whether (b) holds for this transition alone
    psi2 = G_product @ psi1
    rho_O_2_true = rho_O_from_psi(psi2)
    rho_O_2_predicted = u_O @ rho_O_1 @ u_O.conj().T
    purity2 = np.trace(rho_O_2_true @ rho_O_2_true).real

    match = np.allclose(rho_O_2_true, rho_O_2_predicted, atol=1e-10)
    purity_preserved = abs(purity1 - purity2) < 1e-9

    print(f"Step 1 (entangling): purity of rho_O(1) = {purity1:.6f}")
    print("  -> S(rho_O) > 0: (a) [reconstruction] has PERMANENTLY failed (Lemma A).\n")
    print(f"Step 2 (exactly product u_O (x) u_R):")
    print(f"  True rho_O(2)      =\n{np.round(rho_O_2_true, 4)}")
    print(f"  u_O rho_O(1) u_O^dagger =\n{np.round(rho_O_2_predicted, 4)}")
    print(f"  Exact match? {match}")
    print(f"  Purity of rho_O(2) = {purity2:.6f}  (unchanged from step 1? {purity_preserved})")

    assert purity1 < 0.999, "FAILED: step 1 should produce a mixed state"
    assert match, "FAILED: step 2 should reproduce an exact unitary update"
    assert purity_preserved, "FAILED: purity should be exactly preserved by a unitary step"

    print("\nPASSED: (b) [autonomy] holds EXACTLY for step 2 despite (a) having")
    print("already, and permanently, failed after step 1. Stock and flow are")
    print("logically separable, not one fact restated.")


if __name__ == "__main__":
    main()
