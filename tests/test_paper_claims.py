"""
Tests for the statements of "The Embedded-Observer No-Go Theorem" (revised, September 2026).

Each test names the result of the paper it checks. The tests are numerical sanity checks
of proved statements; they do not replace the proofs.
"""
import numpy as np
import pytest
from scipy.linalg import expm

import embedded_observer_checks as ec

SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # O = control


# ------------------------------------------------------------------ Theorem 4.2 / 4.1
@pytest.mark.parametrize("dO,dR", [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)])
def test_theorem_4_2_two_sided_bound(dO, dR):
    """(1+1/dO) E_W <= D_W^2 <= 2 (1+1/dO) E_W for random U and W."""
    c = 1 + 1 / dO
    for t in [0.0, 0.01, 0.1, 0.5, 1.0, 3.0]:
        for _ in range(25):
            U, W0 = ec.random_near_product(dO, dR, t)
            for W in (W0, ec.haar_unitary(dO)):
                E, V0 = ec.E_W_formula(U, W, dO, dR)
                E = max(E, 0.0)
                D2 = ec.D_W_exact(V0, dR) ** 2
                assert c * E <= D2 + 1e-9
                assert D2 <= 2 * c * E + 1e-9


def test_theorem_4_2_a_matches_monte_carlo():
    dO, dR = 2, 2
    U, W = ec.haar_unitary(4), ec.haar_unitary(2)
    formula, _ = ec.E_W_formula(U, W, dO, dR)
    mc = ec.E_W_montecarlo(U, W, dO, dR, n=20000)
    assert abs(formula - mc) < 6e-3


def test_theorem_4_1_product_unitary_is_exactly_autonomous():
    W, V = ec.haar_unitary(2), ec.haar_unitary(3)
    E, _ = ec.E_W_formula(np.kron(W, V), W, 2, 3)
    assert abs(E) < 1e-12


def test_theorem_4_2_upper_constant_is_attained_by_cnot():
    """E_1(CNOT) = 1/3 and D_1(CNOT)^2 = 1 = 2 (1 + 1/2) E."""
    E, V0 = ec.E_W_formula(CNOT, np.eye(2, dtype=complex), 2, 2)
    assert E == pytest.approx(1 / 3, abs=1e-12)
    assert ec.D_W_exact(V0, 2) ** 2 == pytest.approx(1.0, abs=1e-12)


# ------------------------------------------------------------------ Remark 4.5
def test_swap_is_nonproduct_nonentangling_and_nonautonomous():
    """E_W(SWAP) = 1 - 1/d for every W."""
    for _ in range(5):
        E, _ = ec.E_W_formula(SWAP, ec.haar_unitary(2), 2, 2)
        assert E == pytest.approx(0.5, abs=1e-12)


def test_cnot_fixed_preparation_counterexample():
    """rho_O = I/2 is unchanged by CNOT for every state of R, yet CNOT is non-autonomous on average."""
    for _ in range(50):
        phi = ec.haar_state(2)
        rho_out = sum(0.5 * ec.rho_O_of(CNOT @ np.kron(np.eye(2)[k], phi), 2, 2) for k in range(2))
        assert np.linalg.norm(rho_out - np.eye(2) / 2) < 1e-12
    assert ec.E_W_formula(CNOT, np.eye(2, dtype=complex), 2, 2)[0] > 0.3


# ------------------------------------------------------------------ Corollary 4.4 and Proposition 4.7
@pytest.mark.parametrize("dO,dR", [(2, 2), (2, 3), (3, 3)])
def test_corollary_4_4_weak_coupling(dO, dR):
    rng = ec.rng
    def herm(d):
        A = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
        return (A + A.conj().T) / 2
    HO, HR, Hi = herm(dO), herm(dR), herm(dO * dR)
    g = np.linalg.norm(Hi, 2)
    for t in [0.001, 0.01, 0.05, 0.2]:
        for scale in [0.01, 0.1, 1.0]:
            H = np.kron(HO, np.eye(dR)) + np.kron(np.eye(dO), HR) + scale * Hi
            U, W = expm(-1j * H * t), expm(-1j * HO * t)
            E, V0 = ec.E_W_formula(U, W, dO, dR)
            assert ec.D_W_exact(V0, dR) <= t * scale * g + 1e-9
            assert max(E, 0) <= (t * scale * g) ** 2 + 1e-9


@pytest.mark.parametrize("dO,dR", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_proposition_4_7_worst_case(dO, dR):
    """1 - <W psi|rho'|W psi> <= ||(U - W x V)|psi x phi>||^2 <= ||U - W x V||^2 (any V)."""
    rng = ec.rng
    for t in [0.0, 0.1, 1.0, 3.0]:
        for _ in range(40):
            U, W = ec.random_near_product(dO, dR, t)
            V = ec.haar_unitary(dR)
            P = np.kron(W, V)
            psi, phi = ec.haar_state(dO), ec.haar_state(dR)
            rho = ec.rho_O_of(U @ np.kron(psi, phi), dO, dR)
            delta = 1 - np.real((W @ psi).conj() @ rho @ (W @ psi))
            x2 = np.linalg.norm((U - P) @ np.kron(psi, phi)) ** 2
            assert delta <= x2 + 1e-10
            assert x2 <= np.linalg.norm(U - P, 2) ** 2 + 1e-10


# ------------------------------------------------------------------ Proposition 5.3
@pytest.mark.parametrize("dO,dR", [(2, 2), (3, 3), (2, 4), (4, 2)])
def test_proposition_5_3_entropy_bound(dO, dR):
    rng = ec.rng
    for _ in range(300):
        U, W = ec.random_near_product(dO, dR, rng.uniform(0, 2))
        psi, phi = ec.haar_state(dO), ec.haar_state(dR)
        rho = ec.rho_O_of(U @ np.kron(psi, phi), dO, dR)
        delta = 1 - np.real((W @ psi).conj() @ rho @ (W @ psi))
        x = min(max(delta, 0.0), 1 - 1 / dO)
        bound = ec.h_bin(x) + x * np.log(dO - 1)
        assert ec.vn_entropy(rho) <= bound + 1e-9


# ------------------------------------------------------------------ Proposition 3.1 and Corollary 3.3
@pytest.mark.parametrize("dO,dR,r", [(2, 2, 1), (2, 2, 2), (2, 3, 2), (3, 3, 3), (3, 4, 3), (4, 4, 4)])
def test_proposition_3_1_fibre_dimension(dO, dR, r):
    assert ec.fibre_dim(dO, dR, r) == 2 * r * dR - r * r - 1


def test_corollary_3_3_mutual_information_is_twice_entropy():
    dO, dR = 3, 4
    psi = ec.haar_state(dO * dR)
    m = psi.reshape(dO, dR)
    rho_O = m @ m.conj().T
    rho_R = m.T @ m.conj()
    assert ec.vn_entropy(rho_O) + ec.vn_entropy(rho_R) == pytest.approx(2 * ec.vn_entropy(rho_O), abs=1e-10)


# ------------------------------------------------------------------ Section 7
def _chain(n):
    return {e: ec.haar_unitary(4) for e in range(n - 1)}


def test_theorem_7_2_exact_light_cone():
    n = 8
    U = ec.brickwork(n, _chain(n))
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    A, B = ec.op_on(n, 0, X), ec.op_on(n, n - 1, X)   # distance 7, K = 2: exact zero for n <= 3
    An = A.copy()
    for step in range(1, 5):
        An = U.conj().T @ An @ U
        comm = np.linalg.norm(An @ B - B @ An, 2)
        if 2 * step < 7:
            assert comm < 1e-12
    assert comm > 1e-3   # after the cone has arrived the commutator is generically non-zero


def test_proposition_7_3_core_autonomy():
    n, nO = 8, 5          # O = sites 0..4, R = 5..7, core O_0 = {0}, L = 5, K = 2 -> exact for n <= 2
    gates = _chain(n)
    gates_alt = dict(gates)
    for e in range(4, n - 1):
        gates_alt[e] = ec.haar_unitary(4)          # different gates on the interface and inside R
    U_full, U_alt = ec.brickwork(n, gates), ec.brickwork(n, gates_alt)
    U_O = ec.brickwork(nO, {e: gates[e] for e in range(nO - 1)})
    psiO = ec.haar_state(2 ** nO)
    for steps in (1, 2):
        preds = []
        sO = psiO.copy()
        for _ in range(steps):
            sO = U_O @ sO
        pred = ec.reduced_site0(sO, nO)
        for phiR in (ec.haar_state(8), ec.haar_state(8)):
            for Uc in (U_full, U_alt):
                s = np.kron(psiO, phiR)
                for _ in range(steps):
                    s = Uc @ s
                assert np.linalg.norm(ec.reduced_site0(s, n) - pred) < 1e-12
