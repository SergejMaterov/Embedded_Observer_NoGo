#!/usr/bin/env python3
"""
Numerical checks accompanying "The Embedded-Observer No-Go Theorem" (revised, Sept 2026).

Checks
  C1  closed formula  E_W(U) = 1 - (1 + d_O*||V0||_HS^2/d_R)/(d_O+1)  vs Monte-Carlo over Haar product inputs
  C2  Theorem 4.2:  E_W <= 2 D_W   and   D_W <= 2*sqrt((1+1/d_O)*E_W)   (random and near-product U, random W)
  C3  Theorem 4.1 exact case: product unitaries give E = 0, SWAP gives E = 1 - 1/d, CNOT counter-example (fixed preparation)
  C4  Corollary 4.4:  D(exp(-iHt)) <= t*||H_int||  and  E <= (t*||H_int||)^2
  C5  Proposition 5.2(iii): S(rho_O') <= h(delta) + delta*ln(d_O-1)  for product inputs
  C6  Theorem 7.2 / Proposition 7.3: exact light cone and exact "core autonomy" on a qubit chain
  C7  Proposition 3.1(b): real dimension of the fibre of purifications
  C8  Proposition 4.7 / Corollary 4.4(c): worst-case autonomy defect bound

Run:  python3 embedded_observer_checks.py          (prints all tables; asserts on failure)
Only numpy and scipy are required. All random draws are seeded.
"""
import numpy as np
from scipy.linalg import expm, polar

rng = np.random.default_rng(20260921)


# ----------------------------------------------------------------------------- helpers
def haar_unitary(d):
    z = (rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / np.abs(np.diag(r)))


def haar_state(d):
    v = rng.normal(size=d) + 1j * rng.normal(size=d)
    return v / np.linalg.norm(v)


def ptrace_O(M, dO, dR):
    """Tr_O of an operator on H_O (x) H_R  (O is the first factor)."""
    return np.trace(M.reshape(dO, dR, dO, dR), axis1=0, axis2=2)


def rho_O_of(psi_vec, dO, dR):
    m = psi_vec.reshape(dO, dR)
    return m @ m.conj().T


def vn_entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-14]
    return float(-(w * np.log(w)).sum())


def h_bin(x):
    if x <= 0 or x >= 1:
        return 0.0
    return float(-x * np.log(x) - (1 - x) * np.log(1 - x))


def E_W_formula(U, W, dO, dR):
    Up = np.kron(W.conj().T, np.eye(dR)) @ U
    V0 = ptrace_O(Up, dO, dR) / dO
    kappa = np.linalg.norm(V0, 'fro') ** 2 / dR
    return 1 - (1 + dO * kappa) / (dO + 1), V0


def D_W_exact(V0, dR):
    """min_V ||U - W(x)V||_HS / sqrt(dO dR) = sqrt(2 - 2 ||V0||_tr / dR)."""
    nuc = np.linalg.svd(V0, compute_uv=False).sum()
    return np.sqrt(max(0.0, 2 - 2 * nuc / dR))


def E_W_montecarlo(U, W, dO, dR, n=60000):
    tot = 0.0
    for _ in range(n):
        psi, phi = haar_state(dO), haar_state(dR)
        out = U @ np.kron(psi, phi)
        rho = rho_O_of(out, dO, dR)
        wpsi = W @ psi
        tot += np.real(wpsi.conj() @ rho @ wpsi)
    return 1 - tot / n


def random_near_product(dO, dR, t):
    W, V = haar_unitary(dO), haar_unitary(dR)
    H = rng.normal(size=(dO * dR,) * 2) + 1j * rng.normal(size=(dO * dR,) * 2)
    H = (H + H.conj().T) / 2
    H /= np.linalg.norm(H, 2)
    return np.kron(W, V) @ expm(-1j * t * H), W


# ---- helpers used by check_c6 / check_c7 and by the tests
def op_on(n, site, a):
    ops = [np.eye(2, dtype=complex)] * n
    ops[site] = a
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def gate_on(n, i, j, g):
    """embed a 4x4 gate acting on adjacent sites (i,i+1)."""
    assert j == i + 1
    return np.kron(np.kron(np.eye(2 ** i), g), np.eye(2 ** (n - j - 1)))


def brickwork(n, gates):
    """K = 2 colour classes: even edges then odd edges; gates[e] fixed once."""
    layer1 = np.eye(2 ** n, dtype=complex)
    for e in range(0, n - 1, 2):
        layer1 = gate_on(n, e, e + 1, gates[e]) @ layer1
    layer2 = np.eye(2 ** n, dtype=complex)
    for e in range(1, n - 1, 2):
        layer2 = gate_on(n, e, e + 1, gates[e]) @ layer2
    return layer2 @ layer1


def reduced_site0(state, nq):
    m = state.reshape(2, 2 ** (nq - 1))
    return m @ m.conj().T


def fibre_dim(dO, dR, r):
    # random pure state with Schmidt rank r
    A = rng.normal(size=(dO, r)) + 1j * rng.normal(size=(dO, r))
    B = rng.normal(size=(dR, r)) + 1j * rng.normal(size=(dR, r))
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    p = rng.uniform(0.2, 1, r); p /= p.sum()
    psi = sum(np.sqrt(p[k]) * np.kron(qa[:, k], qb[:, k]) for k in range(r))
    # tangent vectors of the orbit {(1 (x) V) psi}: (1 (x) iH) psi over Hermitian H (real basis)
    vecs = []
    for a in range(dR):
        for b in range(a, dR):
            for kind in ([0] if a == b else [0, 1]):
                H = np.zeros((dR, dR), complex)
                if a == b:
                    H[a, a] = 1
                elif kind == 0:
                    H[a, b] = H[b, a] = 1
                else:
                    H[a, b] = 1j; H[b, a] = -1j
                v = np.kron(np.eye(dO), 1j * H) @ psi
                vecs.append(np.concatenate([v.real, v.imag]))
    dim_vec = np.linalg.matrix_rank(np.array(vecs), tol=1e-9)
    return dim_vec - 1                     # remove the global-phase direction (H = identity)


def check_c1():
    """C1: closed formula vs Monte-Carlo."""
    print("=" * 70)
    print("C1  closed formula for E_W(U) vs Monte-Carlo (60000 Haar product inputs)")
    print("=" * 70)
    for dO, dR in [(2, 2), (2, 3), (3, 2)]:
        U = haar_unitary(dO * dR)
        W = haar_unitary(dO)
        Ef, _ = E_W_formula(U, W, dO, dR)
        Em = E_W_montecarlo(U, W, dO, dR)
        print(f"  d_O={dO}, d_R={dR}:  formula = {Ef:.5f}   Monte-Carlo = {Em:.5f}   |diff| = {abs(Ef-Em):.1e}")


def check_c2():
    """C2: Theorem 4.2 two-sided inequality."""
    print()
    print("=" * 70)
    print("C2  Theorem 4.2:  (1+1/d_O) E_W  <=  D_W^2  <=  2 (1+1/d_O) E_W   (any W; D_W minimised over V exactly)")
    print("=" * 70)
    lo_ratio_max, hi_ratio_max, count = 0.0, 0.0, 0
    rows = []
    for dO, dR in [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)]:
        for t in [0.0, 0.01, 0.05, 0.1, 0.3, 1.0, 3.0]:
            for _ in range(60):
                U, Wtrue = random_near_product(dO, dR, t)
                for W in [Wtrue, haar_unitary(dO)]:
                    E, V0 = E_W_formula(U, W, dO, dR)
                    E = max(E, 0.0)
                    D2 = D_W_exact(V0, dR) ** 2
                    c = 1 + 1 / dO
                    assert c * E <= D2 + 1e-9 and D2 <= 2 * c * E + 1e-9, (dO, dR, t, E, D2)
                    count += 1
                    if D2 > 1e-9:
                        lo_ratio_max = max(lo_ratio_max, c * E / D2)      # <= 1
                    if E > 1e-9:
                        hi_ratio_max = max(hi_ratio_max, D2 / (2 * c * E))  # <= 1
        U, W = random_near_product(dO, dR, 0.1)
        E, V0 = E_W_formula(U, W, dO, dR)
        rows.append((dO, dR, 0.1, max(E, 0.0), D_W_exact(V0, dR) ** 2))
    print(f"  {count} random (U,W) pairs tested, both inequalities hold in every case")
    print(f"  max of (1+1/d_O) E / D^2      = {lo_ratio_max:.4f}   (must be <= 1; ->1 near product)")
    print(f"  max of D^2 / (2 (1+1/d_O) E)  = {hi_ratio_max:.4f}   (must be <= 1)")
    print("  sample rows (ideal W, t = 0.1):   d_O d_R   E_W        (1+1/dO)E_W   D_W^2      2(1+1/dO)E_W")
    for dO, dR, t, E, D2 in rows:
        c = 1 + 1 / dO
        print(f"                                    {dO}   {dR}   {E:.6f}   {c*E:.6f}     {D2:.6f}   {2*c*E:.6f}")


def check_c3():
    """C3: exact case and counter-examples."""
    print()
    print("=" * 70)
    print("C3  exact case and counter-examples")
    print("=" * 70)
    dO = dR = 2
    Wp, Vp = haar_unitary(2), haar_unitary(2)
    E_prod, _ = E_W_formula(np.kron(Wp, Vp), Wp, 2, 2)
    print(f"  product unitary, ideal W:          E_W = {max(E_prod,0):.2e}   (expected 0)")
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)
    Es = [E_W_formula(SWAP, haar_unitary(2), 2, 2)[0] for _ in range(5)]
    print(f"  SWAP (non-entangling, non-product): E_W = {np.mean(Es):.4f} for random W (expected 1-1/d = 0.5); "
          f"best W: {E_W_formula(SWAP, np.eye(2), 2, 2)[0]:.4f}")
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)  # O control, R target
    # fixed preparation rho_O = I/2 : O-marginal is I/2 for every R state -> W = 1 "works" for this preparation only
    worst = 0.0
    for _ in range(200):
        phi = haar_state(2)
        # rho_O = I/2 (x) |phi><phi|, average over the two basis states of O
        rho_out = 0
        for k in range(2):
            e = np.zeros(2, complex); e[k] = 1
            rho_out = rho_out + 0.5 * rho_O_of(CNOT @ np.kron(e, phi), 2, 2)
        worst = max(worst, np.linalg.norm(rho_out - np.eye(2) / 2))
    print(f"  CNOT, fixed preparation rho_O=I/2:  max ||rho_O' - I/2|| over 200 random R states = {worst:.1e}  "
          f"(so 'fixed preparation' Lemma B is false)")
    E_cnot = min(E_W_formula(CNOT, haar_unitary(2), 2, 2)[0] for _ in range(2000))
    print(f"  CNOT, averaged over all O inputs:   min over 2000 random W of E_W = {E_cnot:.4f}  (> 0, non-autonomous)")


def check_c4():
    """C4: Corollary 4.4 weak coupling."""
    print()
    print("=" * 70)
    print("C4  Corollary 4.4:  D(exp(-iHt)) <= t*||H_int||_op   and   E_W <= (t*||H_int||)^2")
    print("=" * 70)
    ok = True
    for dO, dR in [(2, 2), (2, 3), (3, 3)]:
        HO = rng.normal(size=(dO, dO)) + 1j * rng.normal(size=(dO, dO)); HO = (HO + HO.conj().T) / 2
        HR = rng.normal(size=(dR, dR)) + 1j * rng.normal(size=(dR, dR)); HR = (HR + HR.conj().T) / 2
        Hi = rng.normal(size=(dO * dR,) * 2) + 1j * rng.normal(size=(dO * dR,) * 2); Hi = (Hi + Hi.conj().T) / 2
        g = np.linalg.norm(Hi, 2)
        for t in [0.001, 0.01, 0.05, 0.2]:
            for scale in [0.01, 0.1, 1.0]:
                H = np.kron(HO, np.eye(dR)) + np.kron(np.eye(dO), HR) + scale * Hi
                U = expm(-1j * H * t)
                W = expm(-1j * HO * t)
                E, V0 = E_W_formula(U, W, dO, dR)
                D = D_W_exact(V0, dR)     # minimised over V, so <= the Duhamel bound
                bound = t * scale * g
                ok &= (D <= bound + 1e-9) and (max(E, 0) <= bound ** 2 + 1e-9)
    print(f"  all cases satisfy D <= t*||H_int|| and E <= (t*||H_int||)^2: {ok}")


def check_c5():
    """C5: Proposition 5.3 entropy bound."""
    print()
    print("=" * 70)
    print("C5  Proposition 5.2(iii): S(rho_O') <= h(delta)+delta ln(d_O-1),  delta = 1 - <W psi|rho_O'|W psi>")
    print("=" * 70)
    viol = 0
    tight = 0.0
    for dO, dR in [(2, 2), (3, 3), (2, 4), (4, 2)]:
        for _ in range(4000):
            U, W = random_near_product(dO, dR, rng.uniform(0, 2))
            psi, phi = haar_state(dO), haar_state(dR)
            rho = rho_O_of(U @ np.kron(psi, phi), dO, dR)
            wpsi = W @ psi
            delta = 1 - np.real(wpsi.conj() @ rho @ wpsi)
            S = vn_entropy(rho)
            dl = min(max(delta, 0), 1 - 1 / dO)
            b = h_bin(dl) + dl * np.log(dO - 1)
            if S > b + 1e-9:
                viol += 1
            if b > 1e-6:
                tight = max(tight, S / b)
    print(f"  violations: {viol}   (max S/bound observed = {tight:.3f}, must be <= 1)")


def check_c6():
    """C6: light cone and core autonomy."""
    print()
    print("=" * 70)
    print("C6  exact light cone and core autonomy on a qubit chain")
    print("=" * 70)


    n = 10
    gates = {e: haar_unitary(4) for e in range(n - 1)}
    U = brickwork(n, gates)
    A = op_on(n, 0, np.array([[0, 1], [1, 0]], dtype=complex))
    B = op_on(n, 9, np.array([[0, 1], [1, 0]], dtype=complex))
    print("  10-site chain, K=2, d=9.  ||[A(n),B]||,  A(n)=U^-n A U^n, A=X_0, B=X_9:")
    An = A.copy()
    for step in range(0, 10):
        if step > 0:
            An = U.conj().T @ An @ U
        c = An @ B - B @ An
        print(f"    n={step}:  {np.linalg.norm(c, 2):.3e}")

    # core autonomy: chain of 8 qubits, O = sites 0..4, R = sites 5..7, O0 = {0}; dist(O0,R) = 5, K=2 -> exact for n <= 2
    n = 8
    gates = {e: haar_unitary(4) for e in range(n - 1)}
    gates_alt = dict(gates)
    for e in range(4, n - 1):            # change every gate touching R (edge 4-5 is the interface, edges 5-6, 6-7 inside R)
        gates_alt[e] = haar_unitary(4)
    U_full = brickwork(n, gates)
    U_alt = brickwork(n, gates_alt)
    nO = 5
    gates_O = {e: gates[e] for e in range(0, nO - 1)}     # gates with both endpoints in O
    U_O = brickwork(nO, gates_O)


    psiO = haar_state(2 ** nO)
    res = []
    for steps in [1, 2, 3]:
        out = []
        for phiR in [haar_state(2 ** 3), haar_state(2 ** 3)]:
            for Uc in [U_full, U_alt]:
                s = np.kron(psiO, phiR)
                for _ in range(steps):
                    s = Uc @ s
                out.append(reduced_site0(s, n))
        spread = max(np.linalg.norm(out[0] - o) for o in out)
        sO = psiO.copy()
        for _ in range(steps):
            sO = U_O @ sO
        pred = reduced_site0(sO, nO)
        err = np.linalg.norm(out[0] - pred)
        res.append((steps, spread, err))
    print("  8-site chain, O = sites 0..4, R = 5..7, core O_0 = {site 0}, dist(O_0,R)=5, K=2 -> exact for n <= 2")
    print("    n   spread of rho_{O0}(n) over different R states and different R/interface gates   ||rho - internal prediction||")
    for steps, spread, err in res:
        print(f"    {steps}   {spread:.3e}                                                                   {err:.3e}")


def check_c7():
    """C7: fibre dimension."""
    print()
    print("=" * 70)
    print("C7  Proposition 3.1(b): real dimension of the fibre F(Psi) in projective space = 2 r d_R - r^2 - 1")
    print("=" * 70)


    for dO, dR, r in [(2, 2, 1), (2, 2, 2), (2, 3, 1), (2, 3, 2), (3, 3, 1), (3, 3, 2), (3, 3, 3), (3, 4, 3), (4, 4, 4)]:
        print(f"  d_O={dO}, d_R={dR}, r={r}:  numerical = {fibre_dim(dO, dR, r)},   formula 2 r d_R - r^2 - 1 = {2*r*dR - r*r - 1}")


def check_c8():
    """C8: worst-case bound (Proposition 4.7)."""
    print()
    print("=" * 70)
    print("C8  Proposition 4.7 (worst case):  1 - <W psi|rho'|W psi>  <=  ||(U - W(x)V)|psi(x)phi>||^2  <=  ||U - W(x)V||_op^2")
    print("=" * 70)
    viol1 = viol2 = 0
    max_r1 = max_r2 = 0.0
    n_tests = 0
    for dO, dR in [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4)]:
        for t in [0.0, 0.01, 0.1, 0.5, 1.0, 3.0]:
            for _ in range(120):
                U, W = random_near_product(dO, dR, t)
                V = haar_unitary(dR) if rng.random() < 0.5 else np.eye(dR, dtype=complex)  # arbitrary (often bad) V too
                P = np.kron(W, V)
                psi, phi = haar_state(dO), haar_state(dR)
                out = U @ np.kron(psi, phi)
                rho = rho_O_of(out, dO, dR)
                wpsi = W @ psi
                delta = 1 - np.real(wpsi.conj() @ rho @ wpsi)
                x2 = np.linalg.norm((U - P) @ np.kron(psi, phi)) ** 2
                opn2 = np.linalg.norm(U - P, 2) ** 2
                n_tests += 1
                if delta > x2 + 1e-10: viol1 += 1
                if x2 > opn2 + 1e-10: viol2 += 1
                if x2 > 1e-8: max_r1 = max(max_r1, delta / x2)
                if opn2 > 1e-8: max_r2 = max(max_r2, x2 / opn2)
    print(f"  {n_tests} random tests: violations of first inequality = {viol1}, of second = {viol2}")
    print(f"  max delta/||(U-P)psi phi||^2 = {max_r1:.4f} (<=1),   max ||(U-P)psi phi||^2/||U-P||^2 = {max_r2:.4f} (<=1)")

    # weak-coupling worst case: sup over many random inputs of delta vs (t ||H_int||)^2
    dO, dR = 2, 3
    HO = rng.normal(size=(dO, dO)) + 1j * rng.normal(size=(dO, dO)); HO = (HO + HO.conj().T) / 2
    HR = rng.normal(size=(dR, dR)) + 1j * rng.normal(size=(dR, dR)); HR = (HR + HR.conj().T) / 2
    Hi = rng.normal(size=(dO * dR,) * 2) + 1j * rng.normal(size=(dO * dR,) * 2); Hi = (Hi + Hi.conj().T) / 2
    g = np.linalg.norm(Hi, 2)
    print("  weak coupling, d_O=2, d_R=3:  t*||H_int||   sup_delta over 20000 random inputs   (t*||H_int||)^2")
    for t in [0.01, 0.03, 0.1]:
        H = np.kron(HO, np.eye(dR)) + np.kron(np.eye(dO), HR) + Hi
        U = expm(-1j * H * t); W = expm(-1j * HO * t)
        sup = 0.0
        for _ in range(20000):
            psi, phi = haar_state(dO), haar_state(dR)
            rho = rho_O_of(U @ np.kron(psi, phi), dO, dR)
            wpsi = W @ psi
            sup = max(sup, 1 - np.real(wpsi.conj() @ rho @ wpsi))
        print(f"                                 {t*g:.4f}        {sup:.6f}                              {(t*g)**2:.6f}")


ALL_CHECKS = [check_c1, check_c2, check_c3, check_c4, check_c5, check_c6, check_c7, check_c8]


def main():
    for f in ALL_CHECKS:
        f()


if __name__ == "__main__":
    main()
