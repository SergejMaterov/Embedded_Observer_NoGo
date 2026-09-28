"""
Numerical check of Corollary 7.5 (two-source window) and Remark 7.6,
Embedded-Observer No-Go Theorem (Materov 2026a).

Verifies, on two structurally different discrete local circuits (a
path and a branching tree), that the reduced state of a core site O_0
is exactly independent of a "slow" source S while L_F <= nK < L_S
(Corollary 7.5(b)), and generically dependent on the "fast" source F
once nK >= L_F (Remark 7.6) -- matching the flash-before-shockwave
motivating example of the Introduction.

Requires numpy only. Uses seeded random numbers for reproducibility.
"""
import numpy as np


def random_unitary(dim, rng):
    z = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph


def apply_two_qubit_gate(psi, gate, i, j, N):
    """Apply a 4x4 unitary `gate` to qubits i,j of an N-qubit statevector
    tensor `psi` of shape (2,)*N. Works for any pair i != j, not just
    tensor-adjacent indices."""
    g = gate.reshape(2, 2, 2, 2)  # (out_i, out_j, in_i, in_j)
    psi_ = np.tensordot(g, psi, axes=([2, 3], [i, j]))
    # tensordot puts (out_i, out_j) first, then the remaining axes in
    # their original relative order.
    remaining_axes = [a for a in range(N) if a not in (i, j)]
    destination = [i, j] + remaining_axes
    return np.moveaxis(psi_, list(range(N)), destination)


def bfs_distances(N, edges, source):
    adj = {v: [] for v in range(N)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    dist = {source: 0}
    frontier = [source]
    while frontier:
        nxt = []
        for v in frontier:
            for w in adj[v]:
                if w not in dist:
                    dist[w] = dist[v] + 1
                    nxt.append(w)
        frontier = nxt
    return dist


def basis_tensor(bits, N):
    v = np.zeros(2 ** N, dtype=complex)
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    v[idx] = 1.0
    return v.reshape([2] * N)


def reduced_rho_core(psi_tensor, core, N):
    psi_moved = np.moveaxis(psi_tensor, core, 0).reshape(2, -1)
    return psi_moved @ psi_moved.conj().T


def trace_distance(rho1, rho2):
    ev = np.linalg.eigvalsh(rho1 - rho2)
    return 0.5 * np.sum(np.abs(ev))


class DiscreteLocalCircuit:
    """A discrete local circuit on graph (N, edges) with a proper edge
    coloring into K disjoint-edge classes (color_classes), applied in
    a fixed order each step: U = U_K ... U_1 (Section 7 conventions)."""

    def __init__(self, N, color_classes, class_order, rng):
        self.N = N
        self.color_classes = color_classes
        self.class_order = class_order
        self.K = len(class_order)
        self.gates = {
            (a, b): random_unitary(4, rng)
            for cls in color_classes
            for (a, b) in color_classes[cls]
        }

    def apply_layer(self, psi, cls):
        for (a, b) in self.color_classes[cls]:
            psi = apply_two_qubit_gate(psi, self.gates[(a, b)], a, b, self.N)
        return psi

    def apply_step(self, psi):
        for cls in self.class_order:
            psi = self.apply_layer(psi, cls)
        return psi

    def run(self, n_steps, flipped_site, core):
        bits = [0] * self.N
        if flipped_site is not None:
            bits[flipped_site] = 1
        psi = basis_tensor(bits, self.N)
        for _ in range(n_steps):
            psi = self.apply_step(psi)
        return reduced_rho_core(psi, core, self.N)


def check_window(circuit, core, F, S, edges, n_range, label):
    dist = bfs_distances(circuit.N, edges, core)
    L_F, L_S = dist[F], dist[S]
    print(f"\n[{label}] K={circuit.K}, L_F={L_F} (site {F}), "
          f"L_S={L_S} (site {S})")
    print(" n |  nK | dep_on_F   | dep_on_S   | predicted regime")
    for n in n_range:
        rho0 = circuit.run(n, None, core)
        dF = trace_distance(rho0, circuit.run(n, F, core))
        dS = trace_distance(rho0, circuit.run(n, S, core))
        nK = n * circuit.K
        if nK < L_F:
            regime = "independent of both"
        elif nK < L_S:
            regime = "dep. on F, indep. of S"
        else:
            regime = "no guarantee"
        print(f"{n:2d} | {nK:3d} | {dF:.3e} | {dS:.3e} | {regime}")


if __name__ == "__main__":
    rng = np.random.default_rng(7)

    # --- Example 1: 8-site path, core at one end, K=2 (even/odd bonds) ---
    N1 = 8
    edges1 = [(i, i + 1) for i in range(N1 - 1)]
    circuit1 = DiscreteLocalCircuit(
        N1,
        color_classes={
            "even": [(0, 1), (2, 3), (4, 5), (6, 7)],
            "odd": [(1, 2), (3, 4), (5, 6)],
        },
        class_order=["even", "odd"],
        rng=rng,
    )
    check_window(circuit1, core=0, F=3, S=7, edges=edges1,
                 n_range=range(5), label="path, N=8")

    # --- Example 2: 9-site branching tree, core at tip of a short arm,
    #     F on a short branch, S at the end of a longer branch, K=3 ---
    N2 = 9
    edges2 = [(0, 1), (1, 2), (2, 3), (2, 4), (4, 5), (5, 6), (6, 7), (7, 8)]
    circuit2 = DiscreteLocalCircuit(
        N2,
        color_classes={
            "A": [(1, 2), (4, 5), (6, 7)],
            "B": [(0, 1), (2, 3), (5, 6), (7, 8)],
            "C": [(2, 4)],
        },
        class_order=["A", "B", "C"],
        rng=rng,
    )
    check_window(circuit2, core=0, F=3, S=8, edges=edges2,
                 n_range=range(5), label="branching tree, N=9")
