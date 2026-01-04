from __future__ import annotations

import numpy as np


class ChainComplex:
    """
    Minimal chain complex over F2:

        C2 --d2--> C1 --d1--> C0

    Internally:
      - d2: (dim_C1 x dim_C2) matrix over F2
      - d1: (dim_C0 x dim_C1) matrix over F2

    We enforce the chain condition d1 ∘ d2 = 0 over F2 at construction time.
    """

    def __init__(
        self,
        dim_C2: int,
        dim_C1: int,
        dim_C0: int,
        d2,
        d1,
        basis_C2=None,
        basis_C1=None,
        basis_C0=None,
        name: str = "",
    ):
        self.dim_C2 = dim_C2  # |C2| = #faces
        self.dim_C1 = dim_C1  # |C1| = #edges
        self.dim_C0 = dim_C0  # |C0| = #vertices

        self.d2 = np.array(d2, dtype=np.int8)  # shape (dim_C1, dim_C2)
        self.d1 = np.array(d1, dtype=np.int8)  # shape (dim_C0, dim_C1)

        assert self.d2.shape == (dim_C1, dim_C2)
        assert self.d1.shape == (dim_C0, dim_C1)

        self.basis_C2 = basis_C2 or list(range(dim_C2))
        self.basis_C1 = basis_C1 or list(range(dim_C1))
        self.basis_C0 = basis_C0 or list(range(dim_C0))

        self.name = name

        # Enforce d1 ∘ d2 = 0 (mod 2)
        prod = (self.d1 @ self.d2) % 2
        if np.any(prod):
            raise ValueError("Chain condition violated: d1 ∘ d2 != 0 over F2")

    def __repr__(self):
        return (
            f"ChainComplex(name={self.name!r}, "
            f"|C2|={self.dim_C2}, |C1|={self.dim_C1}, |C0|={self.dim_C0})"
        )


def surface_square_lattice(d: int) -> ChainComplex:
    """
    Build a d x d square lattice chain complex.

    - Vertices V[x,y], 0 <= x,y < d
    - Horizontal edges Eh[x,y] between V[x,y] and V[x,y+1], 0 <= x < d, 0 <= y < d-1
    - Vertical edges Ev[x,y] between V[x,y] and V[x+1,y], 0 <= x < d-1, 0 <= y < d
    - Faces F[x,y] = unit squares, 0 <= x,y < d-1

    Boundary:
      d2(F[x,y]) = Eh[x,y] + Eh[x+1,y] + Ev[x,y] + Ev[x,y+1]
      d1(Eh[x,y]) = V[x,y] + V[x,y+1]
      d1(Ev[x,y]) = V[x,y] + V[x+1,y]
    """
    if d < 2:
        raise ValueError("d must be >= 2")

    # ---- basis enumeration ----

    # Faces F[x,y]
    faces = [("F", x, y) for x in range(d - 1) for y in range(d - 1)]

    # Horizontal edges Eh[x,y]
    Eh = [("Eh", x, y) for x in range(d) for y in range(d - 1)]

    # Vertical edges Ev[x,y]
    Ev = [("Ev", x, y) for x in range(d - 1) for y in range(d)]

    edges = Eh + Ev

    # Vertices V[x,y]
    verts = [("V", x, y) for x in range(d) for y in range(d)]

    n2 = len(faces)
    n1 = len(edges)
    n0 = len(verts)

    # Index maps
    face_idx = {f: i for i, f in enumerate(faces)}
    edge_idx = {e: i for i, e in enumerate(edges)}
    vert_idx = {v: i for i, v in enumerate(verts)}

    # Allocate d2: C2 -> C1  (shape = (n1, n2))
    d2 = np.zeros((n1, n2), dtype=np.int8)

    # Allocate d1: C1 -> C0  (shape = (n0, n1))
    d1 = np.zeros((n0, n1), dtype=np.int8)

    # ---- define d2 on faces ----
    for x in range(d - 1):
        for y in range(d - 1):
            f = ("F", x, y)
            col = face_idx[f]

            # Boundary of the plaquette:
            e_bottom = ("Eh", x, y)       # between (x,y) and (x,y+1)
            e_top = ("Eh", x + 1, y)      # between (x+1,y) and (x+1,y+1)
            e_left = ("Ev", x, y)         # between (x,y) and (x+1,y)
            e_right = ("Ev", x, y + 1)    # between (x,y+1) and (x+1,y+1)

            for e in (e_bottom, e_top, e_left, e_right):
                row = edge_idx[e]
                d2[row, col] ^= 1  # add edge to boundary (mod 2)

    # ---- define d1 on edges ----

    # Horizontal edges: Eh[x,y] from V[x,y] to V[x,y+1]
    for x in range(d):
        for y in range(d - 1):
            e = ("Eh", x, y)
            col = edge_idx[e]
            v1 = ("V", x, y)
            v2 = ("V", x, y + 1)
            for v in (v1, v2):
                row = vert_idx[v]
                d1[row, col] ^= 1

    # Vertical edges: Ev[x,y] from V[x,y] to V[x+1,y]
    for x in range(d - 1):
        for y in range(d):
            e = ("Ev", x, y)
            col = edge_idx[e]
            v1 = ("V", x, y)
            v2 = ("V", x + 1, y)
            for v in (v1, v2):
                row = vert_idx[v]
                d1[row, col] ^= 1

    return ChainComplex(
        dim_C2=n2,
        dim_C1=n1,
        dim_C0=n0,
        d2=d2,
        d1=d1,
        basis_C2=faces,
        basis_C1=edges,
        basis_C0=verts,
        name=f"surface_square_lattice(d={d})",
    )


def css_from_chain(chain: ChainComplex):
    """
    CSS extraction for qubits on C1:

        - Qubits: basis of C1 (edges).
        - X checks: faces, from d2.
        - Z checks: vertices, from d1.

    Concretely:
        Hx : (#faces x #edges) = d2^T
        Hz : (#verts x #edges) = d1

    Check CSS commutation: Hz @ Hx^T = 0 (mod 2).
    """
    Hx = (chain.d2.T) % 2  # shape: (|C2|, |C1|)
    Hz = (chain.d1) % 2    # shape: (|C0|, |C1|)

    comm = (Hz @ Hx.T) % 2  # shape: (|C0|, |C2|)
    return Hx, Hz, comm


def pretty_print_matrix(M, row_labels=None, col_labels=None, name="M"):
    print(f"\n{name} (shape={M.shape}):")
    if row_labels is not None:
        row_labels = [str(r) for r in row_labels]
    if col_labels is not None:
        col_labels = [str(c) for c in col_labels]

    # Simple printing; you can replace with nicer formatting if you like.
    print(M)


def main():
    d = 3  # smallest interesting surface-like patch
    chain = surface_square_lattice(d)

    print(chain)

    Hx, Hz, comm = css_from_chain(chain)

    print(f"Number of qubits (edges): {chain.dim_C1}")
    print(f"Number of X checks (faces): {Hx.shape[0]}")
    print(f"Number of Z checks (vertices): {Hz.shape[0]}")

    # Print matrices (for d=3 they are still small)
    pretty_print_matrix(Hx, name="Hx (X stabilizers from faces)")
    pretty_print_matrix(Hz, name="Hz (Z stabilizers from vertices)")

    # Verify commutation
    if np.any(comm):
        print("\n[ERROR] CSS commutation violated: Hz * Hx^T != 0 over F2")
    else:
        print("\n[OK] CSS commutation holds: Hz * Hx^T = 0 over F2")


if __name__ == "__main__":
    main()
