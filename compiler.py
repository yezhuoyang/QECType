from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Iterable


# ============================================================
#  Core chain complex (backend)
# ============================================================

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


# ============================================================
#  AST: expressions
# ============================================================

class Expr:
    pass


@dataclass
class Var(Expr):
    name: str


@dataclass
class IntLit(Expr):
    value: int


@dataclass
class BinOp(Expr):
    op: str  # '+', '-', '*' (for now)
    left: Expr
    right: Expr


def eval_expr(expr: Expr, env: Dict[str, int]) -> int:
    if isinstance(expr, Var):
        return env[expr.name]
    elif isinstance(expr, IntLit):
        return expr.value
    elif isinstance(expr, BinOp):
        l = eval_expr(expr.left, env)
        r = eval_expr(expr.right, env)
        if expr.op == "+":
            return l + r
        elif expr.op == "-":
            return l - r
        elif expr.op == "*":
            return l * r
        else:
            raise ValueError(f"Unknown binary op {expr.op!r}")
    else:
        raise TypeError(f"Unknown Expr type: {type(expr)}")


# ============================================================
#  AST: cells, ranges, boundary rules
# ============================================================

@dataclass
class FinRange:
    """
    Represents 'x: Fin(upper_expr)' meaning 0 <= x < upper_expr(env).
    """
    var_name: str
    upper_expr: Expr


@dataclass
class CellFamilyDecl:
    """
    A cell family at a fixed chain dimension.

    dim = 2: faces (C2)
    dim = 1: edges (C1)
    dim = 0: vertices (C0)
    """
    dim: int
    name: str
    index_ranges: List[FinRange]


@dataclass
class CellPattern:
    """
    Pattern used in boundary rules, e.g. F[x,y].
    """
    name: str
    index_vars: List[str]


@dataclass
class CellRefExpr:
    """
    Reference to a concrete cell with index expressions, e.g. Ex[x, y+1].
    """
    name: str
    index_exprs: List[Expr]


@dataclass
class BoundaryRule:
    """
    Boundary rule for an operator (d2 or d1):

        d2(F[x,y]) = Ex[x,y] + Ex[x,y+1] + Ey[x,y] + Ey[x+1,y];

    Encoded as:
      op_name    = "d2"
      src_pattern= CellPattern("F", ["x", "y"])
      terms      = [CellRefExpr("Ex", [...]), ...]
    """
    op_name: str  # "d2" or "d1"
    src_pattern: CellPattern
    terms: List[CellRefExpr]


@dataclass
class CodeDef:
    """
    Top-level code definition:

        code square_lattice(d: Int) as ChainComplex over Z2 { ... }

    For now, we store:
      - name
      - param_names: ["d"]
      - cell_families
      - boundary_rules
    """
    name: str
    param_names: List[str]
    cell_families: List[CellFamilyDecl]
    boundary_rules: List[BoundaryRule]


# ============================================================
#  Compiler: AST -> ChainComplex
# ============================================================

def iterate_family_indices(
    family: CellFamilyDecl, base_env: Dict[str, int]
) -> Iterable[Tuple[Tuple[int, ...], Dict[str, int]]]:
    """
    Enumerate all index tuples for a given cell family, returning:

        (indices_tuple, env_with_params_and_indices)

    where env includes:
      - top-level parameters (from base_env),
      - bound index variables for this family.
    """
    ranges = family.index_ranges

    def rec(i: int, current_indices: List[int], env: Dict[str, int]):
        if i == len(ranges):
            yield (tuple(current_indices), env)
            return

        fr = ranges[i]
        upper = eval_expr(fr.upper_expr, env)
        if upper < 0:
            raise ValueError(f"Fin({upper}) is invalid (upper < 0)")
        for v in range(upper):
            new_env = dict(env)
            new_env[fr.var_name] = v
            current_indices.append(v)
            yield from rec(i + 1, current_indices, new_env)
            current_indices.pop()

    yield from rec(0, [], dict(base_env))


def compile_chain_complex(code: CodeDef, param_values: Dict[str, int]) -> ChainComplex:
    """
    Compile a CodeDef (AST) and concrete parameter values into a ChainComplex.

    This version assumes *periodic* (toric) behavior for index arithmetic:
    whenever a boundary rule produces an index outside the range [0, upper),
    we wrap it modulo 'upper'.
    """
    # 1. Parameter environment
    env_params = dict(param_values)
    for p in code.param_names:
        if p not in env_params:
            raise ValueError(f"Missing value for parameter {p!r}")

    # 2. Index families by name
    family_by_name: Dict[str, CellFamilyDecl] = {
        fam.name: fam for fam in code.cell_families
    }

    # Precompute upper bounds for each family and each index dimension
    upper_bounds_by_family: Dict[str, List[int]] = {}
    for fam in code.cell_families:
        uppers = []
        for fr in fam.index_ranges:
            upper = eval_expr(fr.upper_expr, env_params)
            if upper <= 0:
                raise ValueError(
                    f"Fin({upper}) is invalid for family {fam.name!r} (must be > 0)"
                )
            uppers.append(upper)
        upper_bounds_by_family[fam.name] = uppers

    # 3. Enumerate all cells, grouped by dimension
    cells_dim2: List[Tuple[str, Tuple[int, ...]]] = []
    cells_dim1: List[Tuple[str, Tuple[int, ...]]] = []
    cells_dim0: List[Tuple[str, Tuple[int, ...]]] = []

    cell_index_dim2: Dict[Tuple[str, Tuple[int, ...]], int] = {}
    cell_index_dim1: Dict[Tuple[str, Tuple[int, ...]], int] = {}
    cell_index_dim0: Dict[Tuple[str, Tuple[int, ...]], int] = {}

    for fam in code.cell_families:
        for indices, env_with_indices in iterate_family_indices(fam, env_params):
            key = (fam.name, indices)
            if fam.dim == 2:
                idx = len(cells_dim2)
                cells_dim2.append(key)
                cell_index_dim2[key] = idx
            elif fam.dim == 1:
                idx = len(cells_dim1)
                cells_dim1.append(key)
                cell_index_dim1[key] = idx
            elif fam.dim == 0:
                idx = len(cells_dim0)
                cells_dim0.append(key)
                cell_index_dim0[key] = idx
            else:
                raise ValueError(
                    f"Unsupported dimension {fam.dim} for family {fam.name!r}"
                )

    n2 = len(cells_dim2)
    n1 = len(cells_dim1)
    n0 = len(cells_dim0)

    # 4. Allocate boundary matrices
    d2 = np.zeros((n1, n2), dtype=np.int8)  # C2 -> C1
    d1 = np.zeros((n0, n1), dtype=np.int8)  # C1 -> C0

    # 5. Apply boundary rules
    for rule in code.boundary_rules:
        op = rule.op_name
        src_name = rule.src_pattern.name
        src_family = family_by_name[src_name]

        # Sanity: pattern index vars match family ranges (by count and names)
        if len(rule.src_pattern.index_vars) != len(src_family.index_ranges):
            raise ValueError(
                f"Pattern for {op}({src_name}[...]) has wrong number of indices"
            )
        pattern_vars = rule.src_pattern.index_vars
        family_vars = [fr.var_name for fr in src_family.index_ranges]
        if pattern_vars != family_vars:
            raise ValueError(
                f"Pattern indices {pattern_vars} do not match family indices {family_vars}"
            )

        # For each source cell instance
        for indices, env_with_indices in iterate_family_indices(src_family, env_params):
            key_src = (src_family.name, indices)

            if op == "d2":
                if src_family.dim != 2:
                    raise ValueError("d2 must act from C2 to C1 (src dim != 2)")
                col = cell_index_dim2[key_src]

            elif op == "d1":
                if src_family.dim != 1:
                    raise ValueError("d1 must act from C1 to C0 (src dim != 1)")
                col = cell_index_dim1[key_src]

            else:
                raise ValueError(f"Unknown boundary operator {op!r}")

            # Evaluate each term in the sum, with modular wrap
            for term in rule.terms:
                tgt_family = family_by_name[term.name]
                uppers = upper_bounds_by_family[tgt_family.name]

                raw_indices = [
                    eval_expr(e, env_with_indices) for e in term.index_exprs
                ]
                if len(raw_indices) != len(uppers):
                    raise ValueError(
                        f"Term for {term.name} has wrong arity: "
                        f"expected {len(uppers)}, got {len(raw_indices)}"
                    )

                # Periodic / toric behavior: indices modulo their upper bounds
                term_indices = tuple(
                    (v % ub) for v, ub in zip(raw_indices, uppers)
                )

                key_tgt = (tgt_family.name, term_indices)

                if op == "d2":
                    if tgt_family.dim != 1:
                        raise ValueError("d2 target must be in C1 (dim=1)")
                    row = cell_index_dim1[key_tgt]
                    d2[row, col] ^= 1

                elif op == "d1":
                    if tgt_family.dim != 0:
                        raise ValueError("d1 target must be in C0 (dim=0)")
                    row = cell_index_dim0[key_tgt]
                    d1[row, col] ^= 1

    # 6. Build ChainComplex (this will check d1 ∘ d2 = 0)
    return ChainComplex(
        dim_C2=n2,
        dim_C1=n1,
        dim_C0=n0,
        d2=d2,
        d1=d1,
        basis_C2=cells_dim2,
        basis_C1=cells_dim1,
        basis_C0=cells_dim0,
        name=code.name,
    )



# ============================================================
#  Example: AST for square_lattice(d)
# ============================================================

def square_lattice_ast() -> CodeDef:
    """
    AST for a *toric* square-lattice code:

      code toric_square_lattice(d: Int) as ChainComplex over Z2 {
        cells {
          faces  F[x: Fin d, y: Fin d];
          edgesx Ex[x: Fin d, y: Fin d];  // horizontal
          edgesy Ey[x: Fin d, y: Fin d];  // vertical
          verts  V[x: Fin d, y: Fin d];
        }

        boundary {
          // d2: C2 -> C1
          d2(F[x,y]) =
            Ex[x,y] + Ex[x,y+1] +
            Ey[x,y] + Ey[x+1,y];

          // d1: C1 -> C0
          d1(Ex[x,y]) = V[x,y] + V[x,y+1];
          d1(Ey[x,y]) = V[x,y] + V[x+1,y];
        }
      }

    The '+1' in indices is interpreted modulo d (toric boundary conditions).
    """

    d = Var("d")

    Fin_d = lambda v: FinRange(v, d)

    cell_families = [
        # faces F[x: Fin d, y: Fin d] -> dim 2
        CellFamilyDecl(
            dim=2,
            name="F",
            index_ranges=[Fin_d("x"), Fin_d("y")],
        ),
        # horizontal edges Ex[x: Fin d, y: Fin d] -> dim 1
        CellFamilyDecl(
            dim=1,
            name="Ex",
            index_ranges=[Fin_d("x"), Fin_d("y")],
        ),
        # vertical edges Ey[x: Fin d, y: Fin d] -> dim 1
        CellFamilyDecl(
            dim=1,
            name="Ey",
            index_ranges=[Fin_d("x"), Fin_d("y")],
        ),
        # vertices V[x: Fin d, y: Fin d] -> dim 0
        CellFamilyDecl(
            dim=0,
            name="V",
            index_ranges=[Fin_d("x"), Fin_d("y")],
        ),
    ]

    # d2(F[x,y]) = Ex[x,y] + Ex[x,y+1] + Ey[x,y] + Ey[x+1,y]
    rule_d2 = BoundaryRule(
        op_name="d2",
        src_pattern=CellPattern("F", ["x", "y"]),
        terms=[
            # bottom horizontal edge
            CellRefExpr("Ex", [Var("x"), Var("y")]),
            # top horizontal edge
            CellRefExpr("Ex", [BinOp("+", Var("x"), IntLit(1)), Var("y")]),
            # left vertical edge
            CellRefExpr("Ey", [Var("x"), Var("y")]),
            # right vertical edge
            CellRefExpr("Ey", [Var("x"), BinOp("+", Var("y"), IntLit(1))]),
        ],
    )

    # d1(Ex[x,y]) = V[x,y] + V[x,y+1]
    rule_d1_Ex = BoundaryRule(
        op_name="d1",
        src_pattern=CellPattern("Ex", ["x", "y"]),
        terms=[
            CellRefExpr("V", [Var("x"), Var("y")]),
            CellRefExpr("V", [Var("x"), BinOp("+", Var("y"), IntLit(1))]),
        ],
    )

    # d1(Ey[x,y]) = V[x,y] + V[x+1,y]
    rule_d1_Ey = BoundaryRule(
        op_name="d1",
        src_pattern=CellPattern("Ey", ["x", "y"]),
        terms=[
            CellRefExpr("V", [Var("x"), Var("y")]),
            CellRefExpr("V", [BinOp("+", Var("x"), IntLit(1)), Var("y")]),
        ],
    )

    return CodeDef(
        name="toric_square_lattice",
        param_names=["d"],
        cell_families=cell_families,
        boundary_rules=[rule_d2, rule_d1_Ex, rule_d1_Ey],
    )



# ============================================================
#  Demo / sanity check
# ============================================================

def pretty_print_matrix(M, name="M"):
    print(f"\n{name} (shape={M.shape}):")
    print(M)


def main():
    code = square_lattice_ast()
    chain = compile_chain_complex(code, {"d": 3})

    print(chain)

    Hx, Hz, comm = css_from_chain(chain)

    print(f"Number of qubits (edges): {chain.dim_C1}")
    print(f"Number of X checks (faces): {Hx.shape[0]}")
    print(f"Number of Z checks (vertices): {Hz.shape[0]}")

    pretty_print_matrix(Hx, name="Hx (X stabilizers from faces)")
    pretty_print_matrix(Hz, name="Hz (Z stabilizers from vertices)")

    if np.any(comm):
        print("\n[ERROR] CSS commutation violated: Hz * Hx^T != 0 over F2")
        print("Non-zero entries in commutator matrix:")
        print(comm)
    else:
        print("\n[OK] CSS commutation holds: Hz * Hx^T = 0 over F2")


if __name__ == "__main__":
    main()
