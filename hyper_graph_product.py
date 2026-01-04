from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable, Union


# ============================================================
#  Core chain complex backend
# ============================================================

class ChainComplex:
    """
    Minimal chain complex over F2:

        C2 --d2--> C1 --d1--> C0

    Internally:
      - d2: (dim_C1 x dim_C2) matrix over F2
      - d1: (dim_C0 x dim_C1) matrix over F2

    We enforce the chain condition d1 ∘ d2 = 0 (mod 2).
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
        self.dim_C2 = dim_C2
        self.dim_C1 = dim_C1
        self.dim_C0 = dim_C0

        self.d2 = np.array(d2, dtype=np.int8)  # shape (dim_C1 x dim_C2)
        self.d1 = np.array(d1, dtype=np.int8)  # shape (dim_C0 x dim_C1)

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
    Hx = (chain.d2.T) % 2
    Hz = (chain.d1) % 2
    comm = (Hz @ Hx.T) % 2
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
    op: str  # '+', '-', '*'
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
#  AST: cells, ranges, boundary rules, code defs, combinators
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

    dim = 2: cells in C2 (faces)
    dim = 1: cells in C1 (edges)
    dim = 0: cells in C0 (vertices)
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

        d2(F[x,y]) = Ex[x,y] + Ex[x+1,y] + Ey[x,y] + Ey[x,y+1];
    """
    op_name: str  # "d2" or "d1"
    src_pattern: CellPattern
    terms: List[CellRefExpr]


@dataclass
class CodeDef:
    """
    Primitive chain complex definition:

        code X(d: Int) as ChainComplex over Z2 { cells {...} boundary {...} }
    """
    name: str
    param_names: List[str]
    cell_families: List[CellFamilyDecl]
    boundary_rules: List[BoundaryRule]


@dataclass
class CodeCall:
    """
    A reference to another code with actual parameters:

        repetition(d)
        repetition(d-1)
    """
    name: str
    arg_exprs: List[Expr]


@dataclass
class HypergraphProductDef:
    """
    Code defined by hypergraph product of two other codes:

        code rep_hgp(d: Int) as ChainComplex over Z2 =
          hypergraph_product(repetition(d), repetition(d));
    """
    name: str
    param_names: List[str]
    left: CodeCall
    right: CodeCall


CodeLike = Union[CodeDef, HypergraphProductDef]


# ============================================================
#  Compiler: AST -> ChainComplex (toric / periodic indices)
# ============================================================

def iterate_family_indices(
    family: CellFamilyDecl, base_env: Dict[str, int]
) -> Iterable[Tuple[Tuple[int, ...], Dict[str, int]]]:
    """
    Enumerate all index tuples for a given cell family, returning:

        (indices_tuple, env_with_params_and_indices)
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
    Compile a primitive CodeDef (cells + boundary) and concrete parameter values
    into a ChainComplex.

    This version assumes *periodic* (toric) behavior for index arithmetic:
    whenever a boundary rule produces an index outside the range [0, upper),
    we wrap it modulo 'upper'. For open chains (repetition), nothing goes
    out-of-bounds so wrapping is a no-op.
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
#  Hypergraph product combinator (backend)
# ============================================================

def hypergraph_product(A: ChainComplex, B: ChainComplex) -> ChainComplex:
    """
    Hypergraph product of two *classical* chain complexes A, B that each
    have a single nontrivial differential d1: C1 -> C0.

    We use the standard chain-complex-level construction:

      D2 = C1^A ⊗ C1^B
      D1 = (C1^A ⊗ C0^B) ⊕ (C0^A ⊗ C1^B)
      D0 = C0^A ⊗ C0^B

    with differentials (over F2):

      ∂2(eA ⊗ eB) = (eA ⊗ ∂b(eB)) ⊕ (∂a(eA) ⊗ eB)
      ∂1(eA ⊗ vB, vA ⊗ eB) = ∂a(eA) ⊗ vB + vA ⊗ ∂b(eB)

    This yields a 2D ChainComplex representing the CSS hypergraph product code.
    """
    H1 = A.d1  # shape (m1a, n1a)
    H2 = B.d1  # shape (m1b, n1b)
    m1a, n1a = H1.shape
    m1b, n1b = H2.shape

    # Dimensions
    dim_C2 = n1a * n1b  # edges x edges
    dim_C1 = n1a * m1b + m1a * n1b  # (E1 x V2) ⊕ (V1 x E2)
    dim_C0 = m1a * m1b  # V1 x V2

    d2 = np.zeros((dim_C1, dim_C2), dtype=np.int8)
    d1 = np.zeros((dim_C0, dim_C1), dtype=np.int8)

    # Bases and index maps
    C2_basis: List[Tuple[str, int, int]] = []
    C1_basis: List[Tuple[str, int, int]] = []
    C0_basis: List[Tuple[str, int, int]] = []
    idx_C2: Dict[Tuple[str, int, int], int] = {}
    idx_C1: Dict[Tuple[str, int, int], int] = {}
    idx_C0: Dict[Tuple[str, int, int], int] = {}

    # C2: edgesA x edgesB
    for i in range(n1a):
        for j in range(n1b):
            key = ("E1E2", i, j)
            idx = len(C2_basis)
            C2_basis.append(key)
            idx_C2[key] = idx

    # C1 part1: edgesA x vertsB
    for i in range(n1a):
        for u in range(m1b):
            key = ("E1V2", i, u)
            idx = len(C1_basis)
            C1_basis.append(key)
            idx_C1[key] = idx

    # C1 part2: vertsA x edgesB
    for v in range(m1a):
        for j in range(n1b):
            key = ("V1E2", v, j)
            idx = len(C1_basis)
            C1_basis.append(key)
            idx_C1[key] = idx

    # C0: vertsA x vertsB
    for v in range(m1a):
        for u in range(m1b):
            key = ("V1V2", v, u)
            idx = len(C0_basis)
            C0_basis.append(key)
            idx_C0[key] = idx

    # Fill d2: D2 -> D1
    for i in range(n1a):
        for j in range(n1b):
            col = idx_C2[("E1E2", i, j)]

            # Term: eA_i ⊗ ∂b(eB_j)
            for u in range(m1b):
                if H2[u, j] & 1:
                    row = idx_C1[("E1V2", i, u)]
                    d2[row, col] ^= 1

            # Term: ∂a(eA_i) ⊗ eB_j
            for v in range(m1a):
                if H1[v, i] & 1:
                    row = idx_C1[("V1E2", v, j)]
                    d2[row, col] ^= 1

    # Fill d1: D1 -> D0

    # Part1: E1V2
    for i in range(n1a):
        for u in range(m1b):
            col = idx_C1[("E1V2", i, u)]
            # ∂1(eA_i ⊗ vB_u) = ∂a(eA_i) ⊗ vB_u
            for v in range(m1a):
                if H1[v, i] & 1:
                    row = idx_C0[("V1V2", v, u)]
                    d1[row, col] ^= 1

    # Part2: V1E2
    for v in range(m1a):
        for j in range(n1b):
            col = idx_C1[("V1E2", v, j)]
            # ∂1(vA_v ⊗ eB_j) = vA_v ⊗ ∂b(eB_j)
            for u in range(m1b):
                if H2[u, j] & 1:
                    row = idx_C0[("V1V2", v, u)]
                    d1[row, col] ^= 1

    return ChainComplex(
        dim_C2=dim_C2,
        dim_C1=dim_C1,
        dim_C0=dim_C0,
        d2=d2,
        d1=d1,
        basis_C2=C2_basis,
        basis_C1=C1_basis,
        basis_C0=C0_basis,
        name=f"HGP({A.name},{B.name})",
    )


# ============================================================
#  Tokenizer
# ============================================================

@dataclass
class Token:
    kind: str  # 'IDENT', 'INT', 'SYMBOL', 'KEYWORD'
    value: str


KEYWORDS = {"code", "as", "ChainComplex", "over", "Z2", "cells", "boundary", "Int", "Fin"}


def tokenize(src: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    n = len(src)

    while i < n:
        c = src[i]

        # whitespace
        if c.isspace():
            i += 1
            continue

        # line comment: // ...
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            i += 2
            while i < n and src[i] not in "\n\r":
                i += 1
            continue

        # identifier or keyword
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            val = src[i:j]
            kind = "KEYWORD" if val in KEYWORDS else "IDENT"
            tokens.append(Token(kind, val))
            i = j
            continue

        # integer literal
        if c.isdigit():
            j = i + 1
            while j < n and src[j].isdigit():
                j += 1
            val = src[i:j]
            tokens.append(Token("INT", val))
            i = j
            continue

        # single-character symbols
        if c in "(){}[],:;=+*-":
            tokens.append(Token("SYMBOL", c))
            i += 1
            continue

        raise ValueError(f"Unexpected character {c!r} at position {i}")

    return tokens


# ============================================================
#  Parser
# ============================================================

class Parser:
    def __init__(self, tokens: List[Token]):
        self.toks = tokens
        self.pos = 0

    # ------------- basic utilities -------------

    def peek(self) -> Token | None:
        return self.toks[self.pos] if self.pos < len(self.toks) else None

    def advance(self) -> Token:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected EOF")
        self.pos += 1
        return tok

    def expect_symbol(self, sym: str):
        tok = self.advance()
        if tok.kind != "SYMBOL" or tok.value != sym:
            raise ValueError(f"Expected symbol {sym!r}, got {tok}")
        return tok

    def expect_keyword(self, kw: str):
        tok = self.advance()
        if tok.kind != "KEYWORD" or tok.value != kw:
            raise ValueError(f"Expected keyword {kw!r}, got {tok}")
        return tok

    def match_symbol(self, sym: str) -> bool:
        tok = self.peek()
        if tok and tok.kind == "SYMBOL" and tok.value == sym:
            self.pos += 1
            return True
        return False

    def expect_ident(self) -> str:
        tok = self.advance()
        if tok.kind != "IDENT":
            raise ValueError(f"Expected identifier, got {tok}")
        return tok.value

    # ------------- module level -------------

    def parse_module(self) -> List[CodeLike]:
        """
        Parse a module with one or more 'code ...' definitions.
        """
        codes: List[CodeLike] = []
        while self.peek() is not None:
            self.expect_keyword("code")
            codes.append(self.parse_single_code())
        return codes

    # ------------- single code definition -------------

    def parse_single_code(self) -> CodeLike:
        """
        Parse either:

          code X(d: Int) as ChainComplex over Z2 { ... }

        or

          code Y(d: Int) as ChainComplex over Z2 =
            hypergraph_product(A(...), B(...));
        """
        name = self.expect_ident()

        # parameter list
        self.expect_symbol("(")
        param_names: List[str] = []
        if not self.match_symbol(")"):
            param_names.append(self.parse_param())
            while self.match_symbol(","):
                param_names.append(self.parse_param())
            self.expect_symbol(")")
        # else: empty param list

        self.expect_keyword("as")
        self.expect_keyword("ChainComplex")
        self.expect_keyword("over")
        self.expect_keyword("Z2")

        # Decide between primitive and hypergraph_product form
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected EOF after 'over Z2'")

        if tok.kind == "SYMBOL" and tok.value == "{":
            # Primitive form: cells + boundary
            self.advance()  # consume '{'

            # cells block
            self.expect_keyword("cells")
            self.expect_symbol("{")
            cell_families: List[CellFamilyDecl] = []
            while True:
                tok2 = self.peek()
                if tok2 is None:
                    raise ValueError("Unexpected EOF inside cells block")
                if tok2.kind == "SYMBOL" and tok2.value == "}":
                    self.advance()
                    break
                cell_families.append(self.parse_cell_decl())

            # boundary block
            self.expect_keyword("boundary")
            self.expect_symbol("{")
            boundary_rules: List[BoundaryRule] = []
            while True:
                tok2 = self.peek()
                if tok2 is None:
                    raise ValueError("Unexpected EOF inside boundary block")
                if tok2.kind == "SYMBOL" and tok2.value == "}":
                    self.advance()
                    break
                boundary_rules.append(self.parse_boundary_rule())

            # closing brace of code block
            self.expect_symbol("}")

            return CodeDef(
                name=name,
                param_names=param_names,
                cell_families=cell_families,
                boundary_rules=boundary_rules,
            )

        elif tok.kind == "SYMBOL" and tok.value == "=":
            # Combinator form: hypergraph_product(...)
            self.advance()  # consume '='
            # hypergraph_product
            func_tok = self.advance()
            if func_tok.kind != "IDENT" or func_tok.value != "hypergraph_product":
                raise ValueError(
                    f"Expected 'hypergraph_product', got {func_tok}"
                )
            self.expect_symbol("(")
            left = self.parse_code_call()
            self.expect_symbol(",")
            right = self.parse_code_call()
            self.expect_symbol(")")
            self.expect_symbol(";")

            return HypergraphProductDef(
                name=name,
                param_names=param_names,
                left=left,
                right=right,
            )

        else:
            raise ValueError(
                f"Unexpected token after 'over Z2': {tok}. "
                "Expected '{{' or '='."
            )

    def parse_param(self) -> str:
        name = self.expect_ident()
        self.expect_symbol(":")
        self.expect_keyword("Int")
        return name

    # ------------- cells block -------------

    def parse_cell_decl(self) -> CellFamilyDecl:
        # e.g. faces  F[x: Fin d, y: Fin d];
        kind_tok = self.advance()
        if kind_tok.kind not in ("IDENT", "KEYWORD"):
            raise ValueError(f"Expected cell kind identifier, got {kind_tok}")
        kind = kind_tok.value

        fam_name = self.expect_ident()
        self.expect_symbol("[")
        ranges = [self.parse_fin_range()]
        while self.match_symbol(","):
            ranges.append(self.parse_fin_range())
        self.expect_symbol("]")
        self.expect_symbol(";")

        # Infer dimension from kind
        kind_lower = kind.lower()
        if kind_lower == "faces":
            dim = 2
        elif kind_lower in ("edgesx", "edgesy", "edges", "edge"):
            dim = 1
        elif kind_lower in ("verts", "vertices", "vertex"):
            dim = 0
        else:
            raise ValueError(f"Unknown cell kind {kind!r} (cannot infer dimension)")

        return CellFamilyDecl(dim=dim, name=fam_name, index_ranges=ranges)

    def parse_fin_range(self) -> FinRange:
        # x: Fin d   or   x: Fin(d-1)
        var_name = self.expect_ident()
        self.expect_symbol(":")
        self.expect_keyword("Fin")
        if self.match_symbol("("):
            upper = self.parse_expr()
            self.expect_symbol(")")
        else:
            upper = self.parse_expr()
        return FinRange(var_name=var_name, upper_expr=upper)

    # ------------- boundary block -------------

    def parse_boundary_rule(self) -> BoundaryRule:
        # d2(F[x,y]) = Ex[x,y] + ...;
        op_tok = self.advance()
        if op_tok.kind != "IDENT":
            raise ValueError(f"Expected boundary operator name, got {op_tok}")
        op_name = op_tok.value

        self.expect_symbol("(")
        src_pat = self.parse_cell_pattern()
        self.expect_symbol(")")
        self.expect_symbol("=")
        terms = self.parse_sum_cell_refs()
        self.expect_symbol(";")

        return BoundaryRule(op_name=op_name, src_pattern=src_pat, terms=terms)

    def parse_cell_pattern(self) -> CellPattern:
        name = self.expect_ident()
        self.expect_symbol("[")
        idx_vars = [self.expect_ident()]
        while self.match_symbol(","):
            idx_vars.append(self.expect_ident())
        self.expect_symbol("]")
        return CellPattern(name=name, index_vars=idx_vars)

    def parse_sum_cell_refs(self) -> List[CellRefExpr]:
        terms = [self.parse_cell_ref()]
        while self.match_symbol("+"):
            terms.append(self.parse_cell_ref())
        return terms

    def parse_cell_ref(self) -> CellRefExpr:
        name = self.expect_ident()
        self.expect_symbol("[")
        exprs = [self.parse_expr()]
        while self.match_symbol(","):
            exprs.append(self.parse_expr())
        self.expect_symbol("]")
        return CellRefExpr(name=name, index_exprs=exprs)

    # ------------- code calls for combinators -------------

    def parse_code_call(self) -> CodeCall:
        """
        Parse a code call, e.g. repetition(d) or repetition(d-1).
        """
        name = self.expect_ident()
        self.expect_symbol("(")
        args: List[Expr] = []
        if not self.match_symbol(")"):
            args.append(self.parse_expr())
            while self.match_symbol(","):
                args.append(self.parse_expr())
            self.expect_symbol(")")
        return CodeCall(name=name, arg_exprs=args)

    # ------------- expressions -------------

    # expr -> term (('+'|'-') term)*
    def parse_expr(self) -> Expr:
        node = self.parse_term()
        while True:
            tok = self.peek()
            if tok and tok.kind == "SYMBOL" and tok.value in ("+", "-"):
                op = tok.value
                self.advance()
                right = self.parse_term()
                node = BinOp(op=op, left=node, right=right)
            else:
                break
        return node

    # term -> factor ( '*' factor )*
    def parse_term(self) -> Expr:
        node = self.parse_factor()
        while True:
            tok = self.peek()
            if tok and tok.kind == "SYMBOL" and tok.value == "*":
                self.advance()
                right = self.parse_factor()
                node = BinOp(op="*", left=node, right=right)
            else:
                break
        return node

    def parse_factor(self) -> Expr:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected EOF in expression")

        if tok.kind == "INT":
            self.advance()
            return IntLit(int(tok.value))
        if tok.kind == "IDENT":
            self.advance()
            return Var(tok.value)
        if tok.kind == "SYMBOL" and tok.value == "(":
            self.advance()
            node = self.parse_expr()
            self.expect_symbol(")")
            return node

        raise ValueError(f"Unexpected token in expression: {tok}")


# ============================================================
#  High-level compilation dispatcher
# ============================================================

def compile_code(
    code_env: Dict[str, CodeLike],
    code_name: str,
    param_values: Dict[str, int],
) -> ChainComplex:
    """
    Dispatch compilation based on whether the code is primitive (CodeDef)
    or derived via hypergraph_product (HypergraphProductDef).
    """
    if code_name not in code_env:
        raise ValueError(f"Unknown code {code_name!r}")

    code = code_env[code_name]

    if isinstance(code, CodeDef):
        # Primitive chain complex
        return compile_chain_complex(code, param_values)

    if isinstance(code, HypergraphProductDef):
        # Derived via hypergraph_product
        # 1. Ensure parameters are provided
        env_params = dict(param_values)
        for p in code.param_names:
            if p not in env_params:
                raise ValueError(
                    f"Missing parameter {p!r} for derived code {code.name!r}"
                )

        # 2. Compile left and right args
        def compile_call(call: CodeCall) -> ChainComplex:
            if call.name not in code_env:
                raise ValueError(
                    f"Unknown callee {call.name!r} in hypergraph_product"
                )
            callee = code_env[call.name]
            if not isinstance(callee, CodeDef):
                # For now, restrict to primitive classical codes as inputs
                raise ValueError(
                    "hypergraph_product currently only supports primitive codes as inputs"
                )
            if len(call.arg_exprs) != len(callee.param_names):
                raise ValueError(
                    f"Arity mismatch calling {call.name!r}: "
                    f"expected {len(callee.param_names)} args, "
                    f"got {len(call.arg_exprs)}"
                )
            child_params: Dict[str, int] = {}
            for p_name, arg_expr in zip(callee.param_names, call.arg_exprs):
                child_params[p_name] = eval_expr(arg_expr, env_params)
            return compile_chain_complex(callee, child_params)

        left_chain = compile_call(code.left)
        right_chain = compile_call(code.right)
        return hypergraph_product(left_chain, right_chain)

    raise TypeError(f"Unsupported code definition type: {type(code)}")


# ============================================================
#  Example QECType program with repetition + hypergraph_product
# ============================================================

SOURCE = r"""
code repetition(d: Int) as ChainComplex over Z2 {
  cells {
    edges E[i: Fin(d-1)];
    verts V[i: Fin d];
  }

  boundary {
    d1(E[i]) = V[i] + V[i+1];
  }
}

code rep_hgp(d: Int) as ChainComplex over Z2 =
  hypergraph_product(repetition(d), repetition(d));
"""


def pretty_print_matrix(M, name="M"):
    print(f"\n{name} (shape={M.shape}):")
    print(M)


def main():
    tokens = tokenize(SOURCE)
    parser = Parser(tokens)
    code_list = parser.parse_module()

    # Build environment of code definitions
    code_env: Dict[str, CodeLike] = {c.name: c for c in code_list}

    print("Parsed codes:", list(code_env.keys()))

    # Compile repetition(d_rep)
    d_rep = 5
    rep_chain = compile_code(code_env, "repetition", {"d": d_rep})
    print("\nRepetition chain:", rep_chain)

    # Compile hypergraph product rep_hgp(d_rep) = HGP(repetition(d), repetition(d))
    hgp_chain = compile_code(code_env, "rep_hgp", {"d": d_rep})
    print("Hypergraph product chain:", hgp_chain)

    Hx_hgp, Hz_hgp, comm_hgp = css_from_chain(hgp_chain)
    print(f"Number of qubits (HGP):   {hgp_chain.dim_C1}")
    print(f"Number of X checks (HGP): {Hx_hgp.shape[0]}")
    print(f"Number of Z checks (HGP): {Hz_hgp.shape[0]}")

    if np.any(comm_hgp):
        print("[ERROR] HGP CSS commutation violated")
        pretty_print_matrix(comm_hgp, name="Hz * Hx^T (HGP)")
    else:
        print("[OK] HGP CSS commutation holds")

    if d_rep <= 5:
        pretty_print_matrix(Hx_hgp, name="Hx (HGP)")
        pretty_print_matrix(Hz_hgp, name="Hz (HGP)")


if __name__ == "__main__":
    main()
