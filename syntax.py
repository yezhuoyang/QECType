from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterable


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
    Top-level code definition:

        code toric_square_lattice(d: Int) as ChainComplex over Z2 { ... }
    """
    name: str
    param_names: List[str]
    cell_families: List[CellFamilyDecl]
    boundary_rules: List[BoundaryRule]


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

    # ------------- top-level -------------

    def parse_program(self) -> CodeDef:
        self.expect_keyword("code")
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
        self.expect_symbol("{")

        # cells block
        self.expect_keyword("cells")
        self.expect_symbol("{")
        cell_families: List[CellFamilyDecl] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise ValueError("Unexpected EOF inside cells block")
            if tok.kind == "SYMBOL" and tok.value == "}":
                self.advance()
                break
            cell_families.append(self.parse_cell_decl())

        # boundary block
        self.expect_keyword("boundary")
        self.expect_symbol("{")
        boundary_rules: List[BoundaryRule] = []
        while True:
            tok = self.peek()
            if tok is None:
                raise ValueError("Unexpected EOF inside boundary block")
            if tok.kind == "SYMBOL" and tok.value == "}":
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
#  Example source program (toric square lattice)
# ============================================================

SOURCE = r"""
code toric_square_lattice(d: Int) as ChainComplex over Z2 {
  cells {
    faces  F[x: Fin d, y: Fin d];
    edgesx Ex[x: Fin d, y: Fin d];
    edgesy Ey[x: Fin d, y: Fin d];
    verts  V[x: Fin d, y: Fin d];
  }

  boundary {
    d2(F[x,y]) =
      Ex[x,y] + Ex[x+1,y] +
      Ey[x,y] + Ey[x,y+1];

    d1(Ex[x,y]) = V[x,y] + V[x,y+1];
    d1(Ey[x,y]) = V[x,y] + V[x+1,y];
  }
}
"""


def pretty_print_matrix(M, name="M"):
    print(f"\n{name} (shape={M.shape}):")
    print(M)


def main():
    # In the future you can read from a file:
    #   with open("toric.qec", "r") as f:
    #       src = f.read()
    src = SOURCE

    tokens = tokenize(src)
    parser = Parser(tokens)
    code = parser.parse_program()

    print("Parsed code:", code.name, "params", code.param_names)

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
