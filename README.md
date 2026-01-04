# QECType

A type system and high-level programming language for constructing, composing, and reasoning about **quantum error-correcting (QEC) codes** through **chain complexes** and **verified combinators**.

QECType is not a stabilizer-matrix DSL. Instead, it treats

* **chain complexes**
* **boundary & coboundary operators**
* **homological structure**
* **global code-construction combinators**

as **first-class typed objects**.

The goal is to make QEC code construction:

* mathematically precise
* architecturally composable
* syntactically succinct
* and structurally verifiable

without manually enumerating stabilizers or incidence graphs.

---

## High-level idea

Most major QEC code families share a common foundation:

**they arise from chain complexes and morphisms between them**

[
C_2 \xrightarrow{d_2} C_1 \xrightarrow{d_1} C_0,
\quad\text{with } d_1\circ d_2 = 0.
]

QECType adopts this as the core abstraction.

1. **Users describe a QEC code as a typed chain complex**

* cells (0-cells, 1-cells, 2-cells, …)
* boundary maps (`d₂`, `d₁`, …)
* finite index sets / recursive definitions

2. The compiler automatically derives:

* stabilizer matrices
* qubit placements
* X / Z generators
* commutation guarantees (from `d₁ ∘ d₂ = 0`)

3. Larger codes are constructed via **typed combinators** that preserve structure by construction:

* tensor products
* hypergraph products
* homological products
* graph lifts
* symmetry quotients
* boundary selections (rough/smooth)
* puncturing / holes
* concatenation
* stacking / thickening

Users program **generative structure**, not raw matrices.

---

## Key Principles

QECType is built around four design foundations:

### ✔ Chain complexes as first-class types

```
ChainComplex {
  C2 --d2--> C1 --d1--> C0
}
```

Boundary operators are typed objects, not ad-hoc matrices.

---

### ✔ Global combinators instead of manual construction

Instead of writing stabilizers or adjacency lists, users invoke

* `hypergraph_product(A,B)`
* `homological_product(A,B)`
* `graph_lift(Base, Group)`
* `boundary_select(...)`
* `quotient_by_symmetry(...)`
* `puncture(...)`
* `concatenate(...)`

The compiler ensures:

[
d_1 d_2 = 0 ;\Rightarrow; H_Z H_X^T = 0
]

automatically.

---

### ✔ Recursive & inductive program semantics

Code families are expressed as **inductive constructions**, e.g.

* repetition(d) from repetition(d–1)
* surface(d) from surface(d–1)
* lifted-product codes from lifted base graphs

This enables **structural proofs** and avoids flat index arithmetic.

---

### ✔ Separation of concerns

The language separates:

| Layer                     | Responsibility                                 |
| ------------------------- | ---------------------------------------------- |
| Chain complex definitions | Local structure, boundary maps                 |
| Combinators               | Global algebraic + topological transformations |
| CSS extraction            | Producing stabilizer code representations      |
| Backend                   | Compilation, simulation, export formats        |

This makes the system extensible to new QEC families.

---

# Syntax (Conceptual Overview)

The language supports:

* parameterized chain complexes
* finite index types (`Fin n`)
* recursive definitions
* dependent cell sets
* boundary rules
* combinator application

Example (surface-like 2-complex):

```
code square_lattice(d: Int) as ChainComplex over Z2 {

  cells {
    faces  F[x: Fin(d-1), y: Fin(d-1)];
    edgesx Ex[x: Fin(d-1), y: Fin d];
    edgesy Ey[x: Fin d,     y: Fin(d-1)];
    verts  V[x: Fin d,      y: Fin d];
  }

  boundary {
    d2(F[x,y]) =
      Ex[x,y] + Ex[x,y+1] +
      Ey[x,y] + Ey[x+1,y];

    d1(Ex[x,y]) = V[x,y] + V[x,y+1];
    d1(Ey[x,y]) = V[x,y] + V[x+1,y];
  }
}
```

CSS extraction:

```
css {
  hx = matrix(d2);
  hz = transpose(matrix(d1));
}
```

The compiler guarantees commutation.

---

# Examples

## Surface code patch via boundary combinator

```
let C = square_lattice(d-1);

css = boundary_select(
  C,
  x_cells = interior_faces(C),
  z_cells = interior_vertices(C)
);
```

---

## Hypergraph Product Code

```
let A = repetition(d);
let B = repetition(d);

include hypergraph_product(A,B);
```

Works for any A,B classical chain complexes.

---

## Lifted-Product LDPC Code

```
let Base = classical_ldpc_small();
let L = graph_lift(Base, group = Z_p);

include hypergraph_product(L, L);
```

---

## Concatenation

```
let outer = steane();
let inner = surface_patch(d);

include concatenate(outer, inner);
```

---

# Planned Backend Targets

* stabilizer matrices (HX, HZ)
* Pauli generator representations
* syndrome graph export
* QEC simulators
* circuit-generator backends
* LaTeX / TikZ lattice visualization
* Python binding for experimentation

---

# TODO List

* [ ] Define formal core type system for ChainComplex + CSSCode
* [ ] Specify well-typed boundary operators (`d₁ ∘ d₂ = 0`)
* [ ] Design recursive / inductive code definitions (`Fin n`, structural recursion)
* [ ] Implement core combinator library:

  * [ ] tensor, dual
  * [ ] hypergraph product
  * [ ] homological product
  * [ ] graph lift / covers
  * [ ] quotient by symmetry
  * [ ] boundary selection
  * [ ] puncturing / holes
  * [ ] concatenation
* [ ] Develop CSS extraction + commutation validation
* [ ] Implement matrix + graph + visualization backends
* [ ] Provide canonical examples (surface, toric, color, HGP, lifted-product)
* [ ] Formalize correctness properties & verification hooks
