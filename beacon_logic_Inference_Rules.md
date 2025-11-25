## **3.X Inference Rules for Beacon Logic**

We now recast Beacon Logic in the style of an inference system,
using rule schemata similar to those employed in type systems and operational semantics.

We write judgments of the form:

- \(P, f \vdash_{\text{loc}} n \in B\) — node \(n\) belongs to the *local* beacon set of function \(f\) in program \(P\).
- \(P, f \vdash_{\text{glob}} n \in B\) — node \(n\) belongs to the *global* beacon set of function \(f\).
- \(P \vdash n \in B(P)\) — node \(n\) belongs to the program-level beacon set.

The rules below are intended to operate under the engineering conditions EC1–EC9 described in Section 3.

---

### **3.X.1 Local Logic: \(\mathcal{L}_{\text{loc}}\)**

#### **Judgment Form**

\[
P, f \vdash_{\text{loc}} n \in B
\]

is read as: “under program \(P\), function \(f\), the node \(n\) is in the local beacon set.”

---

#### **Rule L-OUT (Output Root)**

\[
\frac{
n \in O(f)
}{
P, f \vdash_{\text{loc}} n \in B
}
\;\;\textsc{[L-OUT]}
\]

Intuitively: any observable output node is a local beacon root.

---

#### **Rule L-DEP (Dependency Expansion)**

We assume a (possibly over-approximating) dependence relation
\(Dep(n, n')\) meaning “\(n\) influences \(n'\)” (EC1, EC3).

\[
\frac{
P, f \vdash_{\text{loc}} n' \in B
\quad
Dep(n, n')
}{
P, f \vdash_{\text{loc}} n \in B
}
\;\;\textsc{[L-DEP]}
\]

This rule inductively grows the local beacon set backward along data and control dependences.

---

#### **Rule L-VAL (Validation Filtering)**

Validation logic is modeled as a *post-filter* over the local closure.
We introduce an auxiliary judgment

\[
P, f \vdash_{\text{loc}} n \in B^{\text{raw}}
\]

constructed solely by `L-OUT` and `L-DEP`.
We then define a filtered judgment \(B^{\text{flt}}\) as:

\[
\frac{
P, f \vdash_{\text{loc}} n \in B^{\text{raw}}
\quad
n \notin V(f)
}{
P, f \vdash_{\text{loc}} n \in B^{\text{flt}}
}
\;\;\textsc{[L-VAL-KEEP]}
\]

\[
\frac{
P, f \vdash_{\text{loc}} n \in B^{\text{raw}}
\quad
n \in V(f)
}{
P, f \nvdash_{\text{loc}} n \in B^{\text{flt}}
}
\;\;\textsc{[L-VAL-DROP]}
\]

where \(V(f)\) denotes heuristically identified validation nodes (EC4).
Thus, we always construct a *raw* closure first, and then filter validation logic.

---

#### **Rule L-RED (Reduction / Normalization)**

Reduction is modeled as a transformation from the filtered set
\(B^{\text{flt}}\) to a normalized set \(B^{*}\):

\[
\frac{
P, f \vdash_{\text{loc}} n \in B^{\text{flt}}
\quad
Reduce(B^{\text{flt}}) = B^{*}
}{
P, f \vdash_{\text{loc}} n \in B^{*}
}
\;\;\textsc{[L-RED]}
\]

`Reduce` is not defined by inference rules, but by an implementation-specific
procedure (deduplication, ranking, compression) subject to EC5
and may trade completeness for interpretability.

In summary, the “local beacons” used by the global layer are:

\[
B_{\text{local}}^*(f) = \{ n \mid P, f \vdash_{\text{loc}} n \in B^{*} \}.
\]

---

### **3.X.2 Global Logic: \(\mathcal{L}_{\text{glob}}\)**

#### **Judgment Form**

\[
P, f \vdash_{\text{glob}} n \in B
\]

is read as: “under program \(P\), node \(n\) belongs to the global beacon set of function \(f\).”

---

#### **Rule G-BASE (Lift Local to Global)**

Every normalized local beacon is also a global beacon for that function:

\[
\frac{
P, f \vdash_{\text{loc}} n \in B^{*}
}{
P, f \vdash_{\text{glob}} n \in B
}
\;\;\textsc{[G-BASE]}
\]

This rule seeds the global beacon set with the local one.

---

#### **Rule G-CALL (Call Linking / Semantic Inlining)**

Suppose \(c\) is a call-site in function \(f_i\) that invokes function \(f_j\)
and is part of the global beacons of \(f_i\).
We write \(\textit{call}(c, f_j)\) for this relation.

\[
\frac{
P, f_i \vdash_{\text{glob}} c \in B
\quad
\textit{call}(c, f_j)
\quad
P, f_j \vdash_{\text{glob}} n \in B
}{
P, f_i \vdash_{\text{glob}} n \in B
}
\;\;\textsc{[G-CALL]}
\]

This rule propagates the callee’s beacons back into the caller (EC6),
implementing semantic inlining at the level of beacon sets.

---

#### **Rule G-RET (Return-Value Propagation)**

Let \(ret_j\) be the return node of function \(f_j\),
and suppose the value of \(ret_j\) flows into function \(f_i\) and influences its beacons.
We abstract this data-flow relation as \(\textit{RetFlow}(f_j, f_i)\).

\[
\frac{
\textit{RetFlow}(f_j, f_i)
\quad
P, f_j \vdash_{\text{glob}} n \in B
}{
P, f_i \vdash_{\text{glob}} n \in B
}
\;\;\textsc{[G-RET]}
\]

This captures return-value–based interprocedural influence (EC7).

---

#### **Rule G-GLOB (Global State Propagation)**

Let \(g\) be a global variable, and let \(\textit{Read}(g)\) and \(\textit{Write}(g)\)
denote sets of nodes that read and write \(g\) respectively.

We first introduce a program-level judgment:

\[
P \vdash n \in B(P)
\]

and then enforce the following propagation rule:

\[
\frac{
P \vdash n_r \in B(P)
\quad
n_r \in \textit{Read}(g)
\quad
n_w \in \textit{Write}(g)
}{
P \vdash n_w \in B(P)
}
\;\;\textsc{[G-GLOB]}
\]

Symmetrically, if a write to \(g\) is in the beacon set, all corresponding reads are included:

\[
\frac{
P \vdash n_w \in B(P)
\quad
n_w \in \textit{Write}(g)
\quad
n_r \in \textit{Read}(g)
}{
P \vdash n_r \in B(P)
}
\;\;\textsc{[G-GLOB-2]}
\]

These rules model the conservative treatment of global state (EC8).

In practice, the program-level judgment \(P \vdash n \in B(P)\)
is induced from the function-level global judgments by the entry rule below.

---

### **3.X.3 Entry and Program-Level Beacons**

#### **Rule P-ENTRY (Program Entry Beacons)**

Let \(f_{\text{entry}}\) denote the program entry point (main or public API, EC9).
We define:

\[
\frac{
P, f_{\text{entry}} \vdash_{\text{glob}} n \in B
}{
P \vdash n \in B(P)
}
\;\;\textsc{[P-ENTRY]}
\]

In other words, the program-level beacon set \(B(P)\) is the global beacon set
of the chosen entry function.

---

### **3.X.4 Discussion**

The above rules capture the essence of Beacon Logic as an **inference system**:

- `L-OUT` and `L-DEP` generate a raw, over-approximate backward slice.
- `L-VAL` and `L-RED` refine this slice into a compact, human-readable local beacon set.
- `G-BASE`, `G-CALL`, and `G-RET` approximate interprocedural influence via calls and return flows.
- `G-GLOB` and `P-ENTRY` lift this reasoning to a program-wide perspective.

Crucially, the rules are intended to be interpreted **under the engineering conditions EC1–EC9** rather than as purely semantic, idealized reasoning.
In this sense, Beacon Logic occupies a middle ground between classical program slicing and practical, explainability-driven program abstractions, making it especially suitable as a backbone for code retrieval, Beacon Flow Graph construction, and LLM-guided code generation.
