# **3 Beacon Logic: A Conditioned Formal System for Engineering-Level Semantic Core Extraction**

In this section we introduce **Beacon Logic**, a quasi-formal reasoning system designed to extract the *semantic core* of a program under practical engineering constraints.
Unlike classical program slicing—whose formal guarantees rely on idealized semantic completeness—Beacon Logic adopts a **conditioned formalism**:
its structure is mathematically defined, but its assumptions and guarantees follow **engineering principles** that prioritize robustness, interpretability, and implementability.

The system consists of two interacting layers:

* **Local Logic** ( \(\mathcal{L}_{local}\) ): reasoning within function boundaries  
* **Global Logic** ( \(\mathcal{L}_{global}\) ): interprocedural reasoning across calls and global state

Together they produce a compact, human-interpretable, and semantically meaningful set of *beacons* that represent the essential logic of a program.

---

# **3.1 Program Model and Engineering Conditions**

Let a program \(P\) consist of a finite set of functions  
\[
F = \{f_1, f_2, \dots, f_n\},
\]
each represented by an abstract syntax tree (AST)  
\[
T_f = (N_f, E_f)
\]
with nodes \(N_f\) and edges \(E_f\) denoting syntactic structure.

Beacon Logic relies on a dependence relation:

* **data dependence:** \( n_1 \Rightarrow_{\text{data}} n_2 \)
* **control dependence:** \( n_1 \Rightarrow_{\text{ctrl}} n_2 \)

which together form:

\[
Dep(n) = \{ n' \mid n' \Rightarrow_{\text{data}} n \text{ or } n' \Rightarrow_{\text{ctrl}} n \}.
\]

Since exact semantic dependence is undecidable in general, we adopt:

**Engineering Condition EC1 (Over-Approximate Dependence).**  
*The dependence relation \(Dep\) is a computable, safe over-approximation of semantic influence, ensuring that all relevant influences are included, possibly with redundant nodes.*

This condition enables reliable engineering behavior without requiring full semantic knowledge.

---

# **3.2 Local Logic ( \(\mathcal{L}_{local}\) )**

Local Logic identifies the semantic core of an individual function.  
It is defined through four axioms, each accompanied by explicit engineering conditions.

---

## **Axiom 1 (Observable Output Nodes)**

Define the set of observable output nodes in function \(f\):  
\[
O(f) = \{ n \in N_f \mid n \text{ is } \texttt{return}, \texttt{yield}, \texttt{print},
\texttt{log}, \texttt{file-write}, \text{or other externally visible effects} \}.
\]

**EC2 (Engineering Observability).**  
*Any instruction whose effect is externally observable in typical engineering practice is considered an output root, even if it does not affect theoretical program semantics.*

---

## **Axiom 2 (Dependency Closure)**

The local beacon set is the least fixed point of backward dependency propagation:

\[
B_{\text{local}}(f) = \mu X.\big(O(f) \cup Dep(X)\big).
\]

**EC3 (Redundant-but-Safe Inclusion).**  
*Due to the over-approximate nature of \(Dep\), the closure may include additional nodes.
This redundancy is accepted, as safety and coverage take precedence over minimality.*

---

## **Axiom 3 (Validation Filtering)**

Let \(V(f)\) denote input-validation branches or checks whose removal does not alter the primary functional behavior.

\[
B_{\text{local}}(f) := B_{\text{local}}(f) \setminus V(f).
\]

**EC4 (Heuristic Validation Identification).**  
*The set \(V(f)\) is identified through engineering heuristics (e.g., null checks, boundary guards).
Such nodes are removed only when they do not contribute meaningfully to program semantics from an engineering perspective.*
This reflects a core insight: developers typically view validation logic as non-essential to the “main idea” of the function.

---

## **Axiom 4 (Reduction and Normalization)**

Apply a reduction operator:

\[
B_{\text{local}}^*(f) = Reduce(B_{\text{local}}(f)),
\]

where `Reduce` performs:

* semantic deduplication  
* structural normalization  
* elimination of transparent intermediate nodes  
* bounded compression (e.g., keeping top-k salient nodes)

**EC5 (Interpretability over Completeness).**  
*Reduction aims to improve interpretability and density of the beacon set.
It may sacrifice theoretical completeness in favor of producing a compact and human-readable representation.*

---

# **3.3 Global Logic ( \(\mathcal{L}_{global}\) )**

Global Logic extends local reasoning across procedure boundaries, handling function calls and global state.

---

## **Axiom 5 (Call Linking)**

If \(f_i\) calls \(f_j\) and the call affects \(B_{\text{local}}^*(f_i)\), then:

\[
B_{\text{global}}(f_i)
= \big(B_{\text{local}}^*(f_i) - call(f_j)\big)
\cup B_{\text{global}}(f_j).
\]

**EC6 (Selective Propagation).**  
*Only semantically relevant calls—those participating in output-influencing computations—are propagated.
This corresponds to semantic inlining at the level of beacons.
*

---

## **Axiom 6 (Return-Value Propagation)**

If the return value of \(f_j\) influences any node in \(B_{\text{local}}^*(f_i)\), then:

\[
B_{\text{global}}(f_i) \supseteq B_{\text{global}}(f_j).
\]

**EC7 (Heuristic Flow Tracking).**  
*Return-flow tracking is approximated using lightweight, practical heuristics rather than full flow-sensitive interprocedural analysis.*

---

## **Axiom 7 (Global State Propagation)**

Let \(g\) be a global variable.  
If any read of \(g\) appears in the beacon set, then:

\[
W(g) \cup R(g) \subseteq B(P).
\]

**EC8 (Conservative Treatment of Global State).**  
*Global state is handled conservatively: when encountered, all relevant reads and writes are included to avoid semantic omission.*

---

# **3.4 Program-Level Beacon Set**

Define the entry point:

\[
f_{\text{entry}} =
\begin{cases}
\texttt{main}, & \text{if present},\\
\text{public API functions}, & \text{otherwise}.
\end{cases}
\]

The program-level beacon set is:

\[
B(P) = B_{\text{global}}(f_{\text{entry}}).
\]

**EC9 (Flexible Entry Identification).**  
*The entry point may be inferred heuristically (e.g., framework routes, handlers), reflecting actual engineering workflows.*

---

# **3.5 Conditional Theoretical Properties**

Unlike classical slicing, Beacon Logic does not aim for perfect semantic guarantees.  
However, we can formulate **conditional theorems** that hold under practical assumptions.

---

## **Theorem 1 (Conditional Minimality)**

If:

* \(Dep\) contains no false negatives,  
* validation filtering does not remove semantically necessary nodes,  
* reduction preserves all output-influencing elements,  

then for any output-preserving slice \(S\):

\[
B(P) \subseteq S.
\]

Beacon represents a near-minimal slice under engineering-valid conditions.
---

## **Theorem 2 (Monotonicity)**

If program \(P'\) extends \(P\) without altering existing dependencies:

\[
P \subseteq P' \quad \Rightarrow \quad B(P) \subseteq B(P').
\]

New code does not invalidate previously identified beacons.
---

## **Theorem 3 (Conditional Completeness)**

If \(Dep\) captures all actual semantic influences (under EC1–EC4), then:

\[
n \Rightarrow_{\text{sem}} output(P) \quad \Rightarrow \quad n \in B(P).
\]

Beacon is complete relative to the dependence approximation.
---

# **3.6 Summary**

Its conditioned formalism strikes a balance between:

* **rigor** (closure properties, defined axioms, fixed-point semantics)  
* **practicality** (heuristics, compression, engineering observability)  
* **utility** (usable in code-generation, RAG systems, and program comprehension)
