## **System Architecture**

Our system follows a **deterministic, two-round Beacon-guided pipeline**, where structured semantic reasoning is used to constrain code generation and verification.

At a high level, the execution flow is:

[
\text{Input} \rightarrow \text{Beacon Logic} \rightarrow \text{Beacon IR} \rightarrow \text{Generator} \rightarrow \text{Rebuilder} \rightarrow \text{Verifier} \rightarrow (\text{Accept} \mid \text{Revise})
]

If verification fails, a **single revision round** is triggered:

[
\text{Revise} = \text{Generator} \rightarrow \text{Rebuilder} \rightarrow \text{Verifier}
]

The system terminates after at most **two rounds**.

---

## **1. Input Layer (Task + Project Context)**

The system starts from a structured task and its project context:

* A **Task Object** is constructed from the input (e.g., CodeEval record)
* A **Project Index** is built from the repository

This step produces:

[
(\mathcal{T}, \mathcal{P})
]

where:

* (\mathcal{T}): task specification (signature, docstring, metadata)
* (\mathcal{P}): project-level context (symbols, files, dependencies)

---

## **2. Beacon Logic (Semantic Core Extraction)**

Beacon Logic extracts a **structured semantic core** from the task and context.

It consists of two layers:

* **Local Logic**: builds intra-function dependency closure
* **Global Logic**: propagates semantics across calls and global state

The process follows a deterministic pipeline:

1. preprocess
2. local reasoning (output rooting + dependency expansion)
3. global reasoning (call / return / global propagation)
4. IR construction
5. beacon tree construction
6. signature hint extraction
7. refinement
8. normalization
9. constraint summarization

Formally, Beacon Logic computes:

[
(\mathcal{T}, \mathcal{P}) \rightarrow (\mathcal{B}, \mathcal{C})
]

where:

* (\mathcal{B}): Beacon IR
* (\mathcal{C}): constraint summary

This corresponds to the formal system defined in Beacon Logic 

---

## **3. Beacon IR (Structured Semantic Representation)**

Beacon IR is a structured intermediate representation:

[
\mathcal{B} = \langle N, E, \Pi \rangle
]

where:

* (N): semantic nodes (core operations)
* (E): dependency edges

  * dataflow
  * control
  * call
  * return
  * global state
* (\Pi): provenance (rule origins such as L-DEP, G-CALL, etc.)

This representation encodes:

* **what must be implemented**
* **how components depend on each other**
* **why each element exists**

It serves as the **single source of truth** for generation and verification.

---

## **4. Code Generator (Constraint-Guided Generation)**

The generator produces code conditioned on:

* task specification (\mathcal{T})
* beacon structure (\mathcal{B})
* constraints (\mathcal{C})

Unlike standard prompting, Beacon IR is treated as a **hard structural constraint**, not optional context.

The generator is required to:

* follow required call structure
* respect dependency relations
* avoid forbidden patterns
* output **code only**

---

## **5. Rebuilder (Semantic Reconstruction)**

The rebuilder integrates generated code back into the task context:

1. patch generated code into original file
2. reconstruct a new task
3. re-run Beacon Logic

[
\hat{y} \rightarrow \mathcal{B}_{rebuild}
]

This step enables **semantic comparison between intended structure and generated structure**.

---

## **6. Verifier (Structure-Level Validation)**

The verifier checks alignment between:

* original Beacon IR (\mathcal{B})
* rebuilt Beacon IR (\mathcal{B}_{rebuild})

Verification includes:

* beacon coverage (missing nodes)
* dependency consistency
* call/return correctness
* symbol grounding
* constraint violations

The verifier outputs:

* acceptance flag
* structured issues
* revision guidance

---

## **7. One-Step Revision (Bounded Correction)**

If verification fails:

* issues are converted into additional constraints
* generator runs **one more time only**

[
\mathcal{C}' = \mathcal{C} \cup \text{VerifierFeedback}
]

This produces a revised candidate, which is again rebuilt and verified.

**Maximum rounds = 2**

This design ensures:

* bounded cost
* deterministic behavior
* no infinite refinement loop

---

## **8. Final Output**

The final result is:

* the accepted code from either:

  * main round, or
  * revision round

The pipeline also produces structured artifacts:

* Beacon IR
* constraints
* generated code (both rounds)
* verifier reports
* execution trace

