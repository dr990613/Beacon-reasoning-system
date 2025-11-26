#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BeaconExtractor

- Implements Beacon Logic:
  * Local Logic: outputs, dependency closure, validation filter, reduction
  * Global Logic: call graph, entry-driven aggregation
- Adds structural "Beacon Tree" (derivation-tree-like) visualization.

Usage:
    python beacon_extractor_pro_v4.py your_file.py
    python beacon_extractor_pro_v4.py your_file.py --mode full --json beacons.json --tree
"""

import ast
import sys
import argparse
import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple


class BeaconExtractor(ast.NodeVisitor):
    """Beacon Extractor implementing Local + Global Beacon Logic with tree view."""

    def __init__(self, source_code: str):
        self.source_code = source_code.splitlines()
        self.tree = ast.parse(source_code)

        # Node bookkeeping
        self.node_lines: Dict[int, int] = {}        # node_id -> lineno
        self.node_func: Dict[int, str] = {}         # node_id -> function name (or "<module>")
        self.node_ast: Dict[int, ast.AST] = {}      # node_id -> AST node

        # Function ranges: name -> (start_line, end_line)
        self.func_ranges: Dict[str, Tuple[int, int]] = {}

        # Dependency graph: consumer_node -> set(producer_node)
        self.dep_graph: Dict[int, Set[int]] = defaultdict(set)

        # Output nodes per function: func -> set(node_id)
        self.output_nodes: Dict[str, Set[int]] = defaultdict(set)

        # Validation returns per function: func -> set(node_id)
        self.validation_returns: Dict[str, Set[int]] = defaultdict(set)

        # Call graph: func -> set(callee_names)
        self.calls: Dict[str, Set[str]] = defaultdict(set)

        # Variable definitions: func -> var_name -> list[node_id of Assign]
        self.var_defs: Dict[str, Dict[str, List[int]]] = defaultdict(
            lambda: defaultdict(list)
        )

        # Module-related
        self.module_name = "<module>"
        self.current_func: str = self.module_name

    # =========================================================
    # Utility helpers
    # =========================================================

    def _id(self, node: ast.AST) -> int:
        return id(node)

    def _record_node(self, node: ast.AST):
        nid = self._id(node)
        self.node_ast[nid] = node
        lineno = getattr(node, "lineno", None)
        if lineno is not None:
            self.node_lines[nid] = lineno
        self.node_func[nid] = self.current_func

    def _safe_get_line(self, node_id: int) -> str:
        lineno = self.node_lines.get(node_id)
        if lineno is None:
            return ""
        if 1 <= lineno <= len(self.source_code):
            return self.source_code[lineno - 1].rstrip("\n")
        return ""

    # =========================================================
    # Dependency helpers
    # =========================================================

    def _extract_identifiers(self, node: ast.AST) -> List[ast.Name]:
        """Recursively extract all ast.Name nodes inside `node`."""
        ids: List[ast.Name] = []
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                ids.append(child)
        return ids

    def _add_dep(self, node: ast.AST, value: ast.AST):
        """
        Add dependencies: node <- identifiers used in value,
        and link identifiers to their definitions (var_defs).

        This is our over-approximate Dep relation.
        """
        if value is None:
            return
        nid = self._id(node)
        for name_node in self._extract_identifiers(value):
            name_id = self._id(name_node)
            # Expression-level dependency: current node depends on this Name occurrence
            self.dep_graph[nid].add(name_id)
            self._record_node(name_node)

            var_name = name_node.id

            # Definitions in current function
            for def_id in self.var_defs[self.current_func].get(var_name, []):
                self.dep_graph[nid].add(def_id)

            # Definitions at module level (globals / constants)
            for def_id in self.var_defs[self.module_name].get(var_name, []):
                self.dep_graph[nid].add(def_id)

    # =========================================================
    # AST visitor methods
    # =========================================================

    def visit_Module(self, node: ast.Module):
        self.current_func = self.module_name
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        prev_func = self.current_func
        self.current_func = node.name
        self._record_node(node)

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is not None:
            if end is None:
                end = start
            self.func_ranges[node.name] = (start, end)

        self.generic_visit(node)
        self.current_func = prev_func

    def visit_Assign(self, node: ast.Assign):
        """
        Handle assignments, including:
        - simple: x = ...
        - mutable: df[c] = ...  (treated as W(df))
        """
        self._record_node(node)
        nid = self._id(node)

        # RHS dependencies
        self._add_dep(node, node.value)

        # Record definitions for targets
        for target in node.targets:
            # simple variable: x = ...
            if isinstance(target, ast.Name):
                var_name = target.id
                self.var_defs[self.current_func][var_name].append(nid)

            # mutable object update: df[c] = ...
            elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                base_name = target.value.id  # e.g., df
                # treat as a write to the base object (W(df))
                self.var_defs[self.current_func][base_name].append(nid)
                self._record_node(target.value)

        self.generic_visit(node)

    def visit_Return(self, node: ast.Return):
        self._record_node(node)
        nid = self._id(node)

        # Mark as output node (observable behavior)
        self.output_nodes[self.current_func].add(nid)

        # Simple validation heuristic: constant 0/None/False as validation-like
        if isinstance(node.value, ast.Constant):
            if node.value.value in (0, None, False):
                self.validation_returns[self.current_func].add(nid)

        # Dependencies from return expression
        if node.value is not None:
            self._add_dep(node, node.value)

        self.generic_visit(node)

    def visit_Expr(self, node: ast.Expr):
        self._record_node(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        self._record_node(node)
        nid = self._id(node)

        # print(...) is treated as output
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            self.output_nodes[self.current_func].add(nid)

        # Dependencies from args and keywords
        for arg in node.args:
            self._add_dep(node, arg)
        for kw in node.keywords:
            self._add_dep(node, kw.value)

        # Call graph
        if isinstance(node.func, ast.Name):
            callee = node.func.id
            self.calls[self.current_func].add(callee)

        self.generic_visit(node)

    # =========================================================
    # Local Logic: per-function Beacon closure
    # =========================================================

    def compute_local_closure(self, func: str) -> Set[int]:
        """
        Backward closure from outputs of a given function.
        Includes dependencies inside that function and module-level defs.
        """
        outputs = self.output_nodes.get(func, set())
        visited: Set[int] = set()
        stack: List[int] = list(outputs)

        while stack:
            nid = stack.pop()
            if nid in visited:
                continue

            # Only consider nodes from this function or module
            f = self.node_func.get(nid, self.module_name)
            if f not in (func, self.module_name):
                continue

            visited.add(nid)
            for dep in self.dep_graph.get(nid, ()):
                if dep not in visited:
                    stack.append(dep)

        return visited

    def filter_validation_local(self, func: str, beacons: Set[int]) -> Set[int]:
        """Filter validation returns for a given function."""
        valids = self.validation_returns.get(func, set())
        return {nid for nid in beacons if nid not in valids}

    def _score_node(self, node: ast.AST) -> int:
        """
        Heuristic importance score for reduction:
        - Return: 100
        - Call: 90
        - Assign(Call): 80
        - Assign(Subscript...): 70 (e.g., df[c] = ...)
        - If / For / While: 40
        - Others: 10
        """
        if isinstance(node, ast.Return):
            return 100

        if isinstance(node, ast.Call):
            return 90

        if isinstance(node, ast.Assign):
            if isinstance(node.value, ast.Call):
                return 80
            # Assign to subscript: df[c] = ...
            for t in node.targets:
                if isinstance(t, ast.Subscript):
                    return 70
            return 50

        if isinstance(node, (ast.If, ast.For, ast.While)):
            return 40

        return 10

    def reduce_local(
        self,
        func: str,
        beacons: Set[int],
        max_per_func: int = 20,
        mode: str = "compact",
    ) -> List[int]:
        """
        Basic reduction:
        - Deduplicate nodes
        - Merge by line (one representative node per line)
        - Prioritize higher-score nodes
        - In compact mode, drop low-score nodes (noise)
        """
        line_to_nodes: Dict[int, List[int]] = defaultdict(list)
        for nid in beacons:
            if nid not in self.node_lines:
                continue
            line = self.node_lines[nid]
            line_to_nodes[line].append(nid)

        scored: List[Tuple[int, int, int]] = []  # (score, line, node_id)

        for line, nodes in line_to_nodes.items():
            best_nid = None
            best_score = -1
            for nid in nodes:
                node = self.node_ast.get(nid)
                if node is None:
                    continue
                score = self._score_node(node)
                if score > best_score:
                    best_score = score
                    best_nid = nid
            if best_nid is not None:
                scored.append((best_score, line, best_nid))

        # Sort by score desc, then by line asc
        scored.sort(key=lambda x: (-x[0], x[1]))

        if mode == "compact":
            # Drop low-score nodes to keep only salient beacons
            scored = [item for item in scored if item[0] >= 50]

        # Truncate if too long
        if max_per_func is not None and max_per_func > 0:
            scored = scored[:max_per_func]

        # Return node_ids sorted by line number
        result = [nid for _, _, nid in sorted(scored, key=lambda x: x[1])]
        return result

    # =========================================================
    # Global Logic: call graph + entry-driven aggregation
    # =========================================================

    def find_entry_points(self, explicit_entry: str | None = None) -> List[str]:
        """
        Heuristic for entry points:
        - If explicit_entry is given and exists, use it
        - Else prefer 'main'
        - Else fall back to '<module>'
        """
        if explicit_entry is not None:
            if explicit_entry in self.func_ranges or explicit_entry == self.module_name:
                return [explicit_entry]

        entries: List[str] = []
        if "main" in self.func_ranges:
            entries.append("main")
        else:
            entries.append(self.module_name)
        return entries

    def reachable_functions_from(self, entry: str) -> Set[str]:
        """DFS on call graph starting from entry."""
        visited: Set[str] = set()
        stack: List[str] = [entry]

        while stack:
            f = stack.pop()
            if f in visited:
                continue
            visited.add(f)
            for callee in self.calls.get(f, set()):
                # Only consider functions we know
                if callee in self.func_ranges or callee == self.module_name:
                    stack.append(callee)

        return visited

    def compute_all_local_beacons(
        self,
        max_per_func: int = 20,
        mode: str = "compact",
    ) -> Dict[str, List[int]]:
        """
        Compute reduced local beacons for all functions (including <module>).
        """
        all_funcs = set(self.func_ranges.keys()) | {self.module_name}
        result: Dict[str, List[int]] = {}
        for func in all_funcs:
            closure = self.compute_local_closure(func)
            closure = self.filter_validation_local(func, closure)
            reduced = self.reduce_local(func, closure, max_per_func=max_per_func, mode=mode)
            result[func] = reduced
        return result

    def compute_program_beacons(
        self,
        max_per_func: int = 20,
        mode: str = "compact",
        explicit_entry: str | None = None,
    ) -> Dict[str, List[int]]:
        """
        Program-level Beacons:
        - find entry points
        - for each entry, find reachable functions
        - union their local beacons
        """
        local_beacons = self.compute_all_local_beacons(max_per_func=max_per_func, mode=mode)
        entries = self.find_entry_points(explicit_entry=explicit_entry)

        program_beacons: Dict[str, List[int]] = {}

        for entry in entries:
            reachable = self.reachable_functions_from(entry)
            # always include module-level beacons
            reachable.add(self.module_name)

            collected: Set[int] = set()
            for f in reachable:
                collected.update(local_beacons.get(f, []))

            ordered = sorted(collected, key=lambda nid: self.node_lines.get(nid, 999999))
            program_beacons[entry] = ordered

        return program_beacons

    # =========================================================
    # Pretty-print reasoning (flat)
    # =========================================================

    def print_output_nodes(self):
        print("=== Output Nodes (Returns / Prints) ===")
        for func, nodes in self.output_nodes.items():
            print(f"\n[Function: {func}]")
            for nid in sorted(nodes, key=lambda x: self.node_lines.get(x, 999999)):
                line = self.node_lines.get(nid)
                mark = ""
                if nid in self.validation_returns.get(func, set()):
                    mark = "  (validation-like)"
                print(f"  line {line}: {self._safe_get_line(nid)}{mark}")

    def print_dep_graph(self, max_nodes: int = 50):
        print("\n=== Dependency Graph (consumer -> producers) ===")
        count = 0
        for nid, deps in self.dep_graph.items():
            if not deps:
                continue
            line = self.node_lines.get(nid)
            print(f"\n[line {line}] {self._safe_get_line(nid)}")
            print("  depends on:")
            for d in sorted(deps, key=lambda x: self.node_lines.get(x, 999999)):
                dline = self.node_lines.get(d)
                print(f"    -> [line {dline}] {self._safe_get_line(d)}")
            count += 1
            if count >= max_nodes:
                print("  ... (truncated)")
                break

    def print_calls(self):
        print("\n=== Function Calls (call graph) ===")
        for func, callees in self.calls.items():
            if not callees:
                continue
            print(f"[{func}] calls: {', '.join(sorted(callees))}")

    def print_local_beacons(self, local_beacons: Dict[str, List[int]]):
        print("\n=== LOCAL BEACONS (per function, reduced) ===")
        for func, nodes in local_beacons.items():
            print(f"\n[Function: {func}]")
            for nid in nodes:
                line = self.node_lines.get(nid)
                print(f"  Beacon @ line {line}: {self._safe_get_line(nid)}")

    def print_program_beacons(self, program_beacons: Dict[str, List[int]]):
        print("\n=== PROGRAM-LEVEL BEACONS (entry-driven, flat) ===")
        for entry, nodes in program_beacons.items():
            reachable = self.reachable_functions_from(entry)
            print(f"\n[Entry: {entry}]  (reachable functions: {', '.join(sorted(reachable))} )")
            for nid in nodes:
                line = self.node_lines.get(nid)
                func = self.node_func.get(nid, self.module_name)
                print(f"  [{func}] line {line}: {self._safe_get_line(nid)}")

    # =========================================================
    # Beacon Tree (derivation-like structural view)
    # =========================================================

    def print_program_beacon_tree(
        self,
        program_beacons: Dict[str, List[int]],
        explicit_entry: str | None = None,
    ):
        """
        Print a derivation-tree-like view of program beacons:
        - Entry as root
        - Global config cluster
        - Per-function local trees (rooted at outputs, following dependencies)
        - Call sites annotated with "→ calls f(...)"
        """
        print("\n=== PROGRAM-LEVEL BEACON TREE (derivation-style) ===")

        for entry, nodes in program_beacons.items():
            # If user specified entry, skip others
            if explicit_entry is not None and entry != explicit_entry:
                continue

            entry_nodes = set(nodes)
            reachable = self.reachable_functions_from(entry)

            print(f"\nProgram Beacon Tree (Entry = {entry})")
            print("──────────────────────────────────────────")

            # Partition nodes: module-level vs per-function
            module_nodes = [nid for nid in entry_nodes
                            if self.node_func.get(nid, self.module_name) == self.module_name]

            func_nodes_map: Dict[str, List[int]] = defaultdict(list)
            for nid in entry_nodes:
                func = self.node_func.get(nid, self.module_name)
                if func != self.module_name:
                    func_nodes_map[func].append(nid)

            # 1. Global config cluster
            if module_nodes:
                print("┌─ Global config (module-level beacons)")
                for nid in sorted(module_nodes, key=lambda x: self.node_lines.get(x, 999999)):
                    line = self.node_lines.get(nid)
                    code = self._safe_get_line(nid)
                    print(f"│    [<module>] line {line}: {code}")
                print("└─ end of global config\n")

            # 2. Function subtrees
            funcs_sorted = sorted(func_nodes_map.keys())

            for i, func in enumerate(funcs_sorted):
                is_last_func = (i == len(funcs_sorted) - 1)
                prefix = "└─" if is_last_func else "├─"
                print(f"{prefix} Function {func}()")

                local_nodes = set(func_nodes_map[func])
                # roots: output nodes in local_nodes; fallback: nodes not used as deps
                outputs = self.output_nodes.get(func, set()) & local_nodes
                if outputs:
                    roots = sorted(outputs, key=lambda x: self.node_lines.get(x, 999999))
                else:
                    # nodes that are never a dependency of others in same function
                    used_as_dep: Set[int] = set()
                    for nid in local_nodes:
                        for dep in self.dep_graph.get(nid, ()):
                            if dep in local_nodes:
                                used_as_dep.add(dep)
                    roots = sorted(
                        [nid for nid in local_nodes if nid not in used_as_dep],
                        key=lambda x: self.node_lines.get(x, 999999),
                    )

                visited: Set[int] = set()
                # child indent prefix for nodes under this function
                func_indent = "   " if is_last_func else "│  "

                for j, root in enumerate(roots):
                    is_last_root = (j == len(roots) - 1)
                    self._print_node_tree(
                        root,
                        local_nodes,
                        reachable,
                        visited,
                        indent=func_indent,
                        is_last=is_last_root,
                    )

    def _print_node_tree(
        self,
        nid: int,
        local_nodes: Set[int],
        reachable_funcs: Set[str],
        visited: Set[int],
        indent: str,
        is_last: bool,
    ):
        """
        Recursively print a small tree for a function:
        - Node is printed as "[func] line X: code"
        - Children are its dependencies within local_nodes
        - Call nodes are annotated with "→ calls f(...)"
        """
        if nid in visited:
            # avoid infinite loops on cycles, just mark
            line = self.node_lines.get(nid)
            func = self.node_func.get(nid, self.module_name)
            print(f"{indent}{'└─' if is_last else '├─'} [visited] [{func}] line {line}")
            return

        visited.add(nid)
        line = self.node_lines.get(nid)
        func = self.node_func.get(nid, self.module_name)
        code = self._safe_get_line(nid)
        node = self.node_ast.get(nid)

        branch = "└─" if is_last else "├─"
        # Check if node is a call to another beaconed function
        call_annot = ""
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            callee = node.func.id
            if callee in reachable_funcs and callee in self.func_ranges:
                call_annot = f"   → calls {callee}(...)"

        print(f"{indent}{branch} [{func}] line {line}: {code}{call_annot}")

        # children: dependencies within the same function
        children = [d for d in self.dep_graph.get(nid, ())
                    if d in local_nodes and d != nid]

        if not children:
            return

        # sort children by line
        children = sorted(children, key=lambda x: self.node_lines.get(x, 999999))

        # extend indent for next level
        child_indent = indent + ("   " if is_last else "│  ")
        for i, child in enumerate(children):
            self._print_node_tree(
                child,
                local_nodes,
                reachable_funcs,
                visited,
                indent=child_indent,
                is_last=(i == len(children) - 1),
            )

    # =========================================================
    # JSON export
    # =========================================================

    def to_json(
        self,
        local_beacons: Dict[str, List[int]],
        program_beacons: Dict[str, List[int]],
    ) -> Dict:
        """Export beacons and structure to a JSON-serializable dict."""
        def node_info(nid: int) -> Dict:
            return {
                "func": self.node_func.get(nid, self.module_name),
                "line": self.node_lines.get(nid),
                "code": self._safe_get_line(nid),
            }

        local = {
            func: [node_info(nid) for nid in nodes]
            for func, nodes in local_beacons.items()
        }

        program = {
            entry: [node_info(nid) for nid in nodes]
            for entry, nodes in program_beacons.items()
        }

        call_graph = {
            func: sorted(list(callees))
            for func, callees in self.calls.items()
        }

        return {
            "local_beacons": local,
            "program_beacons": program,
            "call_graph": call_graph,
        }


# =============================================================
#  MAIN CLI
# =============================================================

def main():
    parser = argparse.ArgumentParser(
        description="BeaconExtractor Pro v4 - Beacon Logic implementation for Python files."
    )
    parser.add_argument("path", help="Path to the Python file to analyze.")
    parser.add_argument(
        "--mode",
        choices=["compact", "full"],
        default="compact",
        help="Reduction mode: 'compact' keeps only high-importance beacons, 'full' keeps more details.",
    )
    parser.add_argument(
        "--max-per-func",
        type=int,
        default=20,
        help="Maximum number of local beacons per function (after reduction).",
    )
    parser.add_argument(
        "--entry",
        type=str,
        default=None,
        help="Explicit entry function name (otherwise heuristic: main or <module>).",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional path to save beacons as JSON.",
    )
    parser.add_argument(
        "--max-dep-print",
        type=int,
        default=60,
        help="Max number of dependency graph nodes to print for reasoning.",
    )
    parser.add_argument(
        "--tree",
        action="store_true",
        help="Also print a derivation-style Beacon Tree view.",
    )

    args = parser.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        code = f.read()

    extractor = BeaconExtractor(code)
    extractor.visit(extractor.tree)

    print("\n============= BEACON REASONING =============")
    extractor.print_output_nodes()
    extractor.print_dep_graph(max_nodes=args.max_dep_print)
    extractor.print_calls()

    local_beacons = extractor.compute_all_local_beacons(
        max_per_func=args.max_per_func,
        mode=args.mode,
    )
    extractor.print_local_beacons(local_beacons)

    program_beacons = extractor.compute_program_beacons(
        max_per_func=args.max_per_func,
        mode=args.mode,
        explicit_entry=args.entry,
    )
    print("\n============= BEACON RESULT (PROGRAM LEVEL) =============")
    extractor.print_program_beacons(program_beacons)

    if args.tree:
        extractor.print_program_beacon_tree(
            program_beacons,
            explicit_entry=args.entry,
        )

    if args.json is not None:
        data = extractor.to_json(local_beacons, program_beacons)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] JSON beacons written to {args.json}")


if __name__ == "__main__":
    main()
