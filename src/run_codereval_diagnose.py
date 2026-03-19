# -*- coding: utf-8 -*-
"""
CoderEval diagnostic runner
目标：
1. 不改动当前主调用方式（仍然走现有 pipeline）
2. 通过 monkey patch 记录 IR / prompt / generation / result
3. 对以下假设输出可读诊断：
   H1: IR 失败是否主要是格式/传入传出问题
   H2: IR 规则是否过严，导致有效信息被过滤
   H3: CodeEval 全部信息是否没有完整传给模型
   H4: Prompt template / state / output contract 是否定义不足
   H5: 生成中间状态是否缺少约束
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import importlib
import inspect
import json
import os
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except Exception:
    yaml = None


# ============================================================
# 基础工具
# ============================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def try_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except Exception:
        return None


def compact_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def first_line_of_code(code: str) -> str:
    for line in (code or "").splitlines():
        if line.strip():
            return line.rstrip()
    return ""


def extract_signature_from_code(code: str) -> str:
    line = first_line_of_code(code)
    if line.startswith("def "):
        return line
    return ""


def looks_like_code(text: str) -> bool:
    if not text:
        return False
    code_signals = [
        "def ", "return ", "import ", "from ", "class ", "if ", "for ", "while ",
        "(", "):", "=", "None", "True", "False"
    ]
    hit = sum(int(sig in text) for sig in code_signals)
    return hit >= 4


def normalize_for_match(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def loose_contains(big: str, small: str) -> bool:
    if not big or not small:
        return False
    return normalize_for_match(small) in normalize_for_match(big)


def excerpt(s: str, n: int = 300) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + " ...[truncated]..."


def safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def to_jsonable(obj: Any, depth: int = 0, max_depth: int = 6) -> Any:
    if depth > max_depth:
        return f"<max_depth:{type(obj).__name__}>"

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {
            str(k): to_jsonable(v, depth + 1, max_depth)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x, depth + 1, max_depth) for x in obj]

    if dataclasses.is_dataclass(obj):
        return {
            "__dataclass__": obj.__class__.__name__,
            **{
                f.name: to_jsonable(getattr(obj, f.name), depth + 1, max_depth)
                for f in dataclasses.fields(obj)
            }
        }

    if hasattr(obj, "__dict__"):
        try:
            return {
                "__class__": obj.__class__.__name__,
                **{
                    k: to_jsonable(v, depth + 1, max_depth)
                    for k, v in vars(obj).items()
                    if not k.startswith("_")
                }
            }
        except Exception:
            pass

    return repr(obj)


def object_summary(obj: Any) -> Dict[str, Any]:
    info = {
        "type": type(obj).__name__,
        "repr": excerpt(repr(obj), 500),
    }
    if obj is None:
        info["is_none"] = True
        return info

    if isinstance(obj, dict):
        info["len"] = len(obj)
        info["keys"] = list(obj.keys())[:50]
        return info

    if isinstance(obj, (list, tuple, set)):
        info["len"] = len(obj)
        info["sample_types"] = list({type(x).__name__ for x in list(obj)[:10]})
        return info

    if dataclasses.is_dataclass(obj):
        info["fields"] = [f.name for f in dataclasses.fields(obj)]
        return info

    if hasattr(obj, "__dict__"):
        try:
            info["attrs"] = [k for k in vars(obj).keys() if not k.startswith("_")][:50]
        except Exception:
            pass

    return info


def find_strings(obj: Any, min_len: int = 40, limit: int = 50) -> List[str]:
    out: List[str] = []

    def _walk(x: Any) -> None:
        nonlocal out
        if len(out) >= limit:
            return
        if isinstance(x, str):
            if len(x) >= min_len:
                out.append(x)
            return
        if isinstance(x, dict):
            for v in x.values():
                _walk(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                _walk(v)
            return
        if dataclasses.is_dataclass(x):
            for f in dataclasses.fields(x):
                _walk(getattr(x, f.name))
            return
        if hasattr(x, "__dict__"):
            try:
                for v in vars(x).values():
                    _walk(v)
            except Exception:
                return

    _walk(obj)
    return out[:limit]


def token_set_from_context(context_str: str) -> List[str]:
    """
    尽量从 oracle_context / all_context 里提 token。
    """
    if not context_str:
        return []
    raw = context_str
    parsed = try_json_loads(context_str)
    pieces: List[str] = []
    if isinstance(parsed, dict):
        for v in parsed.values():
            if isinstance(v, str):
                pieces.append(v)
            elif isinstance(v, list):
                pieces.extend([str(x) for x in v])
    else:
        pieces.append(raw)

    joined = " ".join(pieces)
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]{1,60}", joined)
    stop = {"import", "file", "class", "classes", "apis", "vars", "param", "return"}
    uniq = []
    seen = set()
    for t in toks:
        if t.lower() in stop:
            continue
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq[:50]


# ============================================================
# 任务 / 配置加载
# ============================================================

def load_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    text = read_text(config_path)
    if yaml is not None:
        return yaml.safe_load(text)
    # 退化：如果没装 yaml，就尝试 json
    return json.loads(text)


def load_record_by_task_id(json_path: Path, task_id: str) -> Dict[str, Any]:
    data = json.loads(read_text(json_path))

    if isinstance(data, dict):
        # 兼容多种顶层格式
        if "RECORDS" in data and isinstance(data["RECORDS"], list):
            data = data["RECORDS"]
        elif "records" in data and isinstance(data["records"], list):
            data = data["records"]
        elif "tasks" in data and isinstance(data["tasks"], list):
            data = data["tasks"]
        elif "data" in data and isinstance(data["data"], list):
            data = data["data"]
        else:
            data = [data]

    if not isinstance(data, list):
        raise ValueError(f"Unsupported json top-level format: {type(data).__name__}")

    task_id = str(task_id).strip()

    for item in data:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("_id", "")).strip()
        alt_id = str(item.get("task_id", "")).strip()
        if item_id == task_id or alt_id == task_id:
            return item

    # 便于排查：打印前几个 id
    sample_ids = []
    for item in data[:10]:
        if isinstance(item, dict):
            sample_ids.append({
                "_id": item.get("_id"),
                "task_id": item.get("task_id"),
                "name": item.get("name"),
            })

    raise ValueError(
        f"task_id not found: {task_id}. sample ids: {sample_ids}"
    )


def build_task_object_from_record(record: Dict[str, Any]) -> Any:
    """
    尽量适配你当前仓库里的 TaskObject；如果失败，就退回 dict。
    """
    signature = extract_signature_from_code(record.get("code", ""))
    name = record.get("name") or ""
    task_id = record.get("_id") or record.get("task_id") or ""
    task_payload = {
        "task_id": task_id,
        "_id": task_id,
        "name": name,
        "entry_function": name,
        "signature": signature,
        "docstring": record.get("docstring"),
        "prompt": record.get("docstring") or record.get("human_label") or "",
        "human_label": record.get("human_label"),
        "code": record.get("code"),
        "file_content": record.get("file_content"),
        "file_path": record.get("file_path"),
        "project": record.get("project"),
        "package": record.get("package"),
        "oracle_context": record.get("oracle_context"),
        "all_context": record.get("all_context"),
        "dependency": record.get("dependency"),
        "level": record.get("level"),
        "lang": "python",
        "runnable_level": record.get("level"),
        "target_file": record.get("file_path"),
    }

    try:
        mod = importlib.import_module("beacon_system.types")
        TaskObject = getattr(mod, "TaskObject", None)
        if TaskObject is None:
            return task_payload

        # dataclass / 普通类都尽量适配
        if dataclasses.is_dataclass(TaskObject):
            fields = {f.name for f in dataclasses.fields(TaskObject)}
            kwargs = {k: v for k, v in task_payload.items() if k in fields}
            return TaskObject(**kwargs)

        sig = inspect.signature(TaskObject)
        kwargs = {}
        for p in sig.parameters.values():
            if p.name in task_payload:
                kwargs[p.name] = task_payload[p.name]
        return TaskObject(**kwargs)

    except Exception:
        return task_payload


# ============================================================
# 运行时记录器
# ============================================================

class Recorder:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.llm_calls: List[Dict[str, Any]] = []
        self.prompt_candidates: List[str] = []
        self.output_candidates: List[str] = []
        self.ir_candidates: List[Dict[str, Any]] = []
        self.constraint_candidates: List[Dict[str, Any]] = []
        self.result_objects: List[Dict[str, Any]] = []
        self.errors: List[str] = []

        # 新增：按调用编号绑定的 LLM 调用记录
        self.llm_call_records: List[Dict[str, Any]] = []
        self._llm_call_seq: int = 0

    def log(self, kind: str, **payload: Any) -> None:
        self.events.append({
            "kind": kind,
            "time": now_ts(),
            **to_jsonable(payload),
        })

    def add_prompt_candidate(self, text: str, source: str) -> None:
        if text and isinstance(text, str):
            self.prompt_candidates.append(f"[SOURCE={source}]\n{text}")

    def add_output_candidate(self, text: str, source: str) -> None:
        if text and isinstance(text, str):
            self.output_candidates.append(f"[SOURCE={source}]\n{text}")

    def add_error(self, err: str) -> None:
        self.errors.append(err)

    def next_llm_call_id(self) -> int:
        self._llm_call_seq += 1
        return self._llm_call_seq

    def add_llm_call_record(self, record: Dict[str, Any]) -> None:
        self.llm_call_records.append(record)

    def best_prompt(self) -> str:
        if not self.prompt_candidates:
            return ""
        return max(self.prompt_candidates, key=len)

    def best_output(self) -> str:
        if not self.output_candidates:
            return ""
        code_like = [x for x in self.output_candidates if looks_like_code(x)]
        if code_like:
            return max(code_like, key=len)
        return max(self.output_candidates, key=len)

    def best_llm_call(self) -> Dict[str, Any]:
        """
        优先选“输出里像代码”的最后一次 LLM 调用。
        """
        if not self.llm_call_records:
            return {}

        scored: List[Tuple[int, int, Dict[str, Any]]] = []
        for rec in self.llm_call_records:
            output_text = extract_output_text_from_call_record(rec)
            score = 0
            if looks_like_code(output_text):
                score += 10
            fns = parse_generated_function_names(output_text)
            if fns:
                score += 5
            score += min(len(output_text), 5000) // 200
            scored.append((score, rec.get("call_id", 0), rec))

        scored.sort(key=lambda x: (x[0], x[1]))
        return scored[-1][2]


def patch_class_methods(cls: Any, recorder: Recorder, method_names: List[str], tag: str) -> List[Tuple[Any, str, Any]]:
    originals = []
    if cls is None:
        return originals

    for name in method_names:
        if not hasattr(cls, name):
            continue
        original = getattr(cls, name)
        if not callable(original):
            continue

        def make_wrapper(method_name: str, func: Any):
            def wrapper(*args, **kwargs):
                try:
                    recorder.log(
                        f"{tag}.{method_name}.call",
                        args_summary=[object_summary(a) for a in args[:3]],
                        kwargs_summary={k: object_summary(v) for k, v in list(kwargs.items())[:10]},
                    )
                    # 抓字符串
                    for s in find_strings({"args": args, "kwargs": kwargs}, min_len=60, limit=20):
                        lowered = s.lower()
                        if "def " in lowered or "return " in lowered or "you are" in lowered or "function" in lowered:
                            recorder.add_prompt_candidate(s, f"{tag}.{method_name}.args")
                except Exception as e:
                    recorder.add_error(f"[patch pre {tag}.{method_name}] {e}")

                result = func(*args, **kwargs)

                try:
                    recorder.log(
                        f"{tag}.{method_name}.return",
                        result_summary=object_summary(result),
                    )
                    for s in find_strings(result, min_len=60, limit=20):
                        if looks_like_code(s):
                            recorder.add_output_candidate(s, f"{tag}.{method_name}.return")
                        else:
                            recorder.add_prompt_candidate(s, f"{tag}.{method_name}.return")
                except Exception as e:
                    recorder.add_error(f"[patch post {tag}.{method_name}] {e}")

                return result
            return wrapper

        originals.append((cls, name, original))
        setattr(cls, name, make_wrapper(name, original))

    return originals


def patch_module_functions(module: Any, recorder: Recorder) -> List[Tuple[Any, str, Any]]:
    originals = []
    if module is None:
        return originals

    interesting_patterns = [
        "prompt", "generate", "assemble", "build", "ir", "constraint"
    ]

    for name in dir(module):
        if name.startswith("_") and name not in {"_stable_json"}:
            continue
        obj = getattr(module, name, None)
        if not callable(obj):
            continue
        if not any(p in name.lower() for p in interesting_patterns):
            continue

        def make_wrapper(func_name: str, func: Any):
            def wrapper(*args, **kwargs):
                try:
                    recorder.log(
                        f"module.{module.__name__}.{func_name}.call",
                        args_summary=[object_summary(a) for a in args[:3]],
                        kwargs_summary={k: object_summary(v) for k, v in list(kwargs.items())[:10]},
                    )
                    for s in find_strings({"args": args, "kwargs": kwargs}, min_len=80, limit=20):
                        recorder.add_prompt_candidate(s, f"{module.__name__}.{func_name}.args")
                except Exception as e:
                    recorder.add_error(f"[patch pre {module.__name__}.{func_name}] {e}")

                result = func(*args, **kwargs)

                try:
                    recorder.log(
                        f"module.{module.__name__}.{func_name}.return",
                        result_summary=object_summary(result),
                    )

                    # 记录 IR / constraints 候选
                    lname = func_name.lower()
                    if "ir" in lname:
                        recorder.ir_candidates.append({
                            "source": f"{module.__name__}.{func_name}",
                            "summary": object_summary(result),
                            "value": to_jsonable(result),
                        })
                    if "constraint" in lname:
                        recorder.constraint_candidates.append({
                            "source": f"{module.__name__}.{func_name}",
                            "summary": object_summary(result),
                            "value": to_jsonable(result),
                        })

                    for s in find_strings(result, min_len=80, limit=20):
                        if looks_like_code(s):
                            recorder.add_output_candidate(s, f"{module.__name__}.{func_name}.return")
                        else:
                            recorder.add_prompt_candidate(s, f"{module.__name__}.{func_name}.return")
                except Exception as e:
                    recorder.add_error(f"[patch post {module.__name__}.{func_name}] {e}")

                return result
            return wrapper

        originals.append((module, name, obj))
        setattr(module, name, make_wrapper(name, obj))

    return originals


def restore_patches(patches: List[Tuple[Any, str, Any]]) -> None:
    for owner, name, original in patches:
        try:
            setattr(owner, name, original)
        except Exception:
            pass


# ============================================================
# 诊断逻辑
# ============================================================

def extract_target_snippets(record: Dict[str, Any]) -> Dict[str, str]:
    code = record.get("code", "") or ""
    docstring = record.get("docstring", "") or ""
    file_content = record.get("file_content", "") or ""

    target_def = extract_signature_from_code(code)
    doc_first = ""
    for line in docstring.splitlines():
        if line.strip():
            doc_first = line.strip()
            break

    # file_content 中围绕目标函数取一段
    target_func_name = record.get("name", "")
    focus = ""
    m = re.search(
        rf"def\s+{re.escape(target_func_name)}\s*\(.*?(?=\n\ndef |\Z)",
        file_content,
        flags=re.S
    )
    if m:
        focus = m.group(0)

    return {
        "name": record.get("name", ""),
        "signature": target_def,
        "doc_first_line": doc_first,
        "file_path": record.get("file_path", ""),
        "human_label": record.get("human_label", "") or "",
        "target_code": code,
        "target_focus_block": focus,
    }


def parse_generated_function_names(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.M)

def extract_prompt_text_from_call_record(call_record: Dict[str, Any]) -> str:
    texts = call_record.get("prompt_strings", []) or []
    if not texts:
        return ""
    return max(texts, key=len)


def extract_output_text_from_call_record(call_record: Dict[str, Any]) -> str:
    texts = call_record.get("output_strings", []) or []
    if not texts:
        return ""
    code_like = [x for x in texts if looks_like_code(x)]
    if code_like:
        return max(code_like, key=len)
    return max(texts, key=len)


def summarize_call_record(call_record: Dict[str, Any]) -> Dict[str, Any]:
    prompt_text = extract_prompt_text_from_call_record(call_record)
    output_text = extract_output_text_from_call_record(call_record)
    return {
        "call_id": call_record.get("call_id"),
        "method": call_record.get("method"),
        "prompt_excerpt": excerpt(prompt_text, 800),
        "output_excerpt": excerpt(output_text, 800),
        "prompt_function_names": parse_generated_function_names(prompt_text),
        "output_function_names": parse_generated_function_names(output_text),
        "prompt_len": len(prompt_text),
        "output_len": len(output_text),
    }

def has_exact_signature(prompt_or_output: str, signature: str) -> bool:
    return loose_contains(prompt_or_output, signature)


def ir_format_diagnosis(recorder: Recorder, prompt_text: str) -> Dict[str, Any]:
    raw_ir = recorder.ir_candidates[-1]["value"] if recorder.ir_candidates else None
    summary = recorder.ir_candidates[-1]["summary"] if recorder.ir_candidates else {"type": "None"}

    serializable = True
    serialize_error = None
    try:
        json.dumps(to_jsonable(raw_ir), ensure_ascii=False)
    except Exception as e:
        serializable = False
        serialize_error = repr(e)

    ir_text = json.dumps(to_jsonable(raw_ir), ensure_ascii=False) if raw_ir is not None else ""
    is_empty_like = (
        raw_ir is None
        or raw_ir == {}
        or raw_ir == []
        or (isinstance(raw_ir, str) and not raw_ir.strip())
        or len(ir_text) < 20
    )

    prompt_mentions_ir = loose_contains(prompt_text, "ir") or loose_contains(prompt_text, ir_text[:120])

    likely_format_issue = (
        (not serializable)
        or (not prompt_mentions_ir and raw_ir is not None and not is_empty_like)
        or (summary.get("type") not in {"dict", "list", "str"} and raw_ir is not None)
    )

    return {
        "raw_ir_summary": summary,
        "raw_ir_present": raw_ir is not None,
        "raw_ir_serializable": serializable,
        "serialize_error": serialize_error,
        "raw_ir_empty_like": is_empty_like,
        "prompt_mentions_ir": prompt_mentions_ir,
        "likely_format_issue": likely_format_issue,
        "evidence_excerpt": excerpt(ir_text, 500),
    }


def strict_rule_diagnosis(recorder: Recorder, record: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    ir_text = ""
    if recorder.ir_candidates:
        ir_text = json.dumps(to_jsonable(recorder.ir_candidates[-1]["value"]), ensure_ascii=False)

    constraint_text = ""
    if recorder.constraint_candidates:
        constraint_text = json.dumps(to_jsonable(recorder.constraint_candidates[-1]["value"]), ensure_ascii=False)

    task_rich = len(record.get("file_content", "") or "") > 500 and len(record.get("code", "") or "") > 20
    ir_too_small = len(ir_text) < 80
    constraints_too_small = len(constraint_text) < 80
    prompt_small = len(prompt_text) < 400

    likely_over_strict = task_rich and ir_too_small and prompt_small

    return {
        "task_context_rich": task_rich,
        "ir_text_len": len(ir_text),
        "constraint_text_len": len(constraint_text),
        "prompt_len": len(prompt_text),
        "likely_over_strict_or_over_filtered": likely_over_strict or (task_rich and ir_too_small and constraints_too_small),
        "evidence": {
            "ir_excerpt": excerpt(ir_text, 300),
            "constraint_excerpt": excerpt(constraint_text, 300),
        }
    }


def info_completeness_diagnosis(record: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    snippets = extract_target_snippets(record)
    oracle_tokens = token_set_from_context(record.get("oracle_context", ""))
    all_context_tokens = token_set_from_context(record.get("all_context", ""))

    checks = {
        "name_in_prompt": loose_contains(prompt_text, snippets["name"]),
        "signature_in_prompt": loose_contains(prompt_text, snippets["signature"]),
        "docstring_in_prompt": loose_contains(prompt_text, snippets["doc_first_line"]),
        "file_path_in_prompt": loose_contains(prompt_text, snippets["file_path"]),
        "human_label_in_prompt": loose_contains(prompt_text, snippets["human_label"]),
        "target_code_block_in_prompt": loose_contains(prompt_text, snippets["target_code"][:160]),
        "focus_block_in_prompt": loose_contains(prompt_text, snippets["target_focus_block"][:160]),
    }

    oracle_hits = [t for t in oracle_tokens if loose_contains(prompt_text, t)]
    all_ctx_hits = [t for t in all_context_tokens if loose_contains(prompt_text, t)]

    available_fields = 7
    matched_fields = sum(int(v) for v in checks.values())
    completeness_ratio = round(matched_fields / available_fields, 3)

    return {
        "field_checks": checks,
        "oracle_tokens_total": oracle_tokens[:20],
        "oracle_tokens_hit": oracle_hits[:20],
        "all_context_tokens_total": all_context_tokens[:20],
        "all_context_tokens_hit": all_ctx_hits[:20],
        "completeness_ratio": completeness_ratio,
        "likely_missing_info_to_model": completeness_ratio < 0.55,
    }


def template_diagnosis(record: Dict[str, Any], prompt_text: str) -> Dict[str, Any]:
    signature = extract_signature_from_code(record.get("code", ""))
    target_name = record.get("name", "")

    controls = {
        "mentions_exact_function_name": loose_contains(prompt_text, target_name),
        "mentions_exact_signature": loose_contains(prompt_text, signature),
        "mentions_only_code": any(loose_contains(prompt_text, x) for x in [
            "only output code", "output only code", "do not explain", "no explanation",
            "single function", "exact function", "no markdown", "no code fence"
        ]),
        "mentions_replace_not_solve": any(loose_contains(prompt_text, x) for x in [
            "replace", "target function", "must keep function name", "must keep signature"
        ]),
        "mentions_project_context": loose_contains(prompt_text, record.get("project", "")) or loose_contains(prompt_text, record.get("file_path", "")),
    }

    weak_template = sum(int(v) for v in controls.values()) <= 2

    return {
        "controls": controls,
        "likely_template_is_weak": weak_template,
        "prompt_len": len(prompt_text),
        "prompt_excerpt": excerpt(prompt_text, 1000),
    }


def constraint_diagnosis(record: Dict[str, Any], prompt_text: str, output_text: str) -> Dict[str, Any]:
    oracle_tokens = token_set_from_context(record.get("oracle_context", ""))
    required_tokens = [t for t in oracle_tokens if t in {"Time", "FixedOffset", "localize", "map", "divmod"}]

    required_hits = [t for t in required_tokens if loose_contains(prompt_text, t)]
    function_names = parse_generated_function_names(output_text)

    output_checks = {
        "single_function_output": len(function_names) == 1,
        "exact_function_name_output": record.get("name", "") in function_names,
        "has_markdown_fence": "```" in (output_text or ""),
        "has_explanatory_text": any(x in (output_text or "").lower() for x in [
            "here is", "explanation", "this function", "模型", "答案", "下面"
        ]),
        "mentions_required_symbols_in_prompt": len(required_hits) >= 2,
    }

    missing_constraints = (
        not output_checks["exact_function_name_output"]
        or output_checks["has_markdown_fence"]
        or output_checks["has_explanatory_text"]
        or not output_checks["mentions_required_symbols_in_prompt"]
    )

    return {
        "required_tokens": required_tokens,
        "required_hits_in_prompt": required_hits,
        "output_checks": output_checks,
        "generated_function_names": function_names,
        "likely_missing_intermediate_constraints": missing_constraints,
    }


def build_hypothesis_report(record: Dict[str, Any], recorder: Recorder) -> Dict[str, Any]:
    selected_call = recorder.best_llm_call()
    prompt_text = extract_prompt_text_from_call_record(selected_call) if selected_call else recorder.best_prompt()
    output_text = extract_output_text_from_call_record(selected_call) if selected_call else recorder.best_output()



    h1 = ir_format_diagnosis(recorder, prompt_text)
    h2 = strict_rule_diagnosis(recorder, record, prompt_text)
    h3 = info_completeness_diagnosis(record, prompt_text)
    h4 = template_diagnosis(record, prompt_text)
    h5 = constraint_diagnosis(record, prompt_text, output_text)

    return {
        "task_id": record.get("_id") or record.get("task_id"),
        "target_name": record.get("name"),
        "target_signature": extract_signature_from_code(record.get("code", "")),
        "best_prompt_excerpt": excerpt(prompt_text, 2000),
        "best_output_excerpt": excerpt(output_text, 2000),
        "selected_llm_call": summarize_call_record(selected_call) if selected_call else {},
        "llm_call_count": len(recorder.llm_call_records),
        "hypotheses": {
            "H1_ir_format_issue": h1,
            "H2_ir_rules_too_strict": h2,
            "H3_incomplete_codeval_info_to_model": h3,
            "H4_weak_prompt_template_or_state_contract": h4,
            "H5_missing_generation_constraints": h5,
        }
    }



def make_markdown_summary(report: Dict[str, Any]) -> str:
    h = report["hypotheses"]
    lines = []
    lines.append(f"# Diagnostic Report for task `{report['task_id']}`")
    lines.append("")
    lines.append(f"- Target function: `{report['target_name']}`")
    lines.append(f"- Target signature: `{report['target_signature']}`")
    lines.append("")
    lines.append("## Quick verdict")
    lines.append("")
    lines.append(f"- H1 IR 格式问题倾向：`{h['H1_ir_format_issue']['likely_format_issue']}`")
    lines.append(f"- H2 IR 规则过严倾向：`{h['H2_ir_rules_too_strict']['likely_over_strict_or_over_filtered']}`")
    lines.append(f"- H3 信息未完整传给模型倾向：`{h['H3_incomplete_codeval_info_to_model']['likely_missing_info_to_model']}`")
    lines.append(f"- H4 Prompt template / state contract 弱倾向：`{h['H4_weak_prompt_template_or_state_contract']['likely_template_is_weak']}`")
    lines.append(f"- H5 中间约束不足倾向：`{h['H5_missing_generation_constraints']['likely_missing_intermediate_constraints']}`")
    lines.append("")
    lines.append("## Evidence")
    lines.append("")
    lines.append("### H1 IR format")
    lines.append(f"- raw_ir_present: `{h['H1_ir_format_issue']['raw_ir_present']}`")
    lines.append(f"- raw_ir_serializable: `{h['H1_ir_format_issue']['raw_ir_serializable']}`")
    lines.append(f"- raw_ir_empty_like: `{h['H1_ir_format_issue']['raw_ir_empty_like']}`")
    lines.append(f"- prompt_mentions_ir: `{h['H1_ir_format_issue']['prompt_mentions_ir']}`")
    lines.append("")
    lines.append("### H2 strict rules")
    lines.append(f"- task_context_rich: `{h['H2_ir_rules_too_strict']['task_context_rich']}`")
    lines.append(f"- ir_text_len: `{h['H2_ir_rules_too_strict']['ir_text_len']}`")
    lines.append(f"- constraint_text_len: `{h['H2_ir_rules_too_strict']['constraint_text_len']}`")
    lines.append(f"- prompt_len: `{h['H2_ir_rules_too_strict']['prompt_len']}`")
    lines.append("")
    lines.append("### H3 completeness")
    lines.append(f"- completeness_ratio: `{h['H3_incomplete_codeval_info_to_model']['completeness_ratio']}`")
    lines.append(f"- field_checks: `{json.dumps(h['H3_incomplete_codeval_info_to_model']['field_checks'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("### H4 template")
    lines.append(f"- controls: `{json.dumps(h['H4_weak_prompt_template_or_state_contract']['controls'], ensure_ascii=False)}`")
    lines.append("")
    lines.append("### H5 constraints")
    lines.append(f"- output_checks: `{json.dumps(h['H5_missing_generation_constraints']['output_checks'], ensure_ascii=False)}`")
    lines.append(f"- generated_function_names: `{h['H5_missing_generation_constraints']['generated_function_names']}`")
    lines.append("")
    lines.append("## Prompt excerpt")
    lines.append("")
    lines.append("```text")
    lines.append(report["best_prompt_excerpt"])
    lines.append("```")
    lines.append("")
    lines.append("## Output excerpt")
    lines.append("")
    lines.append("```text")
    lines.append(report["best_output_excerpt"])
    lines.append("```")
    return "\n".join(lines)

def inspect_llm_module(module: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "module": getattr(module, "__name__", ""),
        "classes": {},
        "functions": [],
    }
    if module is None:
        return info

    for name in dir(module):
        if name.startswith("__"):
            continue
        obj = getattr(module, name, None)

        if inspect.isclass(obj):
            methods = []
            for attr in dir(obj):
                if attr.startswith("__"):
                    continue
                try:
                    val = getattr(obj, attr)
                    if callable(val):
                        methods.append(attr)
                except Exception:
                    continue
            info["classes"][name] = sorted(methods)

        elif callable(obj):
            info["functions"].append(name)

    info["functions"] = sorted(info["functions"])
    return info


# ============================================================
# 主流程
# ============================================================

def maybe_add_src_to_syspath(repo_root: Path, set_pythonpath_src: bool) -> None:
    if not set_pythonpath_src:
        return
    src = repo_root / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def try_import_optional(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--json-path", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--project-root", default="")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/diagnostics")
    parser.add_argument("--set-pythonpath-src", action="store_true")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    maybe_add_src_to_syspath(repo_root, args.set_pythonpath_src)

    output_dir = Path(args.output_dir).resolve() / f"{args.task_id}_{now_ts()}"
    ensure_dir(output_dir)

    recorder = Recorder()
    patches: List[Tuple[Any, str, Any]] = []

    try:
        record = load_record_by_task_id(Path(args.json_path), args.task_id)
        config = load_config(Path(args.config))
        task = build_task_object_from_record(record)

        write_json(output_dir / "task_record.json", record)
        write_json(output_dir / "task_object_preview.json", task)
        write_json(output_dir / "config_preview.json", config)

        # 尝试 patch LLMClient
        llm_mod = try_import_optional("beacon_system.llm.client")
        if llm_mod is not None:
            llm_module_info = inspect_llm_module(llm_mod)
            write_json(output_dir / "llm_module_inspection.json", llm_module_info)
            print("[DEBUG] llm module inspection written to:", output_dir / "llm_module_inspection.json")

            # 优先 patch 模块里所有名字像 client / llm 的类
            for cls_name, method_names in llm_module_info.get("classes", {}).items():
                if "client" in cls_name.lower() or "llm" in cls_name.lower():
                    cls_obj = getattr(llm_mod, cls_name, None)
                    if cls_obj is not None:
                        patches += patch_class_methods(
                            cls_obj,
                            recorder,
                            [
                                "generate", "complete", "chat", "invoke", "__call__",
                                "run", "create", "call", "request", "send",
                                "_generate", "_complete", "_chat", "_invoke",
                            ],
                            tag=cls_name
                        )

            # 再 patch 模块级函数
            for fn_name in llm_module_info.get("functions", []):
                if any(k in fn_name.lower() for k in [
                    "generate", "complete", "chat", "invoke",
                    "call", "request", "create", "send"
                ]):
                    fn_obj = getattr(llm_mod, fn_name, None)
                    if callable(fn_obj):
                        def make_module_llm_wrapper(func_name: str, func: Any):
                            def wrapper(*args, **kwargs):
                                call_id = recorder.next_llm_call_id()
                                call_record = {
                                    "call_id": call_id,
                                    "method": f"llm_module.{func_name}",
                                    "args_summary": [object_summary(a) for a in args[:3]],
                                    "kwargs_summary": {k: object_summary(v) for k, v in list(kwargs.items())[:10]},
                                    "prompt_strings": [],
                                    "output_strings": [],
                                    "result_summary": None,
                                    "error": None,
                                }

                                try:
                                    for s in find_strings({"args": args, "kwargs": kwargs}, min_len=40, limit=50):
                                        call_record["prompt_strings"].append(s)
                                        recorder.add_prompt_candidate(s, f"llm_module.{func_name}.args.call_{call_id}")
                                except Exception as e:
                                    call_record["error"] = f"[module llm pre {func_name}] {e}"

                                result = func(*args, **kwargs)

                                try:
                                    call_record["result_summary"] = object_summary(result)
                                    for s in find_strings(result, min_len=40, limit=50):
                                        call_record["output_strings"].append(s)
                                        if looks_like_code(s):
                                            recorder.add_output_candidate(s,
                                                                          f"llm_module.{func_name}.return.call_{call_id}")
                                        else:
                                            recorder.add_prompt_candidate(s,
                                                                          f"llm_module.{func_name}.return.call_{call_id}")
                                except Exception as e:
                                    call_record["error"] = f"[module llm post {func_name}] {e}"

                                recorder.add_llm_call_record(call_record)
                                return result

                            return wrapper

                        patches.append((llm_mod, fn_name, fn_obj))
                        setattr(llm_mod, fn_name, make_module_llm_wrapper(fn_name, fn_obj))

        # 尝试 patch generator / pipeline / logic 相关模块
        for mod_name in [
            "beacon_system.agents.generator",
            "beacon_system.pipeline",
            "beacon_system.logic.engine",
            "beacon_system.logic.rules_local",
            "beacon_system.logic.rules_global",
        ]:
            mod = try_import_optional(mod_name)
            if mod is not None:
                patches += patch_module_functions(mod, recorder)

        # 正式调用现有 pipeline
        pipeline_mod = importlib.import_module("beacon_system.pipeline")
        run_pipeline = getattr(pipeline_mod, "run_pipeline")

        recorder.log(
            "run.start",
            task_summary=object_summary(task),
            task_id=args.task_id,
            output_dir=str(output_dir),
        )

        result = run_pipeline(task=task, config=config)

        recorder.result_objects.append({
            "summary": object_summary(result),
            "value": to_jsonable(result),
        })
        write_json(output_dir / "pipeline_result.json", result)

        # 尝试从 result 中再抓一些字符串
        for s in find_strings(result, min_len=60, limit=30):
            if looks_like_code(s):
                recorder.add_output_candidate(s, "pipeline_result")
            else:
                recorder.add_prompt_candidate(s, "pipeline_result")

        report = build_hypothesis_report(record, recorder)
        write_json(output_dir / "diagnostic_report.json", report)
        write_text(output_dir / "diagnostic_report.md", make_markdown_summary(report))

        # 原始记录
        write_json(output_dir / "events.json", recorder.events)
        write_json(output_dir / "ir_candidates.json", recorder.ir_candidates)
        write_json(output_dir / "constraint_candidates.json", recorder.constraint_candidates)
        write_json(output_dir / "llm_call_records.json", recorder.llm_call_records)
        write_json(
            output_dir / "llm_call_summaries.json",
            [summarize_call_record(x) for x in recorder.llm_call_records]
        )

        selected_call = recorder.best_llm_call()
        selected_prompt = extract_prompt_text_from_call_record(
            selected_call) if selected_call else recorder.best_prompt()
        selected_output = extract_output_text_from_call_record(
            selected_call) if selected_call else recorder.best_output()

        write_text(output_dir / "best_prompt.txt", selected_prompt)
        write_text(output_dir / "best_output.txt", selected_output)
        write_json(output_dir / "errors.json", recorder.errors)

        print("=" * 100)
        print("Diagnostic completed")
        print(f"Output dir: {output_dir}")
        print("=" * 100)
        print(f"LLM call count: {len(recorder.llm_call_records)}")
        if recorder.llm_call_records:
            print("LLM call summaries:")
            print(
                json.dumps([summarize_call_record(x) for x in recorder.llm_call_records], ensure_ascii=False, indent=2))
        print("=" * 100)
        print(json.dumps(report["hypotheses"], ensure_ascii=False, indent=2))

    except Exception as e:
        tb = traceback.format_exc()
        write_text(output_dir / "fatal_error.txt", tb)
        print("=" * 100)
        print("[FATAL] diagnostic failed")
        print(repr(e))
        print(tb)
        print("=" * 100)
        raise
    finally:
        restore_patches(patches)

    # 兜底 patch OpenAI SDK 常见入口
    try:
        import openai

        # 新版 SDK 常见路径：resources.chat.completions.Completions.create
        try:
            from openai.resources.chat.completions.completions import Completions
            if hasattr(Completions, "create"):
                original_create = Completions.create

                def openai_chat_create_wrapper(*args, **kwargs):
                    call_id = recorder.next_llm_call_id()
                    call_record = {
                        "call_id": call_id,
                        "method": "openai.chat.completions.create",
                        "args_summary": [object_summary(a) for a in args[:3]],
                        "kwargs_summary": {k: object_summary(v) for k, v in list(kwargs.items())[:10]},
                        "prompt_strings": [],
                        "output_strings": [],
                        "result_summary": None,
                        "error": None,
                    }

                    try:
                        for s in find_strings({"args": args, "kwargs": kwargs}, min_len=20, limit=100):
                            call_record["prompt_strings"].append(s)
                            recorder.add_prompt_candidate(s, f"openai.chat.create.args.call_{call_id}")
                    except Exception as e:
                        call_record["error"] = f"[openai create pre] {e}"

                    result = original_create(*args, **kwargs)

                    try:
                        call_record["result_summary"] = object_summary(result)
                        for s in find_strings(result, min_len=20, limit=100):
                            call_record["output_strings"].append(s)
                            if looks_like_code(s):
                                recorder.add_output_candidate(s, f"openai.chat.create.return.call_{call_id}")
                            else:
                                recorder.add_prompt_candidate(s, f"openai.chat.create.return.call_{call_id}")
                    except Exception as e:
                        call_record["error"] = f"[openai create post] {e}"

                    recorder.add_llm_call_record(call_record)
                    return result

                patches.append((Completions, "create", original_create))
                Completions.create = openai_chat_create_wrapper
        except Exception:
            pass

        # responses API 兜底
        try:
            from openai.resources.responses.responses import Responses
            if hasattr(Responses, "create"):
                original_resp_create = Responses.create

                def openai_responses_create_wrapper(*args, **kwargs):
                    call_id = recorder.next_llm_call_id()
                    call_record = {
                        "call_id": call_id,
                        "method": "openai.responses.create",
                        "args_summary": [object_summary(a) for a in args[:3]],
                        "kwargs_summary": {k: object_summary(v) for k, v in list(kwargs.items())[:10]},
                        "prompt_strings": [],
                        "output_strings": [],
                        "result_summary": None,
                        "error": None,
                    }

                    try:
                        for s in find_strings({"args": args, "kwargs": kwargs}, min_len=20, limit=100):
                            call_record["prompt_strings"].append(s)
                            recorder.add_prompt_candidate(s, f"openai.responses.create.args.call_{call_id}")
                    except Exception as e:
                        call_record["error"] = f"[openai responses pre] {e}"

                    result = original_resp_create(*args, **kwargs)

                    try:
                        call_record["result_summary"] = object_summary(result)
                        for s in find_strings(result, min_len=20, limit=100):
                            call_record["output_strings"].append(s)
                            if looks_like_code(s):
                                recorder.add_output_candidate(s, f"openai.responses.create.return.call_{call_id}")
                            else:
                                recorder.add_prompt_candidate(s, f"openai.responses.create.return.call_{call_id}")
                    except Exception as e:
                        call_record["error"] = f"[openai responses post] {e}"

                    recorder.add_llm_call_record(call_record)
                    return result

                patches.append((Responses, "create", original_resp_create))
                Responses.create = openai_responses_create_wrapper
        except Exception:
            pass

    except Exception:
        pass


if __name__ == "__main__":
    main()