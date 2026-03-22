# baseline_codegen/pipeline_baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from generator_baseline import BaselineGenerator
from schema import CodeEvalTask, GenerationResult


@dataclass
class BaselinePipeline:
    """
    Minimal single-task pipeline.

    Flow:
    raw task -> CodeEvalTask -> generator -> GenerationResult
    """
    generator: BaselineGenerator

    def run_task(self, raw_task: Dict[str, Any]) -> GenerationResult:
        """
        Execute one raw benchmark record and return the final structured result.
        """
        task = self.build_task(raw_task)
        return self.run_codegen(task)

    def build_task(self, raw_task: Dict[str, Any]) -> CodeEvalTask:
        """
        Convert one raw benchmark record into a normalized task object.
        """
        task = CodeEvalTask.from_raw(raw_task)
        if not task.task_id:
            raise ValueError("Task id is empty.")
        if not task.prompt.strip():
            raise ValueError(f"Task prompt is empty for task={task.task_id}")
        return task

    def run_codegen(self, task: CodeEvalTask) -> GenerationResult:
        """
        Generate final code result for one normalized task.
        """
        return self.generator.generate_result(task)

    def run_task_to_dict(self, raw_task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convenience method for direct json serialization.
        """
        result = self.run_task(raw_task)
        return result.to_dict()