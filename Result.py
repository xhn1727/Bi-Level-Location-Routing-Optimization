from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional

import numpy as np
import dataset_types as dt


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()

class ResultLogger:
    def __init__(self, dataset_name: str, algorithm_combo: str, mode: str, output_root: str = "results",):

        self.dataset_name = self._sanitize_name(dataset_name)
        self.algorithm_combo = self._sanitize_name(algorithm_combo)
        self.mode = self._sanitize_name(mode)
        self.output_root = output_root
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        self.dataset_dir = os.path.join(self.output_root, self.dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)

        base_name = f"{self.timestamp}_{self.algorithm_combo}_{self.mode}"
        self.log_path = os.path.join(self.dataset_dir, f"{base_name}.txt")
        self.best_path = os.path.join(self.dataset_dir, f"{base_name}_best.txt")

    @staticmethod
    def _sanitize_name(name: str) -> str:
        safe = str(name).strip().replace(" ", "_")
        for ch in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
            safe = safe.replace(ch, "_")
        return safe

    @staticmethod
    def _format_array(arr: np.ndarray) -> str:
        return np.array2string(arr, precision=4, suppress_small=True)

    @staticmethod
    def _format_customer_sets(customer_sets: List[dt.Assigned_Customer_Sets]) -> List[str]:
        lines: List[str] = []
        for cs in customer_sets:
            if not cs.assigned_customer_num:
                continue
            lines.append(
                f"  - warehouse_num={cs.warehouse_num}, assigned_customer_num={cs.assigned_customer_num}"
            )
        if not lines:
            lines.append("  - 无客户分配信息")
        return lines

    @staticmethod
    def _format_plan(plan: Optional[dt.Plan], title: str) -> List[str]:
        lines: List[str] = [title]
        if plan is None:
            lines.append("  None")
            return lines

        lines.extend([
            f"  feasible = {plan.feasible}",
            f"  plan_total_cost = {plan.plan_total_cost}",
            f"  plan_distance = {plan.plan_distance}",
            f"  plan_window_penalty = {plan.plan_window_penalty}",
            f"  plan_decay_cost = {plan.plan_decay_cost}",
            f"  plan_transportation_cost = {plan.plan_transportation_cost}",
            f"  plan_vehicle_used = {plan.plan_vehicle_used}",
        ])

        if not plan.plan_routes:
            lines.append("  plan_routes = []")
            return lines

        lines.append("  plan_routes:")
        for wr in plan.plan_routes:
            lines.extend([
                f"    warehouse_num = {wr.warehouse_num}",
                f"    feasible = {wr.feasible}",
                f"    served_customers_num = {wr.served_customers_num}",
                f"    warehouse_routes_distance = {wr.warehouse_routes_distance}",
                f"    warehouse_window_penalty = {wr.warehouse_window_penalty}",
                f"    warehouse_decay_cost = {wr.warehouse_decay_cost}",
                f"    warehouse_transportation_cost = {wr.warehouse_transportation_cost}",
                f"    vehicle_used_num = {wr.vehicle_used_num}",
            ])
            for route in wr.routes:
                lines.append(
                    f"      route num_path={route.num_path}, distance={route.distance}, window_penalty={route.window_penalty}"
                )
        return lines

    @staticmethod
    def _format_individual(ind: Optional[dt.Individual], title: str) -> List[str]:
        lines: List[str] = [title]
        if ind is None:
            lines.append("  None")
            return lines

        lines.extend([
            f"  y = {ind.y}",
            f"  w_if = {ResultLogger._format_array(ind.w_if)}",
            f"  ind_build_operate_cost = {ind.ind_build_operate_cost}",
            f"  ind_transportation_cost = {ind.ind_transportation_cost}",
            f"  ind_decay_cost = {ind.ind_decay_cost}",
            f"  ind_excess_penalty = {ind.ind_excess_penalty}",
            f"  total_upper_cost = {ind.total_upper_cost}",
            f"  total_cost = {ind.total_cost}",
            "  customer_sets:",
        ])
        lines.extend(ResultLogger._format_customer_sets(ind.customer_sets))
        lines.extend(ResultLogger._format_plan(ind.best_lower_plan, "  best_lower_plan:"))
        return lines

    def _write_lines(self, path: str, lines: List[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            f.write("\n")

    @contextmanager
    def capture_prints(self) -> Iterator["ResultLogger"]:
        original_stdout = sys.stdout
        with open(self.log_path, "a", encoding="utf-8") as log_file:
            sys.stdout = Tee(original_stdout, log_file)
            try:
                yield self
            finally:
                sys.stdout = original_stdout

    def save_nested_result(self, best_individual: Optional[dt.Individual], history: List[dt.Individual]) -> None:
        lines: List[str] = [
            f"dataset_name = {self.dataset_name}",
            f"algorithm_combo = {self.algorithm_combo}",
            f"mode = {self.mode}",
            f"timestamp = {self.timestamp}",
            f"history_length = {len(history)}",
            "",
        ]
        lines.extend(self._format_individual(best_individual, "best_individual:"))
        self._write_lines(self.best_path, lines)

    def save_single_result(
        self,
        best_individual: Optional[dt.Individual],
        upper_history: List[dt.Individual],
        best_plan: Optional[dt.Plan],
        lower_history: List[dt.Iteration],
    ) -> None:
        lines: List[str] = [
            f"dataset_name = {self.dataset_name}",
            f"algorithm_combo = {self.algorithm_combo}",
            f"mode = {self.mode}",
            f"timestamp = {self.timestamp}",
            f"upper_history_length = {len(upper_history)}",
            f"lower_history_length = {len(lower_history)}",
            "",
        ]
        lines.extend(self._format_individual(best_individual, "best_individual:"))
        lines.append("")
        lines.extend(self._format_plan(best_plan, "best_lower_plan:"))
        self._write_lines(self.best_path, lines)