import os
import re
import glob
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

RESULTS_DIR = "供需比变化实验/供需比变化实验结果/results"
OUTPUT_FILE = "best_plan_detail_3_6_25.csv"
OUTPUT_XLSX = "best_plan_detail_3_6_25.xlsx"

# txt 中记录的下层成本为“日维度”结果；为与上层月度成本统一口径，
# 在汇总时统一折算为周期成本。若你的实验周期不是30天，可在此修改。
LOWER_COST_PERIOD_DAYS = 30

# 原始实验数据集搜索目录（会递归搜索）
DATASET_SEARCH_DIRS = [
    "供需比变化实验/供需比变化实验数据集",
    "./generated_datasets_紧时间窗",
    "./generated_datasets_timewindow_sensitivity",
    "./generated_datasets_demand_sensitivity",
    ".",
]


_DATASET_CACHE: Dict[str, Dict[str, pd.DataFrame]] = {}
_DATASET_PATH_CACHE: Dict[str, Optional[str]] = {}


def safe_float(x: str):
    try:
        return float(x)
    except Exception:
        return math.nan


def extract_trailing_int(value, default=np.nan):
    if pd.isna(value):
        return default
    s = str(value).strip()
    m = re.search(r"(\d+)$", s)
    if m:
        return int(m.group(1))
    return default

def summarize_numeric_list(values: List[float], prefix: str) -> Dict[str, float]:
    """
    对数值列表做聚合统计，统一返回均值/最小值/最大值/标准差。
    若列表为空，则全部返回 NaN。
    """
    arr = np.array([v for v in values if not pd.isna(v)], dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_mean": math.nan,
            f"{prefix}_min": math.nan,
            f"{prefix}_max": math.nan,
            f"{prefix}_std": math.nan,
        }
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_std": float(np.std(arr)),
    }


# =========================
# 原始实验数据集读取与距离计算
# =========================
def find_dataset_file(dataset_name: str) -> Optional[str]:
    if not dataset_name:
        return None
    if dataset_name in _DATASET_PATH_CACHE:
        return _DATASET_PATH_CACHE[dataset_name]

    target = f"{dataset_name}.xlsx"

    for base_dir in DATASET_SEARCH_DIRS:
        base_path = Path(base_dir)
        if not base_path.exists():
            continue

        exact_path = base_path / target
        if exact_path.exists():
            _DATASET_PATH_CACHE[dataset_name] = str(exact_path)
            return str(exact_path)

        matches = list(base_path.rglob(target))
        if matches:
            _DATASET_PATH_CACHE[dataset_name] = str(matches[0])
            return str(matches[0])

    _DATASET_PATH_CACHE[dataset_name] = None
    return None


def load_original_dataset(dataset_name: str) -> Optional[Dict[str, pd.DataFrame]]:
    if not dataset_name:
        return None
    if dataset_name in _DATASET_CACHE:
        return _DATASET_CACHE[dataset_name]

    path = find_dataset_file(dataset_name)
    if path is None:
        return None

    try:
        cw_df = pd.read_excel(path, sheet_name="center_warehouse")
        iw_df = pd.read_excel(path, sheet_name="infront_warehouse")
        cm_df = pd.read_excel(path, sheet_name="customer")
        _DATASET_CACHE[dataset_name] = {
            "path": path,
            "cw_df": cw_df,
            "iw_df": iw_df,
            "cm_df": cm_df,
        }
        return _DATASET_CACHE[dataset_name]
    except Exception:
        _DATASET_CACHE[dataset_name] = None
        return None


def haversine_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
    R = 6371.0
    lon1, lat1 = coord1[0], coord1[1]
    lon2, lat2 = coord2[0], coord2[1]
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(R * c)


# =========================
# txt 基础信息解析
# =========================
def extract_basic_info(text: str) -> dict:
    row = {
        "dataset_name": None,
        "mode": None,
        "algorithm_combo": None,
        "timestamp": None,
    }

    patterns = {
        "dataset_name": r"dataset_name\s*=\s*(.+)",
        "mode": r"mode\s*=\s*(.+)",
        "algorithm_combo": r"algorithm_combo\s*=\s*(.+)",
        "timestamp": r"timestamp\s*=\s*(.+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        if m:
            row[key] = m.group(1).strip()

    return row


def extract_scale_info(dataset_name: str) -> dict:
    row = {
        "num_cw": math.nan,
        "num_iw": math.nan,
        "num_cm": math.nan,
        "dataset_index": math.nan,
        "is_extra_dataset": False,
    }

    if not dataset_name:
        return row

    m = re.match(r"^CW_(\d+)_IW_(\d+)_CM_(\d+)_(\d+)(?:_(补充))?(?:_.+)?$", dataset_name)
    if m:
        row["num_cw"] = int(m.group(1))
        row["num_iw"] = int(m.group(2))
        row["num_cm"] = int(m.group(3))
        row["dataset_index"] = int(m.group(4))
        row["is_extra_dataset"] = (m.group(5) == "补充")

    return row


 
def extract_upper_info(text: str) -> dict:
    row = {
        "y_raw": None,
        "warehouse_open_count": math.nan,
        "upper_total_cost": math.nan,

        # 上层成本分项 / 聚合项
        "upper_fixed_cost": math.nan,
        "upper_operation_cost": math.nan,
        "upper_warehouse_cost": math.nan,

        "upper_transportation_variable_cost": math.nan,
        "upper_transportation_fixed_cost": math.nan,
        "upper_transportation_cost": math.nan,

        "upper_decay_cost": math.nan,
        "upper_excess_cost": math.nan,
    }

    m = re.search(r"best_individual:.*?y\s*=\s*\[([^\]]+)\]", text, re.S)
    if m:
        y_raw = m.group(1).strip()
        row["y_raw"] = y_raw
        try:
            y_vals = [int(v.strip()) for v in y_raw.split(",") if v.strip() != ""]
            row["warehouse_open_count"] = sum(y_vals)
        except Exception:
            row["warehouse_open_count"] = math.nan

    # 上层总成本
    m = re.search(r"best_individual:.*?total_upper_cost\s*=\s*([0-9eE\.\+\-]+)", text, re.S)
    if m:
        row["upper_total_cost"] = safe_float(m.group(1))

    # 仅在 best_individual 区域内优先查找，避免与下层同名字段混淆
    best_individual_match = re.search(r"best_individual:(.*?)(?:\nbest_lower_plan:|$)", text, re.S)
    search_text = best_individual_match.group(1) if best_individual_match else text

    upper_patterns = {
        "upper_fixed_cost": [
            r"fixed_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"warehouse_fixed_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_fixed_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        "upper_operation_cost": [
            r"operation_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"warehouse_operation_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_operation_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        # txt 中实际输出的是合并后的建仓+运营成本
        "upper_warehouse_cost": [
            r"ind_build_operate_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"build_operate_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_warehouse_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        "upper_transportation_variable_cost": [
            r"ind_transportation_variable_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"transportation_variable_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_transportation_variable_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        # txt 中实际输出的是合并后的上层运输成本
        "upper_transportation_cost": [
            r"ind_transportation_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"transportation_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_transportation_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        "upper_transportation_fixed_cost": [
            r"ind_transportation_fixed_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"transportation_fixed_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_transportation_fixed_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        # txt 中实际输出的是合并后的上层腐损成本
        "upper_decay_cost": [
            r"ind_decay_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_decay_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"decay_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
        # txt 中实际输出的是超额惩罚
        "upper_excess_cost": [
            r"ind_excess_penalty\s*=\s*([0-9eE\.\+\-]+)",
            r"upper_excess_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"excess_cost\s*=\s*([0-9eE\.\+\-]+)",
            r"ind_excess_cost\s*=\s*([0-9eE\.\+\-]+)",
        ],
    }

    for key, pattern_list in upper_patterns.items():
        for pattern in pattern_list:
            m = re.search(pattern, search_text)
            if m:
                row[key] = safe_float(m.group(1))
                break

    return row

def compute_active_warehouse_structure_metrics(y_raw: Optional[str]) -> dict:
    """
    基于上层 y 向量提取启用前置仓结构字段。
    """
    row = {
        "active_warehouse_ids": None,
        "active_warehouse_count_from_y": math.nan,
        "active_warehouse_id_min": math.nan,
        "active_warehouse_id_max": math.nan,
        "active_warehouse_id_span": math.nan,
    }

    if not y_raw:
        return row

    try:
        y_vals = [int(v.strip()) for v in str(y_raw).split(",") if v.strip() != ""]
        active_ids = [idx + 1 for idx, val in enumerate(y_vals) if val == 1]
        row["active_warehouse_ids"] = ",".join(map(str, active_ids)) if active_ids else ""
        row["active_warehouse_count_from_y"] = len(active_ids)
        if active_ids:
            row["active_warehouse_id_min"] = min(active_ids)
            row["active_warehouse_id_max"] = max(active_ids)
            row["active_warehouse_id_span"] = max(active_ids) - min(active_ids)
    except Exception:
        pass

    return row


def compute_upper_cost_breakdown_metrics(row: dict) -> dict:
    """
    基于已解析的上层分项成本，补充上层聚合成本及其占比指标。
    若某些分项在 txt 中不存在，则相应指标保留 NaN。
    """
    fixed_cost = row.get("upper_fixed_cost", math.nan)
    operation_cost = row.get("upper_operation_cost", math.nan)
    trans_var_cost = row.get("upper_transportation_variable_cost", math.nan)
    trans_fix_cost = row.get("upper_transportation_fixed_cost", math.nan)
    decay_cost = row.get("upper_decay_cost", math.nan)
    excess_cost = row.get("upper_excess_cost", math.nan)
    upper_total_cost = row.get("upper_total_cost", math.nan)
    total_cost = row.get("total_cost", math.nan)

    # 若 txt 已直接输出聚合值，则优先保留；否则再由分项相加得到
    existing_upper_warehouse_cost = row.get("upper_warehouse_cost", math.nan)
    existing_upper_transportation_cost = row.get("upper_transportation_cost", math.nan)

    if math.isnan(existing_upper_warehouse_cost):
        if not math.isnan(fixed_cost) or not math.isnan(operation_cost):
            row["upper_warehouse_cost"] = (
                (0.0 if math.isnan(fixed_cost) else fixed_cost) +
                (0.0 if math.isnan(operation_cost) else operation_cost)
            )
        else:
            row["upper_warehouse_cost"] = math.nan
    else:
        row["upper_warehouse_cost"] = existing_upper_warehouse_cost

    if math.isnan(existing_upper_transportation_cost):
        if not math.isnan(trans_var_cost) or not math.isnan(trans_fix_cost):
            row["upper_transportation_cost"] = (
                (0.0 if math.isnan(trans_var_cost) else trans_var_cost) +
                (0.0 if math.isnan(trans_fix_cost) else trans_fix_cost)
            )
        else:
            row["upper_transportation_cost"] = math.nan
    else:
        row["upper_transportation_cost"] = existing_upper_transportation_cost

    # 占上层总成本的比例
    upper_ratio_pairs = {
        "upper_fixed_ratio": fixed_cost,
        "upper_operation_ratio": operation_cost,
        "upper_transportation_variable_ratio": trans_var_cost,
        "upper_transportation_fixed_ratio": trans_fix_cost,
        "upper_decay_ratio": decay_cost,
        "upper_excess_ratio": excess_cost,
        "upper_warehouse_ratio": row.get("upper_warehouse_cost", math.nan),
        "upper_transportation_ratio": row.get("upper_transportation_cost", math.nan),
    }
    for ratio_key, val in upper_ratio_pairs.items():
        if not math.isnan(val) and not math.isnan(upper_total_cost) and upper_total_cost > 1e-9:
            row[ratio_key] = val / upper_total_cost
        else:
            row[ratio_key] = math.nan

    # 占总成本的比例
    total_ratio_pairs = {
        "upper_fixed_total_ratio": fixed_cost,
        "upper_operation_total_ratio": operation_cost,
        "upper_transportation_variable_total_ratio": trans_var_cost,
        "upper_transportation_fixed_total_ratio": trans_fix_cost,
        "upper_decay_total_ratio": decay_cost,
        "upper_excess_total_ratio": excess_cost,
        "upper_warehouse_total_ratio": row.get("upper_warehouse_cost", math.nan),
        "upper_transportation_total_ratio": row.get("upper_transportation_cost", math.nan),
        "upper_total_ratio": upper_total_cost,
    }
    for ratio_key, val in total_ratio_pairs.items():
        if not math.isnan(val) and not math.isnan(total_cost) and total_cost > 1e-9:
            row[ratio_key] = val / total_cost
        else:
            row[ratio_key] = math.nan

    return row


def extract_nested_lower_block(text: str):
    m = re.search(r"best_individual:\n(.*)$", text, re.S)
    if not m:
        return None

    best_individual_tail = m.group(1)
    search_text = "\n" + best_individual_tail

    matches = list(re.finditer(r"\n\s*best_lower_plan:\n", search_text))
    if not matches:
        return None

    start = matches[-1].end()
    return search_text[start:].strip("\n")


def extract_single_lower_block(text: str):
    matches = list(re.finditer(r"\nbest_lower_plan:\n", text))
    if not matches:
        return None

    start = matches[-1].end()
    return text[start:]


# =========================
# 下层结果解析
# =========================
def extract_warehouse_customer_mapping(lower_text: str) -> Dict[int, List[int]]:
    """
    解析 lower block 中每个前置仓服务的客户编号。
    规则：按 warehouse_num 切块，并收集该块内所有 served_customers_num。
    """
    mapping: Dict[int, List[int]] = {}
    if not lower_text:
        return mapping

    text = lower_text.replace("\t", "    ")

    matches = list(re.finditer(r"warehouse_num\s*=\s*(\d+)", text))
    if not matches:
        return mapping

    for idx, m in enumerate(matches):
        wh_num = int(m.group(1))
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]

        served_lists = re.findall(r"served_customers_num\s*=\s*\[([^\]]*)\]", block)
        customers: List[int] = []
        for lst in served_lists:
            vals = [x.strip() for x in lst.split(",") if x.strip() != ""]
            for v in vals:
                try:
                    customers.append(int(v))
                except Exception:
                    pass

        if customers:
            unique_customers = sorted(set(customers))
            mapping[wh_num] = unique_customers

    return mapping

def extract_warehouse_level_metrics(lower_text: str) -> List[Dict[str, Any]]:
    """
    解析每个 warehouse_num 块下的仓级指标。
    """
    rows: List[Dict[str, Any]] = []
    if not lower_text:
        return rows

    text = lower_text.replace("\t", "    ")
    matches = list(re.finditer(r"warehouse_num\s*=\s*(\d+)", text))
    if not matches:
        return rows

    for idx, m in enumerate(matches):
        wh_num = int(m.group(1))
        start = m.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]

        served_count = math.nan
        served_match = re.search(r"served_customers_num\s*=\s*\[([^\]]*)\]", block)
        if served_match:
            vals = [x.strip() for x in served_match.group(1).split(",") if x.strip() != ""]
            served_count = len(vals)

        def extract_metric(pattern: str) -> float:
            mm = re.search(pattern, block)
            return safe_float(mm.group(1)) if mm else math.nan

        rows.append({
            "warehouse_num": wh_num,
            "served_customers_count": served_count,
            "warehouse_routes_distance": extract_metric(r"warehouse_routes_distance\s*=\s*([0-9eE\.\+\-]+)"),
            "warehouse_window_penalty": extract_metric(r"warehouse_window_penalty\s*=\s*([0-9eE\.\+\-]+)"),
            "warehouse_decay_cost": extract_metric(r"warehouse_decay_cost\s*=\s*([0-9eE\.\+\-]+)"),
            "warehouse_transportation_cost": extract_metric(r"warehouse_transportation_cost\s*=\s*([0-9eE\.\+\-]+)"),
            "vehicle_used_num": extract_metric(r"vehicle_used_num\s*=\s*([0-9eE\.\+\-]+)"),
        })

    return rows

def compute_structure_metrics_from_lower_block(lower_text: Optional[str], mapping: Optional[Dict[int, List[int]]]) -> dict:
    """
    基于下层最优解结构补充结构性字段，方便后续做时间窗灵敏度/结构变化分析。
    """
    row = {
        "customer_set_count": math.nan,
        "active_warehouse_count_from_lower": math.nan,
        "active_warehouse_ids_from_lower": None,
        "warehouse_id_span_from_lower": math.nan,

        "avg_customers_per_active_warehouse": math.nan,
        "min_customers_per_active_warehouse": math.nan,
        "max_customers_per_active_warehouse": math.nan,
        "std_customers_per_active_warehouse": math.nan,

        "avg_vehicle_per_active_warehouse": math.nan,
        "min_vehicle_per_active_warehouse": math.nan,
        "max_vehicle_per_active_warehouse": math.nan,
        "std_vehicle_per_active_warehouse": math.nan,

        "avg_warehouse_routes_distance": math.nan,
        "min_warehouse_routes_distance": math.nan,
        "max_warehouse_routes_distance": math.nan,
        "std_warehouse_routes_distance": math.nan,

        "avg_warehouse_window_penalty": math.nan,
        "min_warehouse_window_penalty": math.nan,
        "max_warehouse_window_penalty": math.nan,
        "std_warehouse_window_penalty": math.nan,

        "avg_warehouse_decay_cost": math.nan,
        "min_warehouse_decay_cost": math.nan,
        "max_warehouse_decay_cost": math.nan,
        "std_warehouse_decay_cost": math.nan,

        "avg_warehouse_transportation_cost": math.nan,
        "min_warehouse_transportation_cost": math.nan,
        "max_warehouse_transportation_cost": math.nan,
        "std_warehouse_transportation_cost": math.nan,
    }

    active_ids: List[int] = []
    if mapping:
        active_ids = sorted(mapping.keys())
        row["customer_set_count"] = len(mapping)
        row["active_warehouse_count_from_lower"] = len(mapping)
        row["active_warehouse_ids_from_lower"] = ",".join(map(str, active_ids)) if active_ids else ""
        if active_ids:
            row["warehouse_id_span_from_lower"] = max(active_ids) - min(active_ids)

        customer_counts = [len(v) for v in mapping.values()]
        if customer_counts:
            row["avg_customers_per_active_warehouse"] = float(np.mean(customer_counts))
            row["min_customers_per_active_warehouse"] = float(np.min(customer_counts))
            row["max_customers_per_active_warehouse"] = float(np.max(customer_counts))
            row["std_customers_per_active_warehouse"] = float(np.std(customer_counts))

    warehouse_rows = extract_warehouse_level_metrics(lower_text or "")
    if warehouse_rows:
        vehicle_list = [r["vehicle_used_num"] for r in warehouse_rows]
        route_distance_list = [r["warehouse_routes_distance"] for r in warehouse_rows]
        penalty_list = [r["warehouse_window_penalty"] for r in warehouse_rows]
        decay_list = [r["warehouse_decay_cost"] for r in warehouse_rows]
        transport_list = [r["warehouse_transportation_cost"] for r in warehouse_rows]

        vehicle_stats = summarize_numeric_list(vehicle_list, "vehicle_per_active_warehouse")
        row["avg_vehicle_per_active_warehouse"] = vehicle_stats["vehicle_per_active_warehouse_mean"]
        row["min_vehicle_per_active_warehouse"] = vehicle_stats["vehicle_per_active_warehouse_min"]
        row["max_vehicle_per_active_warehouse"] = vehicle_stats["vehicle_per_active_warehouse_max"]
        row["std_vehicle_per_active_warehouse"] = vehicle_stats["vehicle_per_active_warehouse_std"]

        route_stats = summarize_numeric_list(route_distance_list, "warehouse_routes_distance")
        row["avg_warehouse_routes_distance"] = route_stats["warehouse_routes_distance_mean"]
        row["min_warehouse_routes_distance"] = route_stats["warehouse_routes_distance_min"]
        row["max_warehouse_routes_distance"] = route_stats["warehouse_routes_distance_max"]
        row["std_warehouse_routes_distance"] = route_stats["warehouse_routes_distance_std"]

        penalty_stats = summarize_numeric_list(penalty_list, "warehouse_window_penalty")
        row["avg_warehouse_window_penalty"] = penalty_stats["warehouse_window_penalty_mean"]
        row["min_warehouse_window_penalty"] = penalty_stats["warehouse_window_penalty_min"]
        row["max_warehouse_window_penalty"] = penalty_stats["warehouse_window_penalty_max"]
        row["std_warehouse_window_penalty"] = penalty_stats["warehouse_window_penalty_std"]

        decay_stats = summarize_numeric_list(decay_list, "warehouse_decay_cost")
        row["avg_warehouse_decay_cost"] = decay_stats["warehouse_decay_cost_mean"]
        row["min_warehouse_decay_cost"] = decay_stats["warehouse_decay_cost_min"]
        row["max_warehouse_decay_cost"] = decay_stats["warehouse_decay_cost_max"]
        row["std_warehouse_decay_cost"] = decay_stats["warehouse_decay_cost_std"]

        transport_stats = summarize_numeric_list(transport_list, "warehouse_transportation_cost")
        row["avg_warehouse_transportation_cost"] = transport_stats["warehouse_transportation_cost_mean"]
        row["min_warehouse_transportation_cost"] = transport_stats["warehouse_transportation_cost_min"]
        row["max_warehouse_transportation_cost"] = transport_stats["warehouse_transportation_cost_max"]
        row["std_warehouse_transportation_cost"] = transport_stats["warehouse_transportation_cost_std"]

    return row


def parse_lower_block(lower_text) -> dict:
    row = {
        "lower_feasible": None,
        "lower_total_cost": math.nan,
        "window_penalty": math.nan,
        "decay_cost": math.nan,
        "transportation_cost": math.nan,
        "vehicle_used": math.nan,
        "served_customers_total": math.nan,
        "avg_customers_per_vehicle": math.nan,
        "warehouse_customer_mapping": None,
        "lower_block_raw": None,
    }

    if not lower_text:
        return row

    row["lower_block_raw"] = lower_text

    lower_text = lower_text.replace("\t", "    ")
    lower_lines = lower_text.splitlines()
    non_empty_lines = [ln for ln in lower_lines if ln.strip() != ""]
    if non_empty_lines:
        min_indent = min(len(ln) - len(ln.lstrip(" ")) for ln in non_empty_lines)
        lower_text = "\n".join(
            ln[min_indent:] if len(ln) >= min_indent else ln
            for ln in lower_lines
        )

    m = re.search(r"feasible\s*=\s*(True|False)", lower_text)
    if m:
        row["lower_feasible"] = (m.group(1) == "True")

    patterns = {
        "lower_total_cost": r"plan_total_cost\s*=\s*([0-9eE\.\+\-]+)",
        "window_penalty": r"plan_window_penalty\s*=\s*([0-9eE\.\+\-]+)",
        "decay_cost": r"plan_decay_cost\s*=\s*([0-9eE\.\+\-]+)",
        "transportation_cost": r"plan_transportation_cost\s*=\s*([0-9eE\.\+\-]+)",
        "vehicle_used": r"plan_vehicle_used\s*=\s*([0-9eE\.\+\-]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, lower_text)
        if m:
            row[key] = safe_float(m.group(1))

    mapping = extract_warehouse_customer_mapping(lower_text)
    if mapping:
        row["warehouse_customer_mapping"] = mapping
        row["served_customers_total"] = sum(len(v) for v in mapping.values())
    else:
        served_lists = re.findall(r"served_customers_num\s*=\s*\[([^\]]*)\]", lower_text)
        served_total = 0
        for lst in served_lists:
            vals = [x.strip() for x in lst.split(",") if x.strip() != ""]
            served_total += len(vals)
        row["served_customers_total"] = served_total

    if (
        not math.isnan(row["served_customers_total"])
        and not math.isnan(row["vehicle_used"])
        and row["vehicle_used"] > 0
    ):
        row["avg_customers_per_vehicle"] = row["served_customers_total"] / row["vehicle_used"]

    return row


# =========================
# 基于原始数据集补充结构指标
# =========================
def compute_extra_metrics_from_dataset(dataset_name: str, mapping: Optional[Dict[int, List[int]]]) -> dict:
    row = {
        "dataset_file_path": None,
        "avg_service_distance_km": math.nan,
        "max_service_distance_km": math.nan,
        "avg_warehouse_service_radius_km": math.nan,
        "max_warehouse_service_radius_km": math.nan,
        "std_warehouse_service_radius_km": math.nan,
        "served_demand_total": math.nan,
        "avg_customer_demand_served": math.nan,
        "avg_warehouse_served_demand": math.nan,
        "max_warehouse_served_demand": math.nan,
        "min_warehouse_served_demand": math.nan,
        "std_warehouse_served_demand": math.nan,
    }

    if not dataset_name or not mapping:
        dataset_info = load_original_dataset(dataset_name) if dataset_name else None
        if dataset_info is not None:
            row["dataset_file_path"] = dataset_info["path"]
        return row

    dataset_info = load_original_dataset(dataset_name)
    if dataset_info is None:
        return row

    row["dataset_file_path"] = dataset_info["path"]
    iw_df = dataset_info["iw_df"].copy()
    cm_df = dataset_info["cm_df"].copy()

    if "infront_warehouse_no" in iw_df.columns:
        iw_df["warehouse_num"] = iw_df["infront_warehouse_no"].apply(extract_trailing_int)
    else:
        iw_df["warehouse_num"] = np.arange(1, len(iw_df) + 1)

    if "customer_no" in cm_df.columns:
        cm_df["customer_num"] = cm_df["customer_no"].apply(extract_trailing_int)
    else:
        cm_df["customer_num"] = np.arange(1, len(cm_df) + 1)

    warehouse_coord_map = {}
    for _, r in iw_df.iterrows():
        wh_num = int(r["warehouse_num"])
        warehouse_coord_map[wh_num] = np.array([float(r["longitude"]), float(r["latitude"])])

    customer_coord_map = {}
    customer_demand_map = {}
    for _, r in cm_df.iterrows():
        c_num = int(r["customer_num"])
        customer_coord_map[c_num] = np.array([float(r["longitude"]), float(r["latitude"])])
        if "demand" in r:
            customer_demand_map[c_num] = float(r["demand"])

    all_distances: List[float] = []
    warehouse_radius_list: List[float] = []
    warehouse_demand_list: List[float] = []
    served_demand_total = 0.0
    served_customer_ids: List[int] = []

    for wh_num, customer_ids in mapping.items():
        if wh_num not in warehouse_coord_map:
            continue
        wh_coord = warehouse_coord_map[wh_num]
        local_distances = []
        local_demand = 0.0

        for c_id in customer_ids:
            if c_id not in customer_coord_map:
                continue
            c_coord = customer_coord_map[c_id]
            dist = haversine_distance(wh_coord, c_coord)
            all_distances.append(dist)
            local_distances.append(dist)
            served_customer_ids.append(c_id)
            if c_id in customer_demand_map:
                served_demand_total += customer_demand_map[c_id]
                local_demand += customer_demand_map[c_id]

        if local_distances:
            warehouse_radius_list.append(max(local_distances))
        if local_demand > 0:
            warehouse_demand_list.append(local_demand)

    if all_distances:
        row["avg_service_distance_km"] = float(np.mean(all_distances))
        row["max_service_distance_km"] = float(np.max(all_distances))

    if warehouse_radius_list:
        row["avg_warehouse_service_radius_km"] = float(np.mean(warehouse_radius_list))
        row["max_warehouse_service_radius_km"] = float(np.max(warehouse_radius_list))
        row["std_warehouse_service_radius_km"] = float(np.std(warehouse_radius_list))

    if served_customer_ids:
        row["served_demand_total"] = served_demand_total
        row["avg_customer_demand_served"] = served_demand_total / len(served_customer_ids)

    if warehouse_demand_list:
        row["avg_warehouse_served_demand"] = float(np.mean(warehouse_demand_list))
        row["max_warehouse_served_demand"] = float(np.max(warehouse_demand_list))
        row["min_warehouse_served_demand"] = float(np.min(warehouse_demand_list))
        row["std_warehouse_served_demand"] = float(np.std(warehouse_demand_list))

    return row


def scale_daily_lower_costs_to_period(row: dict, period_days: int = 30) -> dict:
    """
    将 txt 中按“日维度”输出的下层成本统一折算到周期口径。
    仅放大成本类字段，不改动车辆数、服务半径、客户数等结构性指标。
    """
    if period_days is None or period_days == 1:
        return row

    lower_cost_fields = [
        "lower_total_cost",
        "window_penalty",
        "decay_cost",
        "transportation_cost",
    ]

    for col in lower_cost_fields:
        val = row.get(col, math.nan)
        if not math.isnan(val):
            row[col] = val * period_days

    return row

# =========================
# 单个 txt 解析
# =========================
def parse_txt(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    row = {
        "file_path": path,
        "folder_name": os.path.basename(os.path.dirname(path)),
        "txt_name": os.path.basename(path),
    }

    row.update(extract_basic_info(text))
    row.update(extract_scale_info(row.get("dataset_name")))
    row.update(extract_upper_info(text))
    row.update(compute_active_warehouse_structure_metrics(row.get("y_raw")))

    mode = row.get("mode")
    if mode == "single":
        lower_block = extract_single_lower_block(text)
    elif mode == "nested":
        lower_block = extract_nested_lower_block(text)
    else:
        lower_block = None

    lower_info = parse_lower_block(lower_block)
    row.update(lower_info)

    row.update(
        compute_structure_metrics_from_lower_block(
            lower_text=lower_block,
            mapping=row.get("warehouse_customer_mapping"),
        )
    )
    row = scale_daily_lower_costs_to_period(row, LOWER_COST_PERIOD_DAYS)

    upper_cost = row.get("upper_total_cost", math.nan)
    lower_cost = row.get("lower_total_cost", math.nan)

    if not math.isnan(upper_cost) and not math.isnan(lower_cost):
        row["total_cost"] = upper_cost + lower_cost
    else:
        row["total_cost"] = math.nan
    row = compute_upper_cost_breakdown_metrics(row)

    if (
        not math.isnan(row.get("window_penalty", math.nan))
        and not math.isnan(row.get("lower_total_cost", math.nan))
        and row["lower_total_cost"] > 1e-9
    ):
        row["window_penalty_ratio"] = row["window_penalty"] / row["lower_total_cost"]
    else:
        row["window_penalty_ratio"] = math.nan

    if (
        not math.isnan(row.get("window_penalty", math.nan))
        and not math.isnan(row.get("total_cost", math.nan))
        and row["total_cost"] > 1e-9
    ):
        row["window_penalty_total_ratio"] = row["window_penalty"] / row["total_cost"]
    else:
        row["window_penalty_total_ratio"] = math.nan

    if (
        not math.isnan(row.get("decay_cost", math.nan))
        and not math.isnan(row.get("total_cost", math.nan))
        and row["total_cost"] > 1e-9
    ):
        row["decay_total_ratio"] = row["decay_cost"] / row["total_cost"]
    else:
        row["decay_total_ratio"] = math.nan

    if (
        not math.isnan(row.get("transportation_cost", math.nan))
        and not math.isnan(row.get("total_cost", math.nan))
        and row["total_cost"] > 1e-9
    ):
        row["transportation_total_ratio"] = row["transportation_cost"] / row["total_cost"]
    else:
        row["transportation_total_ratio"] = math.nan

    extra_metrics = compute_extra_metrics_from_dataset(
        row.get("dataset_name"),
        row.get("warehouse_customer_mapping"),
    )
    row.update(extra_metrics)

    if (
        not math.isnan(row.get("served_demand_total", math.nan))
        and not math.isnan(row.get("vehicle_used", math.nan))
        and row["vehicle_used"] > 0
    ):
        row["demand_per_vehicle"] = row["served_demand_total"] / row["vehicle_used"]
    else:
        row["demand_per_vehicle"] = math.nan

    if (
        not math.isnan(row.get("served_demand_total", math.nan))
        and not math.isnan(row.get("warehouse_open_count", math.nan))
        and row["warehouse_open_count"] > 0
    ):
        row["demand_per_open_warehouse"] = row["served_demand_total"] / row["warehouse_open_count"]
    else:
        row["demand_per_open_warehouse"] = math.nan

    if (
        not math.isnan(row.get("warehouse_open_count", math.nan))
        and not math.isnan(row.get("active_warehouse_count_from_lower", math.nan))
    ):
        row["open_count_gap_upper_vs_lower"] = (
            row["warehouse_open_count"] - row["active_warehouse_count_from_lower"]
        )
    else:
        row["open_count_gap_upper_vs_lower"] = math.nan

    return row


# =========================
# 批量构建明细表
# =========================
def build_detail_table(results_dir: str) -> pd.DataFrame:
    txt_files = glob.glob(os.path.join(results_dir, "**", "*_best.txt"), recursive=True)
    rows = []

    for path in txt_files:
        try:
            rows.append(parse_txt(path))
        except Exception as e:
            rows.append({
                "file_path": path,
                "folder_name": os.path.basename(os.path.dirname(path)),
                "txt_name": os.path.basename(path),
                "dataset_name": None,
                "mode": None,
                "algorithm_combo": None,
                "timestamp": None,
                "parse_error": str(e),
            })

    return pd.DataFrame(rows)


def drop_all_empty_columns(df: pd.DataFrame, keep_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    删除整列全为空（NaN / 空字符串 / '-'）的字段，避免导出的明细表中出现大量无效列。
    keep_cols 中的字段即使为空也会保留。
    """
    keep_cols = set(keep_cols or [])
    cleaned = df.copy()

    for col in cleaned.columns:
        if col in keep_cols:
            continue

        series = cleaned[col]

        # 将空字符串和'-'也视为缺失
        normalized = series.replace(r"^\s*$", np.nan, regex=True).replace("-", np.nan)

        if normalized.isna().all():
            cleaned = cleaned.drop(columns=[col])

    return cleaned

# =========================
# 主程序
# =========================
def main():
    df = build_detail_table(RESULTS_DIR)

    export_df = df.drop(columns=["lower_block_raw", "warehouse_customer_mapping"], errors="ignore")

    # 删除全空字段，但保留核心识别列
    export_df = drop_all_empty_columns(
        export_df,
        keep_cols=[
            "file_path",
            "folder_name",
            "txt_name",
            "dataset_name",
            "mode",
            "algorithm_combo",
            "timestamp",
            "num_cw",
            "num_iw",
            "num_cm",
            "dataset_index",
            "warehouse_open_count",
            "upper_total_cost",
            "lower_total_cost",
            "total_cost",
            "active_warehouse_ids",
            "active_warehouse_count_from_y",
            "active_warehouse_ids_from_lower",
            "active_warehouse_count_from_lower",
            "customer_set_count",
            "avg_customers_per_active_warehouse",
            "std_customers_per_active_warehouse",
            "avg_vehicle_per_active_warehouse",
            "avg_warehouse_routes_distance",
            "avg_warehouse_window_penalty",
            "avg_warehouse_service_radius_km",
            "avg_warehouse_served_demand",
            "open_count_gap_upper_vs_lower",
        ],
    )

    export_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    export_df.to_excel(OUTPUT_XLSX, index=False)

    print(f"已生成: {OUTPUT_FILE}")
    print(f"已生成: {OUTPUT_XLSX}")
    print(f"导出列数: {len(export_df.columns)}")

    print("\n=== mode统计 ===")
    if "mode" in df.columns:
        print(df["mode"].value_counts(dropna=False))

    print("\n=== nested 下层缺失统计 ===")
    if {"mode", "lower_total_cost", "dataset_name", "txt_name"}.issubset(df.columns):
        nested_df = df[df["mode"] == "nested"]
        if len(nested_df) > 0:
            missing_df = nested_df[nested_df["lower_total_cost"].isna()]
            print("nested 总数:", len(nested_df))
            print("nested lower_total_cost 缺失数:", len(missing_df))
            if len(missing_df) > 0:
                print("前5个缺失文件:")
                print(missing_df[["txt_name", "dataset_name"]].head(5))

    print("\n=== 原始数据集匹配统计 ===")
    if "dataset_file_path" in df.columns:
        matched = df["dataset_file_path"].notna().sum()
        print(f"匹配到原始xlsx的数据集数量: {matched} / {len(df)}")

    print("\n=== 规模统计 ===")
    scale_cols = [c for c in ["num_cw", "num_iw", "num_cm"] if c in df.columns]
    if len(scale_cols) == 3:
        print(df.groupby(scale_cols).size().reset_index(name="count"))


if __name__ == "__main__":
    main()