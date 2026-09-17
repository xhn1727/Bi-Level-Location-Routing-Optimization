import os
import re
import traceback
from typing import Dict, List
import pandas as pd

from data import Dataset
from ACO import ACO
from GA import GA
from run_optimization import optimization
from cost_evaluation import Evaluation_for_nested, Evaluation_for_single
from compute_freshness import Freshness
from dataset_types import TimeStats


DATASET_DIR = "时间窗灵敏度分析/数据集"
SUMMARY_CSV = "时间窗实验结果.csv"
SUMMARY_XLSX = "时间窗实验结果.xlsx"
PERIOD = 30

# 超参数配置
ACO_PARAMS: Dict = {
    "num_ants": 25,
    "num_iterations": 60,
    "alpha": 1.8,
    "beta": 4.5,
    "evaporation_rate": 0.05,
    "perturbation_strength": 0.01,
    "perturb_prob": 0.02,
    "q_deposit": 150.0,
    "early_penalty": 1.0,
    "late_penalty": 1.5,
    "patience": 15,
    "patience_delta": 1e-3,
}

GA_PARAMS: Dict = {
    "POP_SIZE": 20,
    "target_ratio": 1.15,
    "max_attempts_rate": 200,
    "mutation_rate": 0.12,
    "inject_rate": 0.20,
    "max_mutation_rate": 0.35,
    "max_inject_rate": 0.45,
    "mutation_rate_coef": 0.20,
    "inject_rate_coef": 0.25,
    "crossover_rate": 0.85,
    "max_gnerations": 80,
    "top_k": 2,
    "distance_weight": 0.65,
    "capability_weight": 0.35,
    "trial_rate": 30,
}

COMMON_EVAL_PARAMS: Dict = {
    "excess_penalty_coef": 30,
}

FRESHNESS_PARAMS: Dict = {
    "decay_cost": 15,
    "upper_decay_rate": 0.2,
    "lower_decay_rate": 0.5,
}

OPT_PARAMS: Dict = {
    "patience": 10,
    "delta": 1e-3,
}

def natural_key(filename: str):
    return [
        int(text) if text.isdigit() else text
        for text in re.split(r'(\d+)', filename)
    ]

def parse_counts_from_filename(filename: str):
    stem = os.path.splitext(filename)[0]
    match = re.match(r"CW_(\d+)_IW_(\d+)_CM_(\d+)", stem)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None


def build_components(dataset_path: str, structure: str):
    """
    structure: 'single' 或 'nested'
    """
    input_data = Dataset(dataset_path=dataset_path, period=PERIOD).build_problem()
    time_stats = TimeStats()

    compute_freshness = Freshness(
        input_data=input_data,
        **FRESHNESS_PARAMS,
    )

    aco = ACO(
        input_data=input_data,
        compute_freshness=compute_freshness,
        **ACO_PARAMS,
    )

    if structure == "single":
        evaluation_method = Evaluation_for_single(
            input_data=input_data,
            compute_freshness=compute_freshness,
            time_stats=time_stats,
            **COMMON_EVAL_PARAMS,
        )
    elif structure == "nested":
        evaluation_method = Evaluation_for_nested(
            input_data=input_data,
            compute_freshness=compute_freshness,
            lower_algorithm=aco,
            time_stats=time_stats,
            **COMMON_EVAL_PARAMS,
        )
    else:
        raise ValueError(f"未知结构: {structure}")

    ga = GA(
        input_data=input_data,
        evaluation_method=evaluation_method,
        **GA_PARAMS,
    )

    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    opt = optimization(
        input_data=input_data,
        upper_algorithm=ga,
        lower_algorithm=aco,
        evaluation_method=evaluation_method,
        compute_freshness=compute_freshness,
        dataset_name=dataset_name,
        **OPT_PARAMS,
    )

    return input_data, opt, time_stats


def run_one_experiment(dataset_path: str, structure: str) -> Dict:
    file_name = os.path.basename(dataset_path)
    num_cw, num_iw, num_cm = parse_counts_from_filename(file_name)
    algorithm_combo = "GA-ACO"

    row = {
        "file_name": file_name,
        "num_cw": num_cw,
        "num_iw": num_iw,
        "num_cm": num_cm,
        "model_structure": structure,
        "algorithm_combo": algorithm_combo,
        "best_total_cost": None,
        "total_time": None,
        "upper_time": None,
        "lower_time": None,
        "status": "success",
        "error_message": "",
    }

    try:
        input_data, opt, time_stats = build_components(dataset_path, structure)

        if structure == "single":
            best_individual, _, best_plan, _ = opt.optimize_GA_single()
            if best_plan is not None and best_plan.feasible:
                row["best_total_cost"] = float(best_individual.total_cost + input_data.period * best_plan.plan_total_cost)
            else:
                row["best_total_cost"] = None
                row["status"] = "failed"
                row["error_message"] = "single structure lower plan is infeasible or missing"
        else:
            best_individual, _ = opt.optimize_GA_nested()
            row["best_total_cost"] = float(best_individual.total_cost)

        row["total_time"] = float(time_stats.total_time)
        row["upper_time"] = float(time_stats.upper_time)
        row["lower_time"] = float(time_stats.lower_time)

    except Exception as e:
        row["status"] = "failed"
        row["error_message"] = f"{type(e).__name__}: {e}"
        print(f"[FAILED] {file_name} | {structure} | {e}")
        traceback.print_exc()

    return row


def run_all_experiments(dataset_dir: str) -> pd.DataFrame:
    files = sorted(
    [f for f in os.listdir(dataset_dir) if f.endswith(".xlsx") and not f.startswith("~$")],
    key=natural_key)

    rows: List[Dict] = []

    for file_name in files:
        dataset_path = os.path.join(dataset_dir, file_name)

        for structure in ["single", "nested"]:
        #for structure in ["nested"]:
            print(f"\n===== Running {file_name} | {structure} =====")
            row = run_one_experiment(dataset_path, structure)
            rows.append(row)

            # 每跑完一次就保存一次，防止中途中断丢数据
            df = pd.DataFrame(rows)
            df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
            df.to_excel(SUMMARY_XLSX, index=False)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary_df = run_all_experiments(DATASET_DIR)
    print("\n实验完成，summary如下：")
    print(summary_df)