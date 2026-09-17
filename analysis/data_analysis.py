import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import re


def build_metric_summary_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """""
    输入：原始长表 df_all
    列应包含：['file_name', 'num_cw', 'num_iw', 'num_cm', 'model_structure',
             'algorithm_combo', 'best_total_cost', 'total_time', 'upper_time', 'lower_time', 'status', 'error_message']
    """
    # 只保留成功结果
    df_success = df_all[df_all["status"] == "success"].copy()

    def extract_demand_increase_suffix(file_name: str) -> str:
        """从文件名中提取需求增加后缀：无后缀、+5、+10、+15、+20。"""
        if pd.isna(file_name):
            return "无后缀"
    
        file_name = str(file_name)
        base_name = re.sub(r"\.[^.]+$", "", file_name)  # 去掉扩展名
        match = re.search(r"\+(5|10|15|20|25|30|35|40|45|50|55|60|65|70)\b", base_name)
        if match:
            return f"+{match.group(1)}"
        return "无后缀"
    df_success["demand_increase_suffix"] = df_success["file_name"].apply(extract_demand_increase_suffix)

    # pivot 成 paired 表
    pivot_df = df_success.pivot_table(
        index=["file_name", "num_cw", "num_iw", "num_cm", "demand_increase_suffix"],
        columns="model_structure",
        values=["best_total_cost", "total_time", "upper_time", "lower_time"]
    )

    # 只保留两种结构都成功的样本
    paired = pivot_df.dropna().reset_index()

    # 拉平列名
    flat_cols = []
    for col in paired.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(col)
    paired.columns = flat_cols

    # 要分析的指标
    metrics = ["best_total_cost", "total_time", "upper_time", "lower_time"]

    # 构造 improvement 列
    for metric in metrics:
        paired[f"{metric}_improvement"] = (
            paired[f"{metric}_single"] - paired[f"{metric}_nested"]
        ) / paired[f"{metric}_single"]

    results = []

    # 一个内部函数：计算某个子样本的某个指标汇总
    def summarize_group(sub_df: pd.DataFrame, group_name: str, metric: str):
        single_col = f"{metric}_single"
        nested_col = f"{metric}_nested"
        improvement_col = f"{metric}_improvement"

        n_pairs = len(sub_df)
        if n_pairs == 0:
            return None

        # Wilcoxon 至少需要有一定样本，且不是完全相同
        p_value = np.nan
        try:
            if n_pairs >= 5 and not np.allclose(sub_df[single_col], sub_df[nested_col]):
                _, p_value = wilcoxon(sub_df[single_col], sub_df[nested_col])
        except Exception:
            p_value = np.nan

        return {
            "group": group_name,
            "metric": metric,
            "n_pairs": n_pairs,

            "single_mean": sub_df[single_col].mean(),
            "nested_mean": sub_df[nested_col].mean(),

            "improvement_mean": sub_df[improvement_col].mean(),
            "improvement_median": sub_df[improvement_col].median(),
            "improvement_std": sub_df[improvement_col].std(),

            "nested_win_rate": (sub_df[nested_col] < sub_df[single_col]).mean(),
            "p_value": p_value,
        }

    # =========================
    # 1. 全规模
    # =========================
    for metric in metrics:
        row = summarize_group(paired, "ALL", metric)
        if row is not None:
            results.append(row)

    # =========================
    # 2. 各规模
    # =========================
    for (cw, iw, cm), sub_df in paired.groupby(["num_cw", "num_iw", "num_cm"]):
        group_name = f"CW_{cw}_IW_{iw}_CM_{cm}"
        for metric in metrics:
            row = summarize_group(sub_df, group_name, metric)
            if row is not None:
                results.append(row)

    # =========================
    # 3. 按需求增加后缀汇总
    # =========================
    suffix_order = ["无后缀", "+5", "+10", "+15", "+20"]
    for suffix in suffix_order:
        sub_df = paired[paired["demand_increase_suffix"] == suffix]
        for metric in metrics:
            row = summarize_group(sub_df, f"DEMAND_{suffix}", metric)
            if row is not None:
                results.append(row)

    # =========================
    # 4. 各规模 × 需求增加后缀
    # =========================
    for (cw, iw, cm, suffix), sub_df in paired.groupby(["num_cw", "num_iw", "num_cm", "demand_increase_suffix"]):
        group_name = f"CW_{cw}_IW_{iw}_CM_{cm}_DEMAND_{suffix}"
        for metric in metrics:
            row = summarize_group(sub_df, group_name, metric)
            if row is not None:
                results.append(row)

    summary_df = pd.DataFrame(results)

    # 可选：按 group 和 metric 排序
    metric_order = {
        "best_total_cost": 0,
        "total_time": 1,
        "upper_time": 2,
        "lower_time": 3,
    }
    summary_df["metric_order"] = summary_df["metric"].map(metric_order)

    def build_group_sort_key(group_name: str):
        if group_name == "ALL":
            return (0, 0, 0, 0, 0, "")

        demand_match = re.fullmatch(r"DEMAND_(无后缀|\+5|\+10|\+15|\+20)", str(group_name))
        if demand_match:
            suffix = demand_match.group(1)
            suffix_rank = {"无后缀": 0, "+5": 1, "+10": 2, "+15": 3, "+20": 4}[suffix]
            return (1, 0, 0, 0, suffix_rank, "")

        scale_match = re.fullmatch(r"CW_(\d+)_IW_(\d+)_CM_(\d+)", str(group_name))
        if scale_match:
            cw, iw, cm = map(int, scale_match.groups())
            return (2, cw, iw, cm, 0, "")

        scale_demand_match = re.fullmatch(r"CW_(\d+)_IW_(\d+)_CM_(\d+)_DEMAND_(无后缀|\+5|\+10|\+15|\+20)", str(group_name))
        if scale_demand_match:
            cw, iw, cm = map(int, scale_demand_match.groups()[:3])
            suffix = scale_demand_match.group(4)
            suffix_rank = {"无后缀": 0, "+5": 1, "+10": 2, "+15": 3, "+20": 4}[suffix]
            return (3, cw, iw, cm, suffix_rank, "")

        return (9, 0, 0, 0, 0, str(group_name))

    group_sort_keys = summary_df["group"].apply(build_group_sort_key)
    summary_df[["group_level", "cw_order", "iw_order", "cm_order", "suffix_order", "group_name_order"]] = pd.DataFrame(
        group_sort_keys.tolist(), index=summary_df.index
    )

    summary_df = summary_df.sort_values(
        ["group_level", "cw_order", "iw_order", "cm_order", "suffix_order", "group_name_order", "metric_order"]
    ).drop(
        columns=["metric_order", "group_level", "cw_order", "iw_order", "cm_order", "suffix_order", "group_name_order"]
    ).reset_index(drop=True)

    return summary_df


df = pd.read_excel(r"供需比实验结果.xlsx")

summary_table = build_metric_summary_table(df)
print(summary_table)
summary_table.to_csv("供需比实验汇总.csv", index=False, encoding="utf-8-sig")
summary_table.to_excel("供需比实验汇总.xlsx", index=False)