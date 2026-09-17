import pandas as pd
import numpy as np

# ========= 参数区 =========
INPUT_FILE = "best_plan_detail_6_12_100.xlsx"   # 也可以改成 csv
TARGET_NUM_CW = 6
TARGET_NUM_IW = 12
TARGET_NUM_CM = 100
TARGET_MODE = "nested"                 # 只基于 nested 选样本
EXCLUDE_TIGHT = True                   # 排除“_紧时间窗”数据
N_SELECT = 5

# 后续实验既包含“时间窗收紧”，也包含“需求增加”，
# 1) 成本水平；2) 时间窗压力；3) 车辆资源投入；4) 路径组织密度；5) 开仓结构。
FEATURE_COLS = [
    "total_cost",
    "window_penalty_ratio",
    "vehicle_used",
    "avg_customers_per_vehicle",
    "warehouse_open_count",
]

OUTPUT_SELECTED = "selected_representative_datasets.xlsx"
OUTPUT_POOL = "candidate_pool.xlsx"
OUTPUT_FEATURES = "candidate_pool_features.xlsx"


def load_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("仅支持 .xlsx 或 .csv 文件")


def minmax_scale(series: pd.Series) -> pd.Series:
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or pd.isna(s_max) or abs(s_max - s_min) < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - s_min) / (s_max - s_min)


# 与综合中心的欧氏距离

def euclidean_distance(row_vec: np.ndarray, ref_vec: np.ndarray) -> float:
    return float(np.sqrt(np.sum((row_vec - ref_vec) ** 2)))


# 自动选代表性数据集：
# 第1个选“最接近整体中心”的样本（最典型）
# 之后每次选“与已选样本最远”的样本（覆盖更多系统状态）
# 这样得到的5个样本既包含典型样本，也能覆盖成本、时间窗压力、车辆配置和路径结构差异。

def select_representative_samples(sub: pd.DataFrame, feature_cols: list[str], n_select: int) -> pd.DataFrame:
    work = sub.copy().reset_index(drop=True)

    # 1. 特征归一化
    for col in feature_cols:
        work[f"{col}_norm"] = minmax_scale(work[col])

    norm_cols = [f"{col}_norm" for col in feature_cols]
    feature_matrix = work[norm_cols].to_numpy(dtype=float)

    # 2. 计算整体中心（各归一化特征均值）
    center_vec = feature_matrix.mean(axis=0)
    work["distance_to_center"] = [euclidean_distance(v, center_vec) for v in feature_matrix]

    # 3. 第一个样本：选最接近中心的“典型样本”
    first_idx = int(work["distance_to_center"].idxmin())
    selected_indices = [first_idx]

    # 4. 后续样本：采用 maximin 规则，逐步扩大覆盖范围
    while len(selected_indices) < min(n_select, len(work)):
        candidate_indices = [i for i in range(len(work)) if i not in selected_indices]
        best_idx = None
        best_score = -np.inf

        for idx in candidate_indices:
            vec = feature_matrix[idx]
            min_dist_to_selected = min(
                euclidean_distance(vec, feature_matrix[s_idx])
                for s_idx in selected_indices
            )
            if min_dist_to_selected > best_score:
                best_score = min_dist_to_selected
                best_idx = idx

        selected_indices.append(int(best_idx))

    selected = work.loc[selected_indices].copy().reset_index(drop=True)

    # 5. 给每个样本贴标签，说明其角色
    role_labels = ["典型样本（最接近整体中心）"]
    for i in range(1, len(selected)):
        role_labels.append(f"差异覆盖样本{i}")

    selected.insert(0, "selection_role", role_labels)
    selected.insert(1, "selection_order", np.arange(1, len(selected) + 1))
    selected.insert(2, "total_sample_size", len(work))

    # 6. 输出各特征分位，便于论文解释“代表性”
    for col in feature_cols:
        selected[f"{col}_percentile"] = selected[col].rank(method="average", pct=True)

    return work, selected


def main():
    df = load_table(INPUT_FILE)

    required_cols = {
        "dataset_name", "mode", "num_cw", "num_iw", "num_cm",
        "total_cost", "window_penalty_ratio", "vehicle_used",
        "avg_customers_per_vehicle", "warehouse_open_count"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"输入表缺少必要字段: {missing}")

    # 1. 只取目标规模 + nested
    sub = df[
        (df["num_cw"] == TARGET_NUM_CW) &
        (df["num_iw"] == TARGET_NUM_IW) &
        (df["num_cm"] == TARGET_NUM_CM) &
        (df["mode"] == TARGET_MODE)
    ].copy()

    # 2. 排除紧时间窗数据（只保留基线）
    if EXCLUDE_TIGHT:
        sub = sub[~sub["dataset_name"].astype(str).str.endswith("_紧时间窗")].copy()

    # 3. 去掉关键特征缺失样本
    sub = sub.dropna(subset=FEATURE_COLS).copy()

    # 4. 若有重复 dataset_name，只保留一条
    sub = sub.sort_values("dataset_name").drop_duplicates(subset=["dataset_name"], keep="first")

    if len(sub) < N_SELECT:
        raise ValueError(f"当前可用样本数不足 {N_SELECT} 个，仅有 {len(sub)} 个。")

    # 5. 自动选择代表性样本
    work, selected = select_representative_samples(sub, FEATURE_COLS, N_SELECT)

    # 6. 输出候选池（原始指标）
    pool_cols = [
        "dataset_name", "mode", "num_cw", "num_iw", "num_cm",
        "total_cost", "window_penalty_ratio", "vehicle_used",
        "avg_customers_per_vehicle", "warehouse_open_count",
        "distance_to_center"
    ]
    work[pool_cols].sort_values("distance_to_center", ascending=True).to_excel(OUTPUT_POOL, index=False)

    # 7. 输出候选池（含归一化特征）
    feature_cols_export = pool_cols + [f"{col}_norm" for col in FEATURE_COLS]
    work[feature_cols_export].sort_values("distance_to_center", ascending=True).to_excel(OUTPUT_FEATURES, index=False)

    # 8. 输出最终代表性样本
    selected_cols = [
        "selection_role",
        "selection_order",
        "total_sample_size",
        "dataset_name",
        "mode",
        "num_cw", "num_iw", "num_cm",
        "total_cost",
        "window_penalty_ratio",
        "vehicle_used",
        "avg_customers_per_vehicle",
        "warehouse_open_count",
        "distance_to_center",
        "total_cost_percentile",
        "window_penalty_ratio_percentile",
        "vehicle_used_percentile",
        "avg_customers_per_vehicle_percentile",
        "warehouse_open_count_percentile",
    ]
    selected[selected_cols].to_excel(OUTPUT_SELECTED, index=False)

    print("=== 自动化代表性样本选择完成 ===")
    print(f"输入文件: {INPUT_FILE}")
    print(f"目标规模: ({TARGET_NUM_CW}, {TARGET_NUM_IW}, {TARGET_NUM_CM})")
    print(f"目标结构: {TARGET_MODE}")
    print(f"是否排除紧时间窗样本: {EXCLUDE_TIGHT}")
    print(f"基线样本数: {len(work)}")
    print()
    print("采用方法：多特征归一化 + 中心样本 + maximin差异覆盖")
    print("特征包括：", FEATURE_COLS)
    print()
    print("选中的代表性数据集：")
    print(selected[["selection_role", "dataset_name", "total_cost", "window_penalty_ratio", "vehicle_used"]])
    print()
    print(f"候选池（原始指标）已保存: {OUTPUT_POOL}")
    print(f"候选池（含归一化特征）已保存: {OUTPUT_FEATURES}")
    print(f"代表性样本表已保存: {OUTPUT_SELECTED}")


if __name__ == "__main__":
    main()