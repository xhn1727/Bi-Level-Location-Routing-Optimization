import os
from pathlib import Path
import pandas as pd
import numpy as np

# =========================
# 参数区
# =========================
SELECTED_FILE = "selected_representative_datasets.xlsx"
SOURCE_DIR = Path("./算法性能对比实验/数据集")
OUTPUT_DIR = Path("./generated_datasets_timewindow_sensitivity")

# 一天的时间范围（按你的时间窗是小时制来设）
DAY_START = 0.0
DAY_END = 24.0

# 时间窗梯度：
# L0 = 无时间窗
# 其余 level 按“原始宽度 × factor”缩放
WINDOW_LEVELS = [
    ("L0_no_window", None),   # 特殊：直接设为 [0, 24]
    ("L1_very_loose", 2.0),
    ("L2_loose", 1.5),
    ("L3_base", 1.0),
    ("L4_tight", 0.75),
    ("L5_very_tight", 0.5),
    ("L6_extreme", 0.3),
    ("L7_near_limit", 0.15),
]

# 最小允许时间窗宽度，避免缩得完全重合
MIN_WINDOW_WIDTH = 0.25

# 是否保持 patience 不变
KEEP_PATIENCE = True

# =========================
# 工具函数
# =========================
def adjust_window_by_factor(early: float, late: float, factor: float,
                            day_start: float = 0.0,
                            day_end: float = 24.0,
                            min_width: float = 0.25):
    """
    围绕原始时间窗中心点缩放宽度，并裁剪到 [day_start, day_end] 内。
    """
    center = (early + late) / 2.0
    width = max(late - early, min_width)
    new_width = max(width * factor, min_width)

    new_early = center - new_width / 2.0
    new_late = center + new_width / 2.0

    # 若超出边界，则整体平移回来
    if new_early < day_start:
        shift = day_start - new_early
        new_early += shift
        new_late += shift
    if new_late > day_end:
        shift = new_late - day_end
        new_early -= shift
        new_late -= shift

    # 再次裁剪，双保险
    new_early = max(new_early, day_start)
    new_late = min(new_late, day_end)

    # 若极端情况下仍过窄
    if new_late - new_early < min_width:
        mid = (new_early + new_late) / 2.0
        new_early = max(day_start, mid - min_width / 2.0)
        new_late = min(day_end, mid + min_width / 2.0)

    return float(new_early), float(new_late)


def make_customer_windows(customer_df: pd.DataFrame, level_name: str, factor):
    """
    根据给定 level 生成新的 customer 表
    """
    df = customer_df.copy()

    if "early_time_limit" not in df.columns or "late_time_limit" not in df.columns:
        raise ValueError("customer 表缺少 early_time_limit 或 late_time_limit 字段")

    if level_name == "L0_no_window":
        df["early_time_limit"] = DAY_START
        df["late_time_limit"] = DAY_END
        return df

    new_early_list = []
    new_late_list = []

    for _, row in df.iterrows():
        e = float(row["early_time_limit"])
        l = float(row["late_time_limit"])
        new_e, new_l = adjust_window_by_factor(
            early=e,
            late=l,
            factor=factor,
            day_start=DAY_START,
            day_end=DAY_END,
            min_width=MIN_WINDOW_WIDTH
        )
        new_early_list.append(new_e)
        new_late_list.append(new_l)

    df["early_time_limit"] = new_early_list
    df["late_time_limit"] = new_late_list

    if not KEEP_PATIENCE:
        # 如果后续你想让窗口越紧，惩罚越敏感，可以在这里扩展 patience 调整逻辑
        pass

    return df


def find_source_file(dataset_name: str, source_dir: Path) -> Path:
    """
    根据 dataset_name 找原始 xlsx 文件
    """
    exact_path = source_dir / f"{dataset_name}.xlsx"
    if exact_path.exists():
        return exact_path

    # 若存在子目录或命名略有不同，做一次模糊搜索
    candidates = list(source_dir.rglob(f"{dataset_name}.xlsx"))
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        raise FileNotFoundError(f"找到多个同名文件，请手动确认：{dataset_name}")
    else:
        raise FileNotFoundError(f"未找到原始数据集文件：{dataset_name}.xlsx")


# =========================
# 主流程
# =========================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    selected_df = pd.read_excel(SELECTED_FILE)
    if "dataset_name" not in selected_df.columns:
        raise ValueError("代表性样本表中缺少 dataset_name 列")

    manifest_rows = []

    for _, row in selected_df.iterrows():
        dataset_name = str(row["dataset_name"]).strip()
        source_file = find_source_file(dataset_name, SOURCE_DIR)

        # 读取原始 workbook 三张表
        center_df = pd.read_excel(source_file, sheet_name="center_warehouse")
        iw_df = pd.read_excel(source_file, sheet_name="infront_warehouse")
        customer_df = pd.read_excel(source_file, sheet_name="customer")

        # 统计原始平均时间窗宽度
        base_width_mean = float((customer_df["late_time_limit"] - customer_df["early_time_limit"]).mean())

        dataset_output_dir = OUTPUT_DIR / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)

        for level_name, factor in WINDOW_LEVELS:
            new_customer_df = make_customer_windows(customer_df, level_name, factor)

            out_file = dataset_output_dir / f"{dataset_name}_{level_name}.xlsx"

            with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
                center_df.to_excel(writer, sheet_name="center_warehouse", index=False)
                iw_df.to_excel(writer, sheet_name="infront_warehouse", index=False)
                new_customer_df.to_excel(writer, sheet_name="customer", index=False)

            new_width_mean = float((new_customer_df["late_time_limit"] - new_customer_df["early_time_limit"]).mean())

            manifest_rows.append({
                "dataset_name": dataset_name,
                "source_file": str(source_file),
                "level_name": level_name,
                "factor": factor if factor is not None else "no_window",
                "output_file": str(out_file),
                "base_window_mean": base_width_mean,
                "new_window_mean": new_width_mean,
                "customer_count": len(new_customer_df),
            })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUTPUT_DIR / "timewindow_sensitivity_manifest.xlsx"
    manifest_df.to_excel(manifest_path, index=False)

    print("=== 时间窗灵敏度实验数据集生成完成 ===")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"清单文件: {manifest_path}")
    print()
    print("每个代表性数据集都生成了以下等级：")
    for name, factor in WINDOW_LEVELS:
        print(f"  - {name}: factor = {factor}")


if __name__ == "__main__":
    main()