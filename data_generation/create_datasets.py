import os
import numpy as np
import pandas as pd

SAVE_DIR = "供需比变化实验1"
os.makedirs(SAVE_DIR, exist_ok=True)

# (中心仓数, 前置仓数, 消费者需求点数)
scales = [
    (3, 6, 25),
    #(4, 8, 50),
    #(6, 12, 100),
    #(8, 16, 200),
]


demand = {
    "+5" : (15,35),
    "+10": (20,40),
    "+15": (25,45),
    "+20": (30,50),
    "+25": (35,55),
    "+30": (40,60),
    "+35": (45,65),
    "+40":(50,70),
    "+45":(55,75),
    "+50":(60,80),
    "+55":(65,85),
    "+60":(70,90),
    "+65":(75,95),
    "+70":(80,100)}


def get_scale_profile(num_cw: int, num_iw: int, num_cm: int) -> dict:
    """
    按规模分层设置更合理的随机数据范围。
    核心目标：
    1. 小规模不要天然不可行；
    2. 中大规模保留一定挑战性；
    3. 保证需求、容量、车辆能力、时间窗之间量纲协调。
    """
    if num_cm <= 25:
        # small
        return {
            "lon_range": (120.00, 120.05),
            "lat_range": (30.00, 30.05),
            "cw_upper": (30000, 40000),
            "cw_speed": (40, 60),
            "cw_var_cost": (5, 15),
            "cw_fixed_cost": (100, 200),
            "iw_fixed_cost": (8000, 15000),
            "iw_operation_cost": (8000, 12000),
            "iw_upper": (20000, 25000),
            "iw_vehicle_num": (6, 12),
            "iw_max_load": (120, 220),
            "iw_speed": (25, 40),
            "iw_vehicle_op_cost": (1.0, 2.0),
            "iw_vehicle_load_cost": (0.5, 1.5),
            "cm_demand":(55,75),
            "early_time": (7, 10),
            "late_span": (5, 9),
            "early_patience": (0.1, 0.5),
            "late_patience": (0.5, 1.0),
        }
    elif num_cm <= 80:
        # medium
        return {
            "lon_range": (120.00, 120.10),
            "lat_range": (30.00, 30.10),
            "cw_upper": (60000, 100000),
            "cw_speed": (40, 60),
            "cw_var_cost": (5, 15),
            "cw_fixed_cost": (100, 200),
            "iw_fixed_cost": (8000, 15000),
            "iw_operation_cost": (14000, 18000),
            "iw_upper": (20000, 25000),
            "iw_vehicle_num": (8, 16),
            "iw_max_load": (120, 260),
            "iw_speed": (25, 40),
            "iw_vehicle_op_cost": (1.0, 2.0),
            "iw_vehicle_load_cost": (0.5, 1.5),
            "cm_demand": (15, 40),
            "early_time": (6, 10),
            "late_span": (4, 8),
            "early_patience": (0.1, 0.5),
            "late_patience": (0.5, 1.0),
        }
    else:
        # large
        return {
            "lon_range": (120.00, 120.15),
            "lat_range": (30.00, 30.15),
            "cw_upper": (250000, 300000),
            "cw_speed": (40, 60),
            "cw_var_cost": (5, 15),
            "cw_fixed_cost": (100, 200),
            "iw_fixed_cost": (8000, 15000),
            "iw_operation_cost": (16000, 20000),
            "iw_upper": (20000, 25000),
            "iw_vehicle_num": (12, 24),
            "iw_max_load": (140, 300),
            "iw_speed": (25, 42),
            "iw_vehicle_op_cost": (1.0, 2.0),
            "iw_vehicle_load_cost": (0.5, 1.5),
            "cm_demand": (20, 50),
            "early_time": (6, 10),
            "late_span": (4, 8),
            "early_patience": (0.1, 0.5),
            "late_patience": (0.5, 1.0),
        }


def random_coord(n: int, lon_range: tuple, lat_range: tuple):
    lon = np.random.uniform(*lon_range, n)
    lat = np.random.uniform(*lat_range, n)
    return lon, lat


def generate_dataset(num_cw: int, num_iw: int, num_cm: int):
    profile = get_scale_profile(num_cw, num_iw, num_cm)

    # ---------- 中心仓 ----------
    lon, lat = random_coord(num_cw, profile["lon_range"], profile["lat_range"])
    cw_df = pd.DataFrame({
        "center_warehouse_no": [f"CW{i+1}" for i in range(num_cw)],
        "longitude": lon,
        "latitude": lat,
        # 周期供给能力（kg）
        "upper_limit": np.random.uniform(*profile["cw_upper"], num_cw),
        # 干线运输速度（km/h）
        "max_speed": np.random.uniform(*profile["cw_speed"], num_cw),
        # 单位运输变动成本
        "transportation_variable_cost": np.random.uniform(*profile["cw_var_cost"], num_cw),
        # 每条调拨线路固定成本
        "transportation_fixed_cost": np.random.uniform(*profile["cw_fixed_cost"], num_cw),
    })

    # ---------- 前置仓 ----------
    lon, lat = random_coord(num_iw, profile["lon_range"], profile["lat_range"])
    iw_df = pd.DataFrame({
        "infront_warehouse_no": [f"IW{i+1}" for i in range(num_iw)],
        "longitude": lon,
        "latitude": lat,
        # 周期固定成本
        "fixed_cost": np.random.uniform(*profile["iw_fixed_cost"], num_iw),
        # 周期运营成本
        "operation_cost": np.random.uniform(*profile["iw_operation_cost"], num_iw),
        # 周期承接能力（kg）
        "upper_limit": np.random.uniform(*profile["iw_upper"], num_iw),
        # 最大车辆数
        "max_vehicle_num": np.random.randint(profile["iw_vehicle_num"][0], profile["iw_vehicle_num"][1] + 1, num_iw),
        # 单车最大载重（kg）
        "max_load": np.random.uniform(*profile["iw_max_load"], num_iw),
        # 末端配送速度（km/h）
        "max_speed": np.random.uniform(*profile["iw_speed"], num_iw),
        # 车辆固定运营成本
        "vehicle_operation_cost": np.random.uniform(*profile["iw_vehicle_op_cost"], num_iw),
        # 载重相关单位成本
        "vehicle_load_cost": np.random.uniform(*profile["iw_vehicle_load_cost"], num_iw),
    })

    # ---------- 消费者需求点 ----------
    lon, lat = random_coord(num_cm, profile["lon_range"], profile["lat_range"])
    early_time = np.random.uniform(*profile["early_time"], num_cm)
    late_time = early_time + np.random.uniform(*profile["late_span"], num_cm)

    cm_df = pd.DataFrame({
        "customer_no": [f"CM{i+1}" for i in range(num_cm)],
        "longitude": lon,
        "latitude": lat,
        # 日均需求（kg）——消费者需求点
        "demand": np.random.uniform(*profile["cm_demand"], num_cm),
        # 时间窗（小时）
        "early_time_limit": early_time,
        "early_patience": np.random.uniform(*profile["early_patience"], num_cm),
        "late_time_limit": late_time,
        "late_patience": np.random.uniform(*profile["late_patience"], num_cm),
    })

    return cw_df, iw_df, cm_df


def generate_all(n: int):
    for (cw, iw, cm) in scales:
        for i in range(n):
            cw_df, iw_df, cm_df = generate_dataset(cw, iw, cm)

            filename = f"CW_{cw}_IW_{iw}_CM_{cm}_{i+1}+45.xlsx"
            path = os.path.join(SAVE_DIR, filename)

            with pd.ExcelWriter(path) as writer:
                cw_df.to_excel(writer, sheet_name="center_warehouse", index=False)
                iw_df.to_excel(writer, sheet_name="infront_warehouse", index=False)
                cm_df.to_excel(writer, sheet_name="customer", index=False)

            print(f"已生成: {path}")


if __name__ == "__main__":
    generate_all(100)