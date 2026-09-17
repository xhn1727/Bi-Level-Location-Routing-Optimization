import numpy as np
import pandas as pd
from numpy.typing import NDArray

class Node:
    def __init__(self, 
                 no: str, 
                 coord: NDArray[np.float64], 
                 num: int = None):
        self.no = no
        self.coord = coord
        self.num = num

    @classmethod
    def create_num_col(cls, dataset: pd.DataFrame) -> pd.DataFrame:
        dataset = dataset.copy().reset_index(drop=True)
        dataset['num'] = dataset.index + 1 # 从0开始的行号 + 1
        return dataset

class Center_Warehouse(Node):
    def __init__(self, 
                 no: str, 
                 coord: NDArray[np.float64], 
                 capability: float, 
                 max_speed: float,
                 transportation_variable_cost: float,
                 transportation_fixed_cost: float,
                 num: int = None):
        super().__init__(no, coord, num)
        self.capability = capability
        self.max_speed = max_speed
        self.transportation_variable_cost = transportation_variable_cost
        self.transportation_fixed_cost = transportation_fixed_cost

    @classmethod
    def load_data(cls, dataset: pd.DataFrame) -> list:
        dataset = cls.create_num_col(dataset)
        nodes = []
        for idx, row in dataset.iterrows():
            coord = [row["longitude"], row["latitude"]]
            nodes.append(
                cls(
                    no=row.get("center_warehouse_no", f"CW{idx+1}"),
                    coord=coord,
                    capability=float(row["upper_limit"]),
                    max_speed=float(row["max_speed"]),
                    transportation_variable_cost = float(row["transportation_variable_cost"]),
                    transportation_fixed_cost = float(row["transportation_fixed_cost"]),
                    num=int(row['num']),
                )
            )
        return nodes


class Infront_Warehouse(Node):
    def __init__(self,
                 no: str, 
                 coord: NDArray[np.float64], 
                 capability: float, 
                 fixed_cost: float, 
                 operation_cost: float, 
                 max_vehicle_num: int, 
                 max_load: float, 
                 max_speed: float,
                 vehicle_operation_cost: float,  
                 vehicle_load_cost: float,
                 num: int = None):
        super().__init__(no, coord, num)
        self.capability = capability
        self.fixed_cost = fixed_cost
        self.operation_cost = operation_cost
        self.max_vehicle_num = max_vehicle_num
        self.max_load = max_load
        self.max_speed = max_speed
        self.vehicle_operation_cost = vehicle_operation_cost
        self.vehicle_load_cost=vehicle_load_cost

    @classmethod
    def load_data(cls, dataset: pd.DataFrame) -> list:
        dataset = cls.create_num_col(dataset) 
        nodes = []
        for idx, row in dataset.iterrows():
            coord = [row["longitude"], row["latitude"]]
            nodes.append(
                cls(
                    no=row.get("infront_warehouse_no", row.get("warehouse_no", f"IW{idx+1}")),
                    coord=coord,
                    capability=float(row["upper_limit"]),
                    fixed_cost=float(row["fixed_cost"]),
                    operation_cost=float(row["operation_cost"]),
                    max_vehicle_num=int(row['max_vehicle_num']),
                    max_load=float(row['max_load']),
                    max_speed=float(row['max_speed']),
                    vehicle_operation_cost=float(row['vehicle_operation_cost']),
                    vehicle_load_cost=float(row['vehicle_load_cost']),
                    num=int(row['num']),
                )
            )
        return nodes


class Customer(Node):
    def __init__(self, 
                 no: str, 
                 coord: NDArray[np.float64], 
                 demand: float, 
                 early_time_limit: float, 
                 early_patience: float, 
                 late_time_limit: float, 
                 late_patience: float, 
                 num: int = None):
        super().__init__(no, coord, num)
        self.demand = demand
        self.early_time_limit = early_time_limit
        self.early_patience = early_patience
        self.late_time_limit = late_time_limit
        self.late_patience = late_patience

    @classmethod
    def load_data(cls, dataset: pd.DataFrame) -> list:
        dataset = cls.create_num_col(dataset)
        nodes = []
        for idx, row in dataset.iterrows():
            coord = [row["longitude"], row["latitude"]]
            nodes.append(
                cls(
                    no=row.get("customer_no", f"CM{idx+1}"),
                    coord=coord,
                    demand=float(row["demand"]),
                    early_time_limit=float(row["early_time_limit"]),
                    early_patience=float(row["early_patience"]),
                    late_time_limit=float(row["late_time_limit"]),
                    late_patience=float(row["late_patience"]),
                    num=int(row['num']),
                )
            )
        return nodes
