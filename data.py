import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from dataset_types import InputData
import node as nd  

class Dataset:
    def __init__(self, dataset_path: str, period: int):
        self.dataset_path = dataset_path
        self.period = period
        self._cache: Dict[str, Any] = {}

    def load_sheets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if "cw_df" in self._cache:
            return self._cache["cw_df"], self._cache["iw_df"], self._cache["cm_df"]

        cw_df = pd.read_excel(self.dataset_path, sheet_name="center_warehouse")
        iw_df = pd.read_excel(self.dataset_path, sheet_name="infront_warehouse")
        cm_df = pd.read_excel(self.dataset_path, sheet_name="customer")

        self._cache.update({"cw_df": cw_df, "iw_df": iw_df, "cm_df": cm_df})
        return cw_df, iw_df, cm_df

    @staticmethod
    def haversine_distance(coord1: np.ndarray, coord2: np.ndarray) -> float:
        R = 6371.0
        lon1, lat1 = coord1[0], coord1[1]
        lon2, lat2 = coord2[0], coord2[1]
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

    def compute_distance_matrix(self, src_coords: np.ndarray, dst_coords: np.ndarray) -> np.ndarray:
        d = np.zeros((len(src_coords), len(dst_coords)))
        for i, c1 in enumerate(src_coords):
            for j, c2 in enumerate(dst_coords):
                d[i, j] = self.haversine_distance(c1, c2)
        return d

    def build_problem(self) -> InputData:
        cw_df, iw_df, cm_df = self.load_sheets()

        cw_list = nd.Center_Warehouse.load_data(cw_df)
        iw_list = nd.Infront_Warehouse.load_data(iw_df)
        cm_list = nd.Customer.load_data(cm_df)

        center_coords = cw_df[["longitude", "latitude"]].to_numpy()
        warehouse_coords = iw_df[["longitude", "latitude"]].to_numpy()
        customer_coords = cm_df[["longitude", "latitude"]].to_numpy()

        # 距离矩阵缓存（灵敏度/对比实验会反复用）
        if "d_cw_iw" not in self._cache:
            self._cache["d_cw_iw"] = self.compute_distance_matrix(center_coords, warehouse_coords)
        if "d_iw_cm" not in self._cache:
            self._cache["d_iw_cm"] = self.compute_distance_matrix(warehouse_coords, customer_coords)
        if "d_cm_cm" not in self._cache:
            self._cache["d_cm_cm"] = self.compute_distance_matrix(customer_coords, customer_coords)

        return InputData(
            num_cw=len(cw_list),
            num_iw=len(iw_list),
            num_cm=len(cm_list),
            center_warehouses_list=cw_list,
            infront_warehouses_list=iw_list,
            customers_list=cm_list,
            d_cw_iw=self._cache["d_cw_iw"],
            d_iw_cm=self._cache["d_iw_cm"],
            d_cm_cm=self._cache["d_cm_cm"],
            period = self.period
        )