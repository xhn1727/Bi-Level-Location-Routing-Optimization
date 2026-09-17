import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Optional
from dataset_types import InputData, Individual, Assigned_Customer_Sets

class Freshness:
    def __init__(self,
                 input_data:InputData,
                 decay_cost: float,
                 upper_decay_rate: float,
                 lower_decay_rate: float
                ):
        self.input_data = input_data
        self.decay_cost = decay_cost
        self.upper_decay_rate = upper_decay_rate
        self.lower_decay_rate = lower_decay_rate
    
    def compute_freshness_info(self, w_if: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        d_if = self.input_data.d_cw_iw
        
        # 上层：每条 cw -> iw 调拨边的到达新鲜度
        freshness_matrix = np.zeros_like(d_if, dtype=float)
        for i, cw_node in enumerate(self.input_data.center_warehouses_list):
            freshness_matrix[i, :] = np.exp(-self.upper_decay_rate * d_if[i, :] / cw_node.max_speed)
        
        # 下层：按调拨量加权后的前置仓平均初始新鲜度
        total_weight_per_iw = np.sum(w_if, axis=0).astype(float)
        freshness_vector = np.ones(self.input_data.num_iw, dtype=float)
        
        for iw in range(self.input_data.num_iw):
            if total_weight_per_iw[iw] <= 1e-9:
                freshness_vector[iw] = 1.0
                continue
        
            weighted_freshness = 0.0
            for cw in range(self.input_data.num_cw):
                weighted_freshness += float(w_if[cw, iw]) * freshness_matrix[cw, iw]
        
            freshness_vector[iw] = weighted_freshness / total_weight_per_iw[iw]
        
        return freshness_matrix, freshness_vector
    
    def compute_upper_decay_cost(self, w_if: NDArray[np.float64], freshness_matrix:  NDArray[np.float64]) -> float:
        
        # 腐坏比例矩阵
        decay_matrix = 1.0 - freshness_matrix
        # 按调拨量加权计算上层腐坏成本
        decay_cost = self.decay_cost * np.sum(w_if * decay_matrix)

        return float(decay_cost)


    def compute_lower_decay_cost(self, iw_num: int, num_path: List[int], freshness_vector: NDArray[np.float64]) -> float:
        
        decay_cost = 0.0
        iw_node = self.input_data.infront_warehouses_list[iw_num - 1]
    
        # 当前前置仓的初始新鲜度
        freshness_f = float(freshness_vector[iw_num - 1])
    
        # 路径过短，无客户
        if len(num_path) < 3:
            return 0.0
    
        cumulative_time = 0.0
    
        for idx in range(1, len(num_path)):
            prev_node = num_path[idx - 1]
            curr_node = num_path[idx]
    
            # 计算当前路段距离
            if prev_node == 0 and curr_node != 0:
                distance = self.input_data.d_iw_cm[iw_num - 1, curr_node - 1]
            elif prev_node != 0 and curr_node != 0:
                distance = self.input_data.d_cm_cm[prev_node - 1, curr_node - 1]
            elif prev_node != 0 and curr_node == 0:
                distance = self.input_data.d_iw_cm[iw_num - 1, prev_node - 1]
            else:
                continue
    
            # 累计配送时间
            cumulative_time += distance / float(iw_node.max_speed)
    
            # 只有到达客户节点时才计算该客户的腐坏成本
            if curr_node != 0:
                demand = float(self.input_data.customers_list[curr_node - 1].demand)
    
                decay = 1.0 - freshness_f * np.exp(-self.lower_decay_rate * cumulative_time)
    
                decay_cost += self.decay_cost * demand * decay
    
        return decay_cost