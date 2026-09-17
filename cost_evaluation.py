import numpy as np
from numpy.typing import NDArray
from typing import List, Union
from dataset_types import InputData, Individual, Assigned_Customer_Sets, Plan, TimeStats
from compute_freshness import Freshness
from ACO import ACO
import time

class Evaluation:
    def __init__(self, 
                 input_data: InputData, 
                 excess_penalty_coef: float,
                 compute_freshness: Freshness,
                 time_stats: TimeStats = None
                 ):
       self.input_data = input_data
       self.excess_penalty_coef = excess_penalty_coef
       self.compute_freshness = compute_freshness
       self.time_stats = time_stats

    def compute_upper_excess_cost(self, w_if: NDArray[np.float64]) -> float:
       
       total_supply = float(np.sum(w_if))
       total_demand = self.input_data.total_period_demand
   
       excess_amount = max(total_supply - total_demand, 0.0)
   
       excess_cost = self.excess_penalty_coef * excess_amount
       return float(excess_cost)
    
class Evaluation_for_single(Evaluation):
    def __init__(self, input_data, excess_penalty_coef, compute_freshness, time_stats=None):
        super().__init__(input_data, excess_penalty_coef,compute_freshness, time_stats)

    def evaluation_total_cost(self, y: List[int], w_if: NDArray[np.float64], customer_sets: List[Assigned_Customer_Sets]) ->Individual:

        ind_build_operate_cost = 0.0
        ind_upper_decay_cost = 0.0
        ind_transportation_variable_cost = 0.0
        ind_transportation_fixed_cost = 0.0
        ind_transportation_cost = 0.0


        # 计算上层固定成本、运营成本
        for _ in range(self.input_data.num_iw):
            if y[_] == 1:
                ind_build_operate_cost += (self.input_data.period * (self.input_data.infront_warehouses_list[_].fixed_cost + self.input_data.infront_warehouses_list[_].operation_cost))
        # 计算上层运输成本
        for cw in range(self.input_data.num_cw):
            unit_var_cost = self.input_data.center_warehouses_list[cw].transportation_variable_cost
            unit_fixed_cost = self.input_data.center_warehouses_list[cw].transportation_fixed_cost
        
            ind_transportation_variable_cost += np.sum(self.input_data.d_cw_iw[cw, :] * w_if[cw, :] * unit_var_cost)
            ind_transportation_fixed_cost += np.sum(w_if[cw, :] > 1e-9) * unit_fixed_cost
        
        ind_transportation_cost = ind_transportation_variable_cost + ind_transportation_fixed_cost
        #计算上层运输腐坏成本
        freshness_matrix, _ = self.compute_freshness.compute_freshness_info(w_if=w_if)
        ind_upper_decay_cost += self.compute_freshness.compute_upper_decay_cost(w_if=w_if, freshness_matrix=freshness_matrix)
        #计算上层超额调拨成本
        ind_excess_penalty = self.compute_upper_excess_cost(w_if=w_if)
        # 上层总成本
        total_upper_cost = ind_build_operate_cost + ind_transportation_cost + ind_upper_decay_cost + ind_excess_penalty
       
        return Individual(y=y,
                          w_if=w_if,
                          customer_sets=customer_sets,
                          ind_build_operate_cost=ind_build_operate_cost,
                          ind_transportation_cost=ind_transportation_cost,
                          ind_decay_cost=ind_upper_decay_cost,
                          ind_excess_penalty=ind_excess_penalty,
                          best_lower_plan=Plan(),
                          total_upper_cost=total_upper_cost,
                          total_cost= total_upper_cost)

class Evaluation_for_nested(Evaluation):

    def  __init__(self, input_data, excess_penalty_coef, compute_freshness, lower_algorithm: Union[ACO], time_stats=None):
        super().__init__(input_data, excess_penalty_coef, compute_freshness, time_stats)
        self.lower_algorithm = lower_algorithm
    
    def evaluation_total_cost(self, y: List[int], w_if: NDArray[np.float64], customer_sets: List[Assigned_Customer_Sets]) ->Individual:

        ind_build_operate_cost = 0.0
        ind_upper_decay_cost = 0.0
        ind_transportation_variable_cost = 0.0
        ind_transportation_fixed_cost = 0.0
        ind_transportation_cost = 0.0
        # ----------------------------------计算上层总成本----------------------------------
        # 计算上层固定成本、运营成本
        for _ in range(self.input_data.num_iw):
            if y[_] == 1:
                ind_build_operate_cost += (self.input_data.period * (self.input_data.infront_warehouses_list[_].fixed_cost + self.input_data.infront_warehouses_list[_].operation_cost))
        # 计算上层运输成本
        for cw in range(self.input_data.num_cw):
            unit_var_cost = self.input_data.center_warehouses_list[cw].transportation_variable_cost
            unit_fixed_cost = self.input_data.center_warehouses_list[cw].transportation_fixed_cost
        
            ind_transportation_variable_cost += np.sum(self.input_data.d_cw_iw[cw, :] * w_if[cw, :] * unit_var_cost)
            ind_transportation_fixed_cost += np.sum(w_if[cw, :] > 1e-9) * unit_fixed_cost
        
        ind_transportation_cost = ind_transportation_variable_cost + ind_transportation_fixed_cost
        #计算上层运输腐坏成本
        freshness_matrix, freshness_vector = self.compute_freshness.compute_freshness_info(w_if=w_if)
        ind_upper_decay_cost += self.compute_freshness.compute_upper_decay_cost(w_if=w_if, freshness_matrix=freshness_matrix)
        #计算上层超额调拨成本
        ind_excess_penalty = self.compute_upper_excess_cost(w_if=w_if)
        # 上层总成本
        total_upper_cost = ind_build_operate_cost + ind_transportation_cost + ind_upper_decay_cost + ind_excess_penalty
        # ----------------------------------计算下层总成本----------------------------------
        # 计算下层最优方案
        t_lower_start = time.perf_counter()
        best_lower_plan, _ = self.lower_algorithm.run_lower_algorithm(y=y, customer_sets=customer_sets, freshness_vector=freshness_vector)
        if self.time_stats is not None:
            self.time_stats.lower_time += time.perf_counter() - t_lower_start
        #===================================返回总成本==========================
        return Individual(y=y,
                          w_if=w_if,
                          customer_sets=customer_sets,
                          ind_build_operate_cost=ind_build_operate_cost,
                          ind_transportation_cost=ind_transportation_cost,
                          ind_decay_cost=ind_upper_decay_cost,
                          ind_excess_penalty=ind_excess_penalty,
                          best_lower_plan=best_lower_plan,
                          total_upper_cost=total_upper_cost,
                          total_cost= total_upper_cost + self.input_data.period * best_lower_plan.plan_total_cost)