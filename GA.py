import numpy as np
from numpy.typing import NDArray
import random
from typing import List, Optional, Union
import copy
from dataset_types import InputData, Assigned_Customer_Sets, Individual, Population, Plan
from cost_evaluation import Evaluation_for_single, Evaluation_for_nested

class GA:
    def __init__(
            self,
            input_data: InputData,
            POP_SIZE: int,
            target_ratio: float,
            evaluation_method: Union[Evaluation_for_single, Evaluation_for_nested],
            max_attempts_rate: int,
            mutation_rate: float,
            inject_rate: float,
            max_mutation_rate: float,
            max_inject_rate: float,
            mutation_rate_coef: float,
            inject_rate_coef: float,
            crossover_rate: float,
            max_gnerations: int,
            top_k: int,
            distance_weight: float,
            capability_weight: float,
            trial_rate: int
        ):
        self.input_data = input_data
        self.POP_SIZE = int(POP_SIZE)
        self.target_ratio = float(target_ratio)
        self.evaluation_method = evaluation_method
        self.max_attempts_rate = max_attempts_rate
        self.mutation_rate = mutation_rate
        self.inject_rate = inject_rate
        self.max_mutation_rate = max_mutation_rate
        self.max_inject_rate = max_inject_rate
        self.mutation_rate_coef = mutation_rate_coef
        self.inject_rate_coef = inject_rate_coef
        self.crossover_rate = crossover_rate
        self.max_generaton = max_gnerations
        self.top_k = top_k
        self.distance_weight = distance_weight
        self.capability_weight = capability_weight
        self.trial_rate = trial_rate
        
    def generate_feasible_w_if(self, enabled_iw_list: List[int]) -> NDArray[np.float64]:
           
        required_total = self.input_data.total_period_demand * self.target_ratio
        
        w_if = np.zeros((self.input_data.num_cw, self.input_data.num_iw), dtype=float)
        
        cw_capability = np.array([cw.capability for cw in self.input_data.center_warehouses_list], dtype=float)
        remaining_cw_capability = cw_capability.copy()
        
        iw_capability = np.array([iw.capability for iw in self.input_data.infront_warehouses_list], dtype=float)
        remaining_iw_capability = iw_capability.copy()
        
        if not enabled_iw_list:
            return w_if
        
        max_cw_capability = max(cw_capability) if len(cw_capability) > 0 else 1.0
        
        # 第一步：按“归一化距离 + 剩余中心仓能力惩罚”进行启发式初始分配
        for iw in enabled_iw_list:
            dist_array = self.input_data.d_cw_iw[:, iw]
            dist_min = float(np.min(dist_array))
            dist_max = float(np.max(dist_array))
        
            sorted_cw_list = sorted(range(self.input_data.num_cw), key=lambda cw: (self.distance_weight * (self.input_data.d_cw_iw[cw, iw] - dist_min) / (dist_max - dist_min + 1e-6) + self.capability_weight * (1.0 - remaining_cw_capability[cw] / max_cw_capability)))
        
            for cw in sorted_cw_list:
                max_send = min(remaining_cw_capability[cw], remaining_iw_capability[iw])
        
                if max_send <= 0:
                    continue
        
                send = max_send * random.uniform(0.6, 0.95)
        
                w_if[cw, iw] += send
                remaining_cw_capability[cw] -= send
                remaining_iw_capability[iw] -= send
        
                if remaining_iw_capability[iw] <= 1e-9:
                    break
        
        total_supply = float(np.sum(w_if))
        
        # 若已满足目标或根本无货可供，直接返回
        if total_supply >= required_total or total_supply <= 1e-9:
            return w_if
        
        # 第二步：若未达目标，则按比例扩展已有分配
        scale = required_total / (total_supply + 1e-6)
        
        for cw in range(self.input_data.num_cw):
            for iw in enabled_iw_list:
                if w_if[cw, iw] <= 0:
                    continue
        
                add = w_if[cw, iw] * (scale - 1.0)
                add = min(add, remaining_cw_capability[cw], remaining_iw_capability[iw])
        
                if add <= 0:
                    continue
        
                w_if[cw, iw] += add
                remaining_cw_capability[cw] -= add
                remaining_iw_capability[iw] -= add
        
        return w_if
    
    def generate_feasible_customer_sets(self, enabled_iw_list: List[int], w_if: NDArray[np.float64], verbose: bool = False) -> Optional[List[Assigned_Customer_Sets]]:

            # 每个前置仓的上层调拨
            iw_capability: NDArray[np.float64] = np.sum(w_if, axis=0).astype(float)
        
            total_supply: float = float(np.sum(iw_capability))
            total_demand: float = self.input_data.total_period_demand
        
            # 供需总量硬约束
            if total_supply + 1e-6 < total_demand:
                if verbose:
                    print(f"不可行：上层调拨量不足 (供给={total_supply:.2f} < 需求={total_demand:.2f})")
                return None
        
            # 初始化每个前置仓的客户集合
            customer_sets: List[Assigned_Customer_Sets] = [
                Assigned_Customer_Sets(
                    warehouse_num=self.input_data.infront_warehouses_list[iw].num,
                    assigned_customer_num=[])
                for iw in range(self.input_data.num_iw)]
        
            # 逐个客户分配（按距离从近到远）
            for i in range(self.input_data.num_cm):
                demand_i = float(self.input_data.period * self.input_data.customers_list[i].demand)

                dist_array = self.input_data.d_iw_cm[:, i]
                dist_min = float(np.min(dist_array))
                dist_max = float(np.max(dist_array))

                pressure_list = []
                for f in enabled_iw_list:
                    pressure = demand_i / (iw_capability[f] + 1e-6)
                    pressure_list.append(pressure)
        
                pressure_min = float(min(pressure_list))
                pressure_max = float(max(pressure_list))
                
                sorted_f = sorted(enabled_iw_list, key=lambda f: (self.distance_weight * (self.input_data.d_iw_cm[f, i] - dist_min) / (dist_max - dist_min + 1e-6) + self.capability_weight * ((demand_i / (iw_capability[f] + 1e-6)) - pressure_min) / (pressure_max - pressure_min + 1e-6)))
                assigned = False

                for f in sorted_f:
                    if iw_capability[f] >= demand_i:
                        iw_capability[f] -= demand_i
                        # 记录客户分配
                        customer_sets[f].assigned_customer_num.append(self.input_data.customers_list[i].num)
                        assigned = True
                        break
        
                # 若客户未分配，直接判定不可行
                if not assigned:
                    if verbose:
                        print(f"不可行：客户 {self.input_data.customers_list[i].no} " f"(需求={demand_i:.2f}) 无可用仓满足")
                    return None
        
            if verbose:
                print(f"客户全部分配成功：启用仓数={len(enabled_iw_list)}, "f"总供给={total_supply:.2f}, 总需求={total_demand:.2f}")
                print(f"各仓剩余容量: {np.round(iw_capability, 2).tolist()}")
                print(f"总剩余容量 = {float(np.sum(iw_capability)):.2f}")
        
            return customer_sets
        
    def initialize_population(self) -> Population:
        attempts = 0
        max_attempts = self.POP_SIZE * self.max_attempts_rate
        population_list: List[Individual] = []

        best_individual: Optional[Individual] = None
        best_lower_plan: Optional[Plan] = None
        best_total_cost = float("inf")

        fail_customer_sets = 0
        fail_lower_plan = 0
        fail_zero_y = 0
        success_count = 0

        while len(population_list) < self.POP_SIZE and attempts < max_attempts:
            attempts += 1

            y = np.random.choice([0, 1], size=self.input_data.num_iw).tolist()
            if np.sum(y) == 0:
                fail_zero_y += 1
                continue

            enabled_iw_list = [iw for iw in range(self.input_data.num_iw) if y[iw] == 1]

            w_if = self.generate_feasible_w_if(enabled_iw_list=enabled_iw_list)
            customer_sets = self.generate_feasible_customer_sets(enabled_iw_list=enabled_iw_list, w_if=w_if)

            if customer_sets is None:
                fail_customer_sets += 1
                continue

            ind = self.evaluation_method.evaluation_total_cost(y=y, w_if=w_if, customer_sets=customer_sets)

            if ind.best_lower_plan is None or not ind.best_lower_plan.feasible:
                fail_lower_plan += 1
                continue

            population_list.append(ind)
            success_count += 1

            total_cost = ind.total_upper_cost + ind.best_lower_plan.plan_total_cost
            if total_cost < best_total_cost:
                best_total_cost = total_cost
                best_individual = ind
                best_lower_plan = ind.best_lower_plan

        print(f"初始化结束: attempts={attempts}, success={success_count}, "
              f"fail_customer_sets={fail_customer_sets},fail_zero_y = {fail_zero_y}, fail_lower_plan={fail_lower_plan},")

        if best_individual is None or best_lower_plan is None:
            raise ValueError("初始化种群失败：未生成任何可行个体")

        return Population(
            population_list=population_list,
            best_individual=best_individual)
    
    @staticmethod
    def select(individuals: List[Individual], fitness_probs: List[float]) -> Individual:
        idx = np.random.choice(len(individuals), p=fitness_probs)
        return copy.deepcopy(individuals[idx])

    def evolve_population(self, popluation: Population, generation_index: int = 0, global_elite: Optional[Individual] = None) -> Population:
        new_population: List[Individual] = []
        fitness_scores: List[float] = []
        individuals: List[Individual] = []
        cost_records: List[float] = []

        max_generation = max(1, int(self.max_generaton))
        mutation_rate = min(self.max_mutation_rate, self.mutation_rate + self.mutation_rate_coef * (generation_index / max_generation),)
        inject_rate = min(self.max_inject_rate,self.inject_rate + self.inject_rate_coef * (generation_index / max_generation),)
        

        for ind in popluation.population_list:
            if (ind.best_lower_plan is None or not ind.best_lower_plan.feasible or not np.isfinite(ind.total_cost) or ind.total_cost > 1e9):
                fitness = 1e-8
            else:
                fitness = np.exp(-ind.total_cost / 1e5)

            fitness_scores.append(fitness)
            individuals.append(ind)
            cost_records.append(ind.total_cost)

        if not individuals:
            raise ValueError("当前种群为空，无法进化种群")

        fitness_sum = sum(fitness_scores)
        fitness_probs = ([f / fitness_sum for f in fitness_scores] if fitness_sum > 0 else [1 / len(fitness_scores)] * len(fitness_scores))

        elite_indices = np.argsort(cost_records)[: self.top_k]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(individuals[idx]))

        if global_elite is not None:
            if all(tuple(ind.y) != tuple(global_elite.y) for ind in new_population):
                new_population.append(copy.deepcopy(global_elite))

        target_size = self.POP_SIZE
        seen_y = {tuple(ind.y) for ind in new_population}

        max_trials = target_size * self.trial_rate
        trial = 0

        while len(new_population) < target_size and trial < max_trials:
            trial += 1

            # ===== 随机注入 =====
            if random.random() < inject_rate:
                rand_y = np.random.choice([0, 1], size=self.input_data.num_iw).tolist()

                if np.sum(rand_y) == 0:
                    rand_y[random.randint(0, self.input_data.num_iw - 1)] = 1

                y_tuple = tuple(rand_y)
                if y_tuple in seen_y:
                    continue

                enabled_iw_list = [iw for iw in range(self.input_data.num_iw) if rand_y[iw] == 1]
                rand_w_if = self.generate_feasible_w_if(enabled_iw_list=enabled_iw_list)
                rand_customer_sets = self.generate_feasible_customer_sets(enabled_iw_list=enabled_iw_list, w_if=rand_w_if)

                if rand_customer_sets is None:
                    continue

                rand_ind = self.evaluation_method.evaluation_total_cost(y=rand_y, w_if=rand_w_if, customer_sets=rand_customer_sets)

                if rand_ind.best_lower_plan is None or not rand_ind.best_lower_plan.feasible:
                    continue

                new_population.append(rand_ind)
                seen_y.add(y_tuple)
                continue

            # ===== 遗传生成 =====
            parent1 = self.select(individuals, fitness_probs)
            parent2 = self.select(individuals, fitness_probs)
            y1, y2 = parent1.y, parent2.y

            if random.random() < self.crossover_rate:
                mask = np.random.randint(0, 2, size=len(y1))
                child_y = (mask * np.array(y1) + (1 - mask) * np.array(y2)).clip(0, 1).astype(int).tolist()
            else:
                child_y = copy.deepcopy(y1)

            if random.random() < mutation_rate:
                idx = random.randint(0, len(child_y) - 1)
                child_y[idx] = 1 - child_y[idx]

            if np.sum(child_y) == 0:
                child_y[random.randint(0, len(child_y) - 1)] = 1

            y_tuple = tuple(child_y)
            if y_tuple in seen_y:
                continue

            enabled_iw_list = [iw for iw in range(self.input_data.num_iw) if child_y[iw] == 1]
            child_w_if = self.generate_feasible_w_if(enabled_iw_list=enabled_iw_list)
            child_customer_sets = self.generate_feasible_customer_sets(enabled_iw_list=enabled_iw_list, w_if=child_w_if)

            if child_customer_sets is None:
                continue

            child_ind = self.evaluation_method.evaluation_total_cost(y=child_y, w_if=child_w_if, customer_sets=child_customer_sets)

            if child_ind.best_lower_plan is None or not child_ind.best_lower_plan.feasible:
                continue

            new_population.append(child_ind)
            seen_y.add(y_tuple)

        if len(new_population) < target_size:
            print(f"警告：去重后仅生成 {len(new_population)} 个个体，未达到目标种群规模 {target_size}")

        best_individual = min(new_population, key=lambda ind: ind.total_cost)
        best_total_cost = best_individual.total_cost

        return Population(
            population_list=new_population,
            best_individual=best_individual)









