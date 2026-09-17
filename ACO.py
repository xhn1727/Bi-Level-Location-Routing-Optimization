import numpy as np
import random
from numpy.typing import NDArray
from typing import List, Tuple
import node as nd
from data import Dataset
from dataset_types import InputData,  Matrix, Assigned_Customer_Sets, Route, Warehouse_Routes, Plan, Iteration
from compute_freshness import Freshness

class ACO:
    def __init__(
        self,
        input_data: InputData,
        num_ants: int,
        num_iterations: int,
        alpha: float,
        beta: float,
        evaporation_rate: float,
        perturbation_strength: float,
        perturb_prob: float,
        q_deposit:float,
        early_penalty: float,
        late_penalty: float,
        patience: int,
        patience_delta: float,
        compute_freshness: Freshness
    ):
        self.input_data = input_data
        self.num_ants = int(num_ants)
        self.num_iterations = int(num_iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.evaporation_rate = float(evaporation_rate)
        self.perturbation_strength = float(perturbation_strength)
        self.perturb_prob = perturb_prob
        self.q_deposit = q_deposit
        self.early_penalty = float(early_penalty)
        self.late_penalty = float(late_penalty)
        self.patience = patience
        self.patience_delta = patience_delta
        self.compute_freshness = compute_freshness
        self.matrix_list = self.initialize_pheromone_visibility_for_all_iw() # 初始化矩阵
        # 注意：每次调用 run_lower_algorithm 时都会重置一次 matrix_list，避免不同上层个体之间共享信息素
    def reset_pheromone_visibility_for_all_iw(self) -> None:
        self.matrix_list = self.initialize_pheromone_visibility_for_all_iw()
    
    # ===============================================功能函数=====================================================================================
    def initialize_pheromone_visibility_for_single_iw(self, warehouse_coord: NDArray[np.float64], customer_coords: List[NDArray[np.float64]]
        ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        
        full_coords = np.vstack([warehouse_coord, customer_coords])

        pheromone = np.ones((self.input_data.num_cm + 1, self.input_data.num_cm + 1), dtype=float)
        visibility = np.zeros_like(pheromone)

        for i in range(self.input_data.num_cm + 1):
            for j in range(self.input_data.num_cm + 1):
                if i == j:
                    continue
                distance = Dataset.haversine_distance(full_coords[i], full_coords[j]) + 1e-6
                visibility[i, j] = 1.0 / distance

        return pheromone, visibility
    
    def initialize_pheromone_visibility_for_all_iw(self) -> List[Matrix]:
        
        initial_matrix_list = []
        customer_coords = [c.coord for c in self.input_data.customers_list]

        for i in range(self.input_data.num_iw):
           warehouse_coord = self.input_data.infront_warehouses_list[i].coord
           init_pheromone, init_visibility = self.initialize_pheromone_visibility_for_single_iw(warehouse_coord, customer_coords)
           initial_matrix_list.append(Matrix(
               warehouse_num = self.input_data.infront_warehouses_list[i].num,
               pheromone = init_pheromone,
               visibility = init_visibility))
        
        return initial_matrix_list   
        
          
    def evaluate_time_window(self, arrival_time: float, cm_node: nd.Customer) -> float:

        if arrival_time > cm_node.late_time_limit:
            return self.late_penalty * (np.e ** (cm_node.late_patience * (arrival_time - cm_node.late_time_limit)) - 1)
        
        if arrival_time < cm_node.early_time_limit:
            return self.early_penalty * (np.e ** (cm_node.early_patience * (cm_node.early_time_limit - arrival_time)) - 1)
        
        return 0.0

    def select_next_local(self, current_node_num: int, feasible_node_nums: List[int], pheromone: NDArray[np.float64], visibility: NDArray[np.float64]) -> int:

        if not feasible_node_nums:
            raise ValueError("缺少可行的消费者节点")

        probs = []
        for j in feasible_node_nums:
            tau = pheromone[current_node_num, j]
            eta = visibility[current_node_num, j]
            probs.append((tau ** self.alpha) * (eta ** self.beta))

        probs = np.array(probs, dtype=float)
        s = probs.sum()

        if s <= 0:
            return random.choice(feasible_node_nums) # 概率完全失效，均匀随机选一个
        probs /= s

        return random.choices(feasible_node_nums, weights=probs, k=1)[0]
    
    def perturb_pheromone(self, pheromone: NDArray[np.float64]) -> NDArray[np.float64]:

        perturbation = np.random.uniform(-self.perturbation_strength, self.perturbation_strength, pheromone.shape)
        pheromone += perturbation
        np.clip(pheromone, 0.0, 1.0, out=pheromone)
        return pheromone
    
    def compute_transportation_cost(self, iw_num: int, num_path: List[int]) -> float:
        load_weight = 0.0
        transportation_cost = 0.0
    
        # 反向路径（用于累计载重）
        reversed_path = list(reversed(num_path))
    
        for i in range(len(reversed_path) - 1):
            a = reversed_path[i]
            b = reversed_path[i + 1]
    
            if a == b:
                return transportation_cost
    
            # 前置仓 -> 客户
            if a == 0 and b != 0:
                dis_iw_first = self.input_data.d_iw_cm[iw_num - 1, b - 1]
    
                transportation_cost += (
                    self.input_data.infront_warehouses_list[iw_num - 1].vehicle_operation_cost
                    * dis_iw_first
                )
    
            # 客户 -> 客户
            elif a != 0 and b != 0:
                dis_cm_cm = self.input_data.d_cm_cm[a - 1, b - 1]
    
                load_weight += self.input_data.customers_list[a - 1].demand
    
                transportation_cost += (
                    self.input_data.infront_warehouses_list[iw_num - 1].vehicle_operation_cost
                    * dis_cm_cm
                    + self.input_data.infront_warehouses_list[iw_num - 1].vehicle_load_cost
                    * dis_cm_cm
                    * load_weight
                )
    
            # 客户 -> 前置仓
            elif a != 0 and b == 0:
                dis_last_iw = self.input_data.d_iw_cm[iw_num - 1, a - 1]
    
                transportation_cost += (
                    self.input_data.infront_warehouses_list[iw_num - 1].vehicle_operation_cost
                    * dis_last_iw
                )
    
        return transportation_cost
    
    # ===============================================ACO算法核心函数======================================================================================================
    def construct_routes_for_single_warehouse(self, iw_num: int, assigned_customers_num: List[int], freshness_vector: NDArray[np.float64], pheromone: NDArray[np.float64], visibility: NDArray[np.float64]
        ) -> Warehouse_Routes:

        iw_nodes: List[nd.Infront_Warehouse] = self.input_data.infront_warehouses_list
        cm_nodes: List[nd.Customer] = self.input_data.customers_list
        iw_node: nd.Infront_Warehouse = iw_nodes[iw_num-1]

        remaining_customers_num = set(assigned_customers_num)

        routes_list: List[Route] = []                            # 该前置仓所有路径
        total_served_num: List[int] = []                         # 该前置仓已服务的消费者编码          
        total_distance = 0.0                                     # 该前置仓所有距离
        total_window_penalty = 0.0                               # 该前置仓所有时间窗惩罚
        total_decay_cost = 0.0                                   # 该前置仓所有腐坏成本
        total_transportation_cost = 0.0                          # 该前置仓所有运输成本
        total_vehicle_count = 0                                  # 该前置仓使用的所有车辆数量
        
        # 如果缺少最大车辆数 或 缺少最大载重量 或 缺少最大速度
        if iw_node.max_vehicle_num is None or iw_node.max_load is None or iw_node.max_speed is None:
            raise ValueError(f'{iw_num}号前置仓缺少最大车辆数或最大载重量或最大速度数据')
        
        # 当仍然有消费者未被服务且使用车辆数量小于最大车辆数时：
        while remaining_customers_num and total_vehicle_count < int(iw_node.max_vehicle_num):
            current_node_num= 0                                  # 设0为前置仓的临时局部编号
            current_time = 0.0                                   # 设时间0时开始运输
            remaining_capacity = float(iw_node.max_load)         # 电瓶车剩余载重量，初始值为最大值
            
            local_served_num: List[int] = []                     # 该次路径规划已服务的消费者编码          
            local_route_distance = 0.0                           # 该次路径规划的总路程
            local_window_penalty = 0.0                           # 该次路径规划的时间窗惩罚
            local_decay_cost = 0.0                               # 该次路径规划的腐坏成本
            local_transportation_cost = 0.0                      # 该次路径规划的运输成本 
            
            # 无限循环
            while True:
                feasible_cm_nodes_num = [] # 可行的消费者编号集合
                for l in sorted(remaining_customers_num):
                    if float(cm_nodes[l-1].demand) <= remaining_capacity:
                        feasible_cm_nodes_num.append(l)
                # 如果没有可行消费者
                if not feasible_cm_nodes_num:
                    break
                
                # 选择下一个num
                next_num = self.select_next_local(current_node_num, feasible_cm_nodes_num, pheromone, visibility)
                # 下一个num所对应的node
                chosen_node = cm_nodes[next_num-1]

                # 查询距离
                if current_node_num == 0:  # 从当前前置仓发往第一个消费者
                    distance = self.input_data.d_iw_cm[iw_num-1, next_num-1]
                else:   # 消费者发往消费者
                    distance = self.input_data.d_cm_cm[current_node_num-1, next_num-1]
                
                travel_time = distance / float(iw_node.max_speed)               # 计算所需时间
                arrival_time = current_time + travel_time                       # 计算到达时间
                penalty = self.evaluate_time_window(arrival_time, chosen_node)  # 计算时间窗惩罚
                
                current_time = arrival_time                                     # 设定当前时间
                current_node_num = next_num                                     # 设定当前node_num

                remaining_capacity -= float(chosen_node.demand)                 # 更新车辆剩余载重
                local_route_distance += float(distance)                         # 更新行驶总里程
                local_window_penalty += float(penalty)                          # 更新时间窗惩罚
                
                local_served_num.append(int(chosen_node.num))

                total_served_num.append(int(chosen_node.num))

                remaining_customers_num.remove(next_num)

            if local_served_num:
                
                return_distance = self.input_data.d_iw_cm[ iw_num-1, current_node_num-1]
                local_route_distance += float(return_distance)
                total_vehicle_count += 1

                num_path = [0] + local_served_num + [0]
                 # 计算该条路径的运输成本
                local_transportation_cost = self.compute_transportation_cost(iw_num=iw_num, num_path=num_path)
                local_decay_cost = self.compute_freshness.compute_lower_decay_cost(iw_num=iw_num, num_path=num_path, freshness_vector=freshness_vector)


                routes_list.append(Route(num_path, local_route_distance, local_window_penalty))
            
                total_distance += float(local_route_distance)
                total_decay_cost += float(local_decay_cost)
                total_window_penalty += float(local_window_penalty)
                total_transportation_cost += float(local_transportation_cost)
            #else:
            #    if remaining_customers_num:
            #        print(f"[IW {iw_num}] 当前车辆无法继续服务，仍有未服务客户: {sorted(list(remaining_customers_num))}")
            #        print(f"[IW {iw_num}] 已服务客户: {sorted(local_served_num)}")
            #        print(f"[IW {iw_num}] 已用车辆数/最大车辆数: {total_vehicle_count}/{int(iw_node.max_vehicle_num)}")
            #    break
               
        iw_feasible = len(remaining_customers_num) == 0

        return Warehouse_Routes(
                warehouse_num = iw_node.num, 
                routes = routes_list,
                served_customers_num = total_served_num,
                warehouse_routes_distance = total_distance,
                warehouse_window_penalty = total_window_penalty,
                warehouse_decay_cost =  total_decay_cost,
                warehouse_transportation_cost= total_transportation_cost,
                vehicle_used_num = total_vehicle_count,
                feasible = iw_feasible
            )
    
    def create_plan_for_all_warehouses(self, y: List[int], customer_sets: List[Assigned_Customer_Sets], freshness_vector: NDArray[np.float64]) -> Plan:
     
        plan_routes: List[Warehouse_Routes] = []
        plan_distance = 0.0
        plan_total_cost = 0.0
        plan_window_penalty = 0.0
        plan_decay_cost = 0.0
        plan_transportation_cost = 0.0
        plan_vehicle_used = 0
        feasible = True
        
        for i in range(self.input_data.num_iw):
            if y[i] == 0 or not customer_sets[i].assigned_customer_num:
                continue
     
            p = self.matrix_list[i].pheromone
            v = self.matrix_list[i].visibility
     
            warehouse_routes = self.construct_routes_for_single_warehouse(
                iw_num=i + 1,
                assigned_customers_num=customer_sets[i].assigned_customer_num,
                freshness_vector=freshness_vector,
                pheromone=p,
                visibility=v
            )
     
            plan_routes.append(warehouse_routes)
            plan_distance += self.input_data.period * warehouse_routes.warehouse_routes_distance
            plan_window_penalty += self.input_data.period * warehouse_routes.warehouse_window_penalty
            plan_decay_cost += self.input_data.period * warehouse_routes.warehouse_decay_cost
            plan_transportation_cost += self.input_data.period * warehouse_routes.warehouse_transportation_cost
            plan_total_cost += self.input_data.period * (warehouse_routes.warehouse_decay_cost + warehouse_routes.warehouse_transportation_cost + warehouse_routes.warehouse_window_penalty)
            plan_vehicle_used += warehouse_routes.vehicle_used_num
     
            if not warehouse_routes.feasible:
                print(f"[PLAN] 前置仓 {warehouse_routes.warehouse_num} 不可行，导致整体方案不可行")
                feasible = False #任意一仓不可行，则整个方案不可行
        
        return Plan(
            plan_routes=plan_routes,
            plan_distance=plan_distance,
            plan_total_cost=plan_total_cost,
            plan_window_penalty=plan_window_penalty,
            plan_decay_cost=plan_decay_cost,
            plan_transportation_cost=plan_transportation_cost,
            plan_vehicle_used=plan_vehicle_used,
            feasible=feasible
        )

    def run_lower_algorithm(self, y: List[int], customer_sets: List[Assigned_Customer_Sets], freshness_vector: NDArray[np.float64], verbose: int = 0) -> Tuple[Plan, List[Plan]]:

        # 每次下层求解都从同一组初始信息素出发，避免不同上层解之间共享信息素
        self.reset_pheromone_visibility_for_all_iw()
        best_plan = Plan(plan_total_cost=float("inf"))
        history: List[Iteration] = []
        no_improve_count = 0

        # ACO主循环
        for it in range(self.num_iterations):
            iteration_best_plan = Plan(plan_total_cost=float("inf"))

            #  每轮迭代中的蚂蚁
            for ant in range(self.num_ants):
                # 开始规划方案
                ant_plan: Plan = self.create_plan_for_all_warehouses(y=y, customer_sets=customer_sets, freshness_vector=freshness_vector)
                # 如果方案不行，则开始跳过该蚂蚁
                if not ant_plan.feasible:
                    continue
                # 记录本次方案最优方案
                if ant_plan.plan_total_cost < iteration_best_plan.plan_total_cost:
                    iteration_best_plan = ant_plan
                    
            # 如果该次迭代没有可行解
            if not iteration_best_plan.feasible:
                history.append(Iteration(iteration=it, plan = iteration_best_plan))
                no_improve_count += 1
                if verbose:
                    print(f"第{it}轮迭代: 未找到可行解")
                if no_improve_count >= self.patience:
                    if verbose:
                       print(f"连续 {self.patience} 轮无改进，提前停止。")
                    break
                continue
            # 记录本次迭代最优解
            history.append(Iteration(iteration=it, plan=iteration_best_plan))

            # 先更新全局最优，再按“全局最优”强化信息素，提升收敛稳定性
            if iteration_best_plan.plan_total_cost < best_plan.plan_total_cost - self.patience_delta:
                best_plan = iteration_best_plan
                no_improve_count = 0
            else:
                no_improve_count += 1

            # 在全局最优路径的基础上蒸发、扰动与沉积
            for matrix in self.matrix_list:
                if y[matrix.warehouse_num - 1] == 1:
                    # e蒸发
                    matrix.pheromone *= (1.0 - self.evaporation_rate)

                    # 选择性扰动
                    if random.random() < self.perturb_prob:
                        matrix.pheromone[:] = self.perturb_pheromone(matrix.pheromone.copy())

                    if best_plan.plan_total_cost == float("inf"):
                        continue

                    delta = self.q_deposit / max(best_plan.plan_total_cost, 1e-6)

                    # 找到当前前置仓在全局最优方案中的路径
                    matched_routes = None
                    for warehouse_routes in best_plan.plan_routes:
                        if warehouse_routes.warehouse_num == matrix.warehouse_num:
                            matched_routes = warehouse_routes.routes
                            break

                    if matched_routes is None:
                        continue

                    for route in matched_routes:
                        served_globals = [int(x) for x in route.num_path[1:-1]]
                        if not served_globals:
                            continue

                        local_path = [0] + served_globals + [0]

                        for a, b in zip(local_path[:-1], local_path[1:]):
                            if 0 <= a < matrix.pheromone.shape[0] and 0 <= b < matrix.pheromone.shape[1]:
                                matrix.pheromone[a, b] += delta
                                matrix.pheromone[b, a] += delta
    
            if verbose:
                print(
                    f"第{it}轮迭代: 最优下层成本 = {iteration_best_plan.plan_total_cost:.4f}, "
                    f"Global = {best_plan.plan_total_cost:.4f}, "
                    f"no_improve = {no_improve_count}"
                )
    
            # 早停
            if no_improve_count >= self.patience:
                if verbose:
                    print(f"连续 {self.patience} 轮无改进，提前停止。")
                break
    
        if best_plan.plan_total_cost == float('inf'):
            best_plan = Plan(feasible=False, plan_total_cost=float("inf"))
            return best_plan, history
        
        return best_plan, history