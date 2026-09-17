from typing import List, Union, Optional
from cost_evaluation import Evaluation
from GA import GA
from ACO import ACO
import dataset_types as dt 
from Result import ResultLogger
from compute_freshness import Freshness
import time

class optimization():
    def __init__(self, 
                 input_data:dt.InputData, 
                 upper_algorithm: Union[GA], 
                 lower_algorithm: Union[ACO], 
                 patience: int, delta: float, 
                 evaluation_method: Evaluation,
                 compute_freshness: Optional[Freshness],  
                 dataset_name: str,
                 output_root: str = "results"
        ):
        self.input_data = input_data
        self.upper_algorithm = upper_algorithm
        self.lower_algorithm = lower_algorithm
        self.patience = patience
        self.delta = delta
        self.evaluation_method = evaluation_method
        self.compute_freshness = compute_freshness
        self.dataset_name = dataset_name
        self.output_root = output_root
        self.time_stats = evaluation_method.time_stats

    def _build_logger(self, mode: str) -> ResultLogger:
        algorithm_combo = f"{self.upper_algorithm.__class__.__name__}+{self.lower_algorithm.__class__.__name__}"
        return ResultLogger(
            dataset_name=self.dataset_name,
            algorithm_combo=algorithm_combo,
            mode=mode,
            output_root=self.output_root,
        )

    def optimize_GA_nested(self):
        logger = self._build_logger("nested")
        
        with logger.capture_prints():
            self.time_stats.upper_time = 0.0
            self.time_stats.lower_time = 0.0
            self.time_stats.total_time = 0.0
            t_total_start = time.perf_counter()

            best_individual: dt.Individual = None
            global_elite: dt.Individual = None
            history: List[dt.Individual] = []

            # 早停机制参数
            no_improve_count = 0           # 当前未提升代数计数
            prev_best_cost = float('inf')  # 上一最优解

            population: dt.Population = self.upper_algorithm.initialize_population()

            for gen in range(self.upper_algorithm.max_generaton):
                best_individual = population.best_individual
                print(f"第 {gen + 1} 代最优 → 成本: {best_individual.total_cost:.2f}")
                history.append(best_individual)

                # 早停判断逻辑
                if abs(prev_best_cost - best_individual.total_cost) < self.delta:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    prev_best_cost = best_individual.total_cost
                    global_elite = best_individual

                if no_improve_count >= self.patience:
                    print(f"提前停止：连续 {self.patience} 代无显著提升 ({abs(prev_best_cost - best_individual.total_cost)} < delta{self.delta})")
                    break

                population = self.upper_algorithm.evolve_population(popluation=population, generation_index=gen, global_elite=global_elite)
                print(f"第 {gen + 2} 代种群大小为{len(population.population_list)}")

            self.time_stats.total_time = time.perf_counter() - t_total_start
            self.time_stats.upper_time = self.time_stats.total_time - self.time_stats.lower_time
            print(f"upper_time = {self.time_stats.upper_time:.4f}")
            print(f"lower_time = {self.time_stats.lower_time:.4f}")
            print(f"total_time = {self.time_stats.total_time:.4f}")

        logger.save_nested_result(best_individual=best_individual, history=history)
        return best_individual, history
    
    def optimize_GA_single(self):
        logger = self._build_logger("single")
        with logger.capture_prints():
            self.time_stats.upper_time = 0.0
            self.time_stats.lower_time = 0.0
            self.time_stats.total_time = 0.0
            t_total_start = time.perf_counter()

            best_individual: dt.Individual = None
            global_elite: dt.Individual = None
            upper_history: List[dt.Individual] = []
    
            # 早停机制参数
            no_improve_count = 0           # 当前未提升代数计数
            prev_upper_best_cost = float('inf')  # 上一最优解
    
            population: dt.Population = self.upper_algorithm.initialize_population()
    
            for gen in range(self.upper_algorithm.max_generaton):
                best_individual = population.best_individual
                print(f"第 {gen + 1} 代最优 → 成本: {best_individual.total_cost:.2f}")
                upper_history.append(best_individual)
    
                # 早停判断逻辑
                if abs(prev_upper_best_cost - best_individual.total_cost) < self.delta:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                    prev_upper_best_cost = best_individual.total_cost
                    global_elite = best_individual
    
                if no_improve_count >= self.patience:
                    print(f"提前停止：连续 {self.patience} 代无显著提升 ({abs(prev_upper_best_cost - best_individual.total_cost)} < delta {self.delta})")
                    break
                
                population = self.upper_algorithm.evolve_population(popluation=population, generation_index=gen, global_elite=global_elite)
                print(f"第 {gen + 2} 代种群大小为{len(population.population_list)}")
            
            print("上层优化完成，开始优化下层")
    
            _, freshness_vector = self.compute_freshness.compute_freshness_info(w_if=best_individual.w_if)
            
            t_lower_start = time.perf_counter()
            best_plan, lower_history = self.lower_algorithm.run_lower_algorithm(y=best_individual.y, customer_sets=best_individual.customer_sets, freshness_vector=freshness_vector, verbose=True)
            self.time_stats.lower_time += time.perf_counter() - t_lower_start

            print(f"total_cost = {best_individual.total_upper_cost + self.input_data.period * best_plan.plan_total_cost}")
            self.time_stats.total_time = time.perf_counter() - t_total_start
            self.time_stats.upper_time = self.time_stats.total_time - self.time_stats.lower_time
            print(f"upper_time = {self.time_stats.upper_time:.4f}")
            print(f"lower_time = {self.time_stats.lower_time:.4f}")
            print(f"total_time = {self.time_stats.total_time:.4f}")
        logger.save_single_result(best_individual=best_individual, upper_history=upper_history, best_plan=best_plan, lower_history=lower_history)
        return best_individual, upper_history, best_plan, lower_history



         
          



        


        

        

        