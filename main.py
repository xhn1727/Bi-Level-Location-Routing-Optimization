from data import Dataset
from ACO import ACO
from GA import GA
from run_optimization import optimization
from cost_evaluation import Evaluation_for_nested, Evaluation_for_single
from compute_freshness import Freshness
from dataset_types import TimeStats
import os

def main():
    time_stats = TimeStats()

    dataset_path = r"generated_datasets/CW_3_IW_5_CM_50_1.xlsx"
    input_data = Dataset(dataset_path=dataset_path, period=30).build_problem()

    compute_freshness = Freshness(
        input_data=input_data,
        decay_cost=10, 
        upper_decay_rate=0.1, 
        lower_decay_rate=0.5)

    aco = ACO(input_data=input_data,
              num_ants = 20,
              num_iterations = 80,
              alpha = 1.2,
              beta = 3.5,
              evaporation_rate = 0.2,
              perturbation_strength = 0.05,
              perturb_prob = 0.1,
              q_deposit = 100.0,
              early_penalty = 5.0,
              late_penalty = 10.0,
              patience = 20,
              patience_delta = 1e-3,
              compute_freshness=compute_freshness)
    
    evaluation_method = Evaluation_for_nested(
        input_data=input_data, 
        excess_penalty_coef=300, 
        compute_freshness=compute_freshness,
        time_stats=time_stats,
        lower_algorithm=aco)

    ga = GA(input_data=input_data,
            POP_SIZE = 30,
            target_ratio = 1.10,
            evaluation_method=evaluation_method,
            max_attempts_rate = 150,
            mutation_rate = 0.08,
            inject_rate = 0.12,
            max_mutation_rate = 0.25,
            max_inject_rate = 0.35,
            mutation_rate_coef = 0.15,
            inject_rate_coef = 0.20,
            crossover_rate = 0.80,
            max_gnerations = 60,
            top_k = 3,
            distance_weight = 0.70,
            capability_weight = 0.30,
            trial_rate = 30)
    
    opt = optimization(input_data=input_data,
                       upper_algorithm=ga,
                       lower_algorithm=aco,
                       patience=30,
                       delta=1e-3,
                       evaluation_method=evaluation_method,
                       compute_freshness=compute_freshness,
                       dataset_name=os.path.splitext(os.path.basename(dataset_path))[0]
                       )
    
    opt.optimize_GA_nested()

if __name__ == "__main__":
    main()