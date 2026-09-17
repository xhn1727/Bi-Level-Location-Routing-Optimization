from dataclasses import dataclass, field
import numpy as np
from typing import List, Optional
from numpy.typing import NDArray
from node import Center_Warehouse, Infront_Warehouse, Customer

@dataclass(frozen=True)
class InputData:
    num_cw: int = 0
    num_iw: int = 0
    num_cm: int = 0
    center_warehouses_list: List[Center_Warehouse] = field(default_factory=list)
    infront_warehouses_list: List[Infront_Warehouse] = field(default_factory=list)
    customers_list: List[Customer] = field(default_factory=list)
    d_cw_iw: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    d_iw_cm: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    d_cm_cm: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    period: int = 0
    total_period_demand: float = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "total_period_demand", float(self.period * sum(c.demand for c in self.customers_list)))

@dataclass
class Matrix:
    warehouse_num: int = 0
    pheromone: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    visibility: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=float))

@dataclass(frozen=True)
class Assigned_Customer_Sets:
    warehouse_num: int = 0
    assigned_customer_num: List[int] = field(default_factory=list)

@dataclass(frozen=True)
class Route:
    num_path: List[int] = field(default_factory=list)
    distance: float = 0.0
    window_penalty: float = 0.0

@dataclass(frozen=True)
class Warehouse_Routes:
    warehouse_num: int = 0
    routes: List[Route] = field(default_factory=list)
    served_customers_num: List[int] = field(default_factory=list)
    warehouse_routes_distance: float = 0.0
    warehouse_window_penalty: float = 0.0
    warehouse_decay_cost: float = 0.0
    warehouse_transportation_cost: float = 0.0
    vehicle_used_num: int = 0
    feasible: bool = True

@dataclass(frozen=True)
class Plan:
    plan_routes: List[Warehouse_Routes] = field(default_factory=list)
    plan_distance: float = 0.0
    plan_total_cost: float = 0.0
    plan_window_penalty: float = 0.0
    plan_decay_cost: float = 0.0
    plan_transportation_cost: float = 0.0
    plan_vehicle_used: int = 0
    feasible: bool = True

@dataclass(frozen=True)
class Iteration:
    iteration: int = 0
    plan: Plan = field(default_factory=Plan)

@dataclass(frozen=True)
class Individual:
    y: List[int] = field(default_factory=list)
    w_if: NDArray[np.float64] = field(default_factory=lambda: np.empty((0, 0), dtype=float))
    customer_sets: List[Assigned_Customer_Sets] = field(default_factory=list)
    ind_build_operate_cost: float = 0.0
    ind_transportation_cost: float = 0.0
    ind_decay_cost: float = 0.0
    ind_excess_penalty: float = 0.0
    total_upper_cost: float = 0.0
    best_lower_plan: Plan = field(default_factory=Plan)
    total_cost: float = 0.0

@dataclass(frozen=True)
class Population:
    population_list: List[Individual] = field(default_factory=list)
    best_individual: Optional[Individual] = None

@dataclass
class TimeStats:
    upper_time: float = 0.0
    lower_time: float = 0.0
    total_time: float = 0.0
