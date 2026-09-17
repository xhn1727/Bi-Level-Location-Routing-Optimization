# Bi-Level Location-Routing Optimization

A bi-level optimization framework for jointly solving front-warehouse location and vehicle routing problems in fresh e-commerce logistics.

## Overview

This project develops a bi-level optimization framework for coordinating front-warehouse location decisions and last-mile vehicle routing in a fresh e-commerce distribution network.

The upper-level problem determines front-warehouse openings and demand allocation, while the lower-level problem optimizes multi-vehicle delivery routes subject to vehicle capacity, customer time windows, and freshness-related costs.

A hybrid Genetic Algorithm–Ant Colony Optimization (GA–ACO) approach is used to solve the problem. The Genetic Algorithm searches over front-warehouse configurations at the upper level, while Ant Colony Optimization solves the corresponding routing problem at the lower level. The routing results are fed back into the upper-level fitness evaluation, allowing location and routing decisions to be optimized jointly rather than sequentially.

## Problem Formulation

The distribution network consists of three types of nodes:

- **Central warehouses**, which supply products to front warehouses;
- **Candidate front warehouses**, which may be opened to serve local demand;
- **Customer demand points**, which receive last-mile deliveries from opened front warehouses.

The optimization problem is divided into two interconnected levels.

### Upper-Level Problem

The upper level determines:

- which candidate front warehouses should be opened;
- how customer demand should be assigned to opened front warehouses;
- how products should be allocated from central warehouses to front warehouses.

The objective considers major system-level costs, including front-warehouse construction and operation, transportation from central warehouses to front warehouses, and freshness-related product loss.

Front-warehouse opening decisions are represented using binary variables, resulting in a discrete combinatorial optimization problem.

### Lower-Level Problem

Given an upper-level location and allocation solution, the lower level determines delivery routes for vehicles operating from each opened front warehouse.

The routing problem considers:

- vehicle capacity;
- the number of available vehicles;
- customer time windows;
- transportation and vehicle operating costs;
- freshness loss during delivery;
- penalties associated with early or late delivery.

The lower-level objective is to construct feasible multi-vehicle delivery routes while minimizing the resulting delivery-related costs.

## Solution Approach

The model is solved using a hybrid Genetic Algorithm–Ant Colony Optimization (GA–ACO) framework.

### Genetic Algorithm

The Genetic Algorithm operates at the upper level. Each individual represents a candidate front-warehouse opening configuration using binary encoding.

The GA explores alternative network configurations through population-based search, including selection, crossover, mutation, and population diversification mechanisms.

For each candidate configuration, demand and product flows are allocated according to the selected front warehouses before the corresponding routing problem is evaluated.

### Ant Colony Optimization

Ant Colony Optimization operates at the lower level.

For each opened front warehouse, ACO constructs delivery routes among its assigned customer demand points. Route construction accounts for vehicle capacity, customer time windows, delivery costs, and freshness-related losses.

The search is improved iteratively through pheromone evaporation and deposition, while additional perturbation mechanisms are used to maintain search diversity and reduce premature convergence.

### Bi-Level Feedback Mechanism

Two optimization structures are implemented for comparison:

1. **Sequential optimization** — upper-level location and allocation decisions are optimized first, after which vehicle routing is performed without feeding the routing results back into the upper-level search.

2. **Nested bi-level optimization** — the lower-level routing problem is evaluated during the upper-level search, and the resulting routing costs contribute to the fitness evaluation of upper-level solutions.

The nested structure therefore allows location and routing decisions to interact throughout the optimization process rather than treating them as independent stages.

## Experiments

### Synthetic Data Generation

Experiments are conducted using synthetically generated problem instances.

The data-generation process creates configurable central warehouses, candidate front warehouses, customer demand points, demand levels, and related operational parameters.

Using synthetic instances allows the optimization framework to be evaluated under different problem scales and operating conditions.

### Sequential vs. Nested Optimization

The sequential and nested optimization structures are evaluated under comparable algorithmic settings.

Both approaches use the same upper-level Genetic Algorithm and lower-level routing algorithm. The primary difference is whether lower-level routing outcomes are incorporated into upper-level fitness evaluation.

This experimental design isolates the effect of the bi-level feedback mechanism when comparing the two optimization structures.

### Sensitivity Analysis

Additional experiments examine how changes in demand and model parameters affect:

- total system cost;
- front-warehouse opening decisions;
- allocation decisions;
- routing costs;
- overall network structure.

These experiments are used to study how location and routing decisions respond jointly to changes in the operating environment.

## Project Structure

```text
Bi-Level-Location-Routing-Optimization/
├── main.py
├── run_optimization.py
├── batch_run_experiments.py
│
├── GA.py
├── ACO.py
├── cost_evaluation.py
├── compute_freshness.py
│
├── data.py
├── dataset_types.py
├── node.py
├── Result.py
│
├── data_generation/
│   ├── create_datasets.py
│   └── select_datasets_for_window_exp.py
│
├── analysis/
│   ├── data_analysis.py
│   ├── filter_datasets.py
│   └── txt_to_csv.py
│
├── README.md
└── .gitignore
```

### Core Optimization

| File | Description |
| --- | --- |
| `main.py` | Main entry point for configuring and running an optimization experiment |
| `run_optimization.py` | Controls the sequential and nested GA–ACO optimization workflows |
| `batch_run_experiments.py` | Runs optimization experiments across multiple datasets and parameter settings |
| `GA.py` | Genetic Algorithm for upper-level front-warehouse location optimization |
| `ACO.py` | Ant Colony Optimization for lower-level vehicle routing |
| `cost_evaluation.py` | Evaluates system costs and connects upper-level solutions with lower-level routing |
| `compute_freshness.py` | Computes freshness-related losses and costs |
| `data.py` | Loads and processes problem instances |
| `dataset_types.py` | Defines shared data structures used throughout the optimization framework |
| `node.py` | Defines central warehouse, front warehouse, and customer node structures |
| `Result.py` | Stores and outputs optimization results |

### Data Generation

| File | Description |
| --- | --- |
| `data_generation/create_datasets.py` | Generates synthetic problem instances for experiments |
| `data_generation/select_datasets_for_window_exp.py` | Prepares datasets for time-window experiments |

### Analysis and Post-processing

| File | Description |
| --- | --- |
| `analysis/data_analysis.py` | Performs statistical analysis of experimental results |
| `analysis/filter_datasets.py` | Filters generated datasets according to experimental requirements |
| `analysis/txt_to_csv.py` | Parses experimental outputs and converts extracted results into structured CSV files |

## Usage

The repository contains scripts for synthetic data generation, individual optimization runs, batch experiments, and result analysis.

A typical experimental workflow is:

1. Generate synthetic problem instances using `data_generation/create_datasets.py`.
2. Configure an optimization experiment in `main.py`.
3. Run the sequential or nested GA–ACO optimization workflow through `run_optimization.py`.
4. Use `batch_run_experiments.py` to conduct experiments across multiple datasets or parameter settings.
5. Process and analyze experimental results using the scripts under `analysis/`.

Runtime parameters, dataset paths, and algorithm settings are currently configured directly in the corresponding experiment scripts.

## Background

This project was originally developed as an undergraduate thesis project on integrated front-warehouse location and vehicle routing optimization for fresh e-commerce logistics.

The project focuses on the interaction between strategic facility-location decisions and operational vehicle-routing decisions, and investigates whether incorporating lower-level routing outcomes into upper-level optimization can improve coordination across the distribution network.