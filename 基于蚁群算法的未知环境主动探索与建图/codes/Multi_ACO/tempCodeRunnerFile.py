
# --- Configuration ---
MAP_WIDTH = 50; MAP_HEIGHT = 50
OBSTACLE_PERCENTAGE = 0.25
MAP_TYPE = "random"
CELL_SIZE = 15
MAX_SIMULATION_STEPS = 3000 # 增加最大步数以确保能完全探索
RANDOM_SEED = 42

NUM_ROBOTS = 3
ROBOT_SENSOR_RANGE = 7
PHEROMONE_UPDATE_INTERVAL = 15

ACO_SHARED_PHEROMONE_CONFIG = {
    'n_ants_update': 8, 'n_iterations_update': 3, 'ant_max_steps': 50,
    'alpha_step_choice': 1.0, 'beta_step_heuristic': 2.5,
    'evaporation_rate_map': 0.05, 'pheromone_min_map': 0.01,
    'pheromone_max_map': 10.0, 'q_path_deposit_factor': 1.5,
    'max_pheromone_nav_steps': 100, 'eta_weight_to_unknown': 3.5,
    'eta_weight_to_frontiers_centroid': 2.0,
}

VISUALIZE = True
SIM_DELAY_MS = 20