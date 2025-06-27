import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import pygame
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt
import random
import shutil

try:
    from environment import Environment, FREE, OBSTACLE, UNKNOWN
    from robot import Robot
    from visualizer import Visualizer
    from planners.deep_aco_simple_planner import DeepACOSimplePlanner
    from planners.aco_planner import ACOPlanner as OriginalACOPlanner # The frontier-based one
    from planners.fbe_planner import FBEPlanner
    from planners.ige_planner import IGEPlanner
except ImportError as e:
    print(f"Import Error: {e}. Please check file paths.")
    exit(1)

# --- Evaluation Configuration ---
EVAL_MAP_SIZES = [(30, 30)]
EVAL_OBSTACLE_PERCENTAGES = [0.15]
EVAL_MAP_TYPES = ["random"]
NUM_RUNS_PER_CONFIG = 3 # Run multiple times for robust comparison
MASTER_SEED_EVAL = 2024 

# Planners to compare. Use a unique name for your new planner.
PLANNERS_TO_EVALUATE = ["DeepACO", "OriginalACO"] 

# Path to the trained model for DeepACO
TRAINED_MODEL_PATH = "deep_aco_simple_ppo_model.pth"

# General simulation parameters for evaluation
ROBOT_SENSOR_RANGE_EVAL = 5
MAX_PHYSICAL_STEPS_EVAL = 60
MAX_PLANNING_CYCLES_EVAL = 400

# Visualization
VISUALIZE_EVALUATION = False # Set to True to watch some runs
EVAL_CELL_SIZE = 12
EVAL_SIM_DELAY = 0.01

# --- Parameters for initializing planners ---
# For DeepACOSimplePlanner (must match trained architecture)
DEEP_ACO_EVAL_PARAMS = {
    'n_ants': 8, 'n_iterations': 5, 'alpha': 1.0, 'beta': 2,
    'evaporation_rate': 0.1, 'pheromone_initial': 0.1, 'pheromone_min': 0.01,
    'pheromone_max': 5.0, 'local_frontier_ig_sum_radius': 5,
    'robot_sensor_range': ROBOT_SENSOR_RANGE_EVAL,
    'is_training': False # <<< CRITICAL: Set to False for evaluation
}
# For OriginalACOPlanner (based on your aco_planner.py)
ORIGINAL_ACO_EVAL_PARAMS = {
    'n_ants': 18, 'n_iterations': 20, 'alpha': 1.0, 'beta': 2,
    'evaporation_rate': 0.25, 'q0': 0.7, 'ig_weight_heuristic': 3.0,
    'robot_sensor_range_for_heuristic': ROBOT_SENSOR_RANGE_EVAL
}

RESULTS_DIR_EVAL = "evaluation_results_comparison"
os.makedirs(RESULTS_DIR_EVAL, exist_ok=True)


def run_single_evaluation_episode(planner_name, planner_instance, env_config, seed):
    """Runs a single evaluation episode and returns exploration history."""
    # print(f"    Running seed {seed}...")
    np.random.seed(seed); random.seed(seed)
    env = Environment(env_config['width'], env_config['height'], 
                      env_config['obs_perc'], env_config['map_type'])
    
    planner_instance.environment = env # Ensure planner has correct env reference

    robot = Robot(env.robot_start_pos_ref, ROBOT_SENSOR_RANGE_EVAL)
    robot.sense(env)
    
    show_this_run = VISUALIZE_EVALUATION and (seed == MASTER_SEED_EVAL) # e.g., only viz the first seed run
    visualizer = None
    if show_this_run:
        try:
            visualizer = Visualizer(env.width, env.height, EVAL_CELL_SIZE)
            pygame.display.set_caption(f"Evaluating: {planner_name} on Seed {seed}")
        except pygame.error: visualizer = None

    total_steps, total_cycles = 0, 0
    current_path = []
    
    # Store history as list of tuples: (step, explored_percentage)
    explored_history = [(0, np.sum(env.grid_known != UNKNOWN) / (env.width * env.height) * 100)]

    while total_steps < MAX_PHYSICAL_STEPS_EVAL and total_cycles < MAX_PLANNING_CYCLES_EVAL:
        if visualizer and not visualizer.handle_events_simple(): break
        
        if np.sum(env.grid_known != UNKNOWN) / (env.width * env.height) >= 0.995: break

        if not current_path:
            target_pos, path = planner_instance.plan_next_action(robot.get_position(), env.grid_known)
            total_cycles += 1
            if not target_pos or not path or len(path) <= 1: break
            current_path = path[1:]

        if current_path:
            next_step = current_path.pop(0)
            if not robot.attempt_move_one_step(next_step, env):
                current_path = []
            total_steps += 1
            robot.sense(env)
            
            explored_history.append((total_steps, np.sum(env.grid_known != UNKNOWN) / (env.width * env.height) * 100))

        if visualizer:
            visualizer.clear_screen(); visualizer.draw_grid_map(env.grid_known)
            visualizer.draw_robot_path(robot.actual_pos_history)
            visualizer.update_display(); time.sleep(EVAL_SIM_DELAY)
    
    if visualizer: visualizer.quit()
    return explored_history

def evaluate_all_planners():
    print(f"--- Evaluating planners: {PLANNERS_TO_EVALUATE} ---")
    
    planners_to_test = {}
    dummy_env = Environment(10,10,0.1,"random") 

    for planner_name in PLANNERS_TO_EVALUATE:
        if planner_name == "OriginalACO":
            planner_instance = DeepACOSimplePlanner(dummy_env, **DEEP_ACO_EVAL_PARAMS)
            if os.path.exists(TRAINED_MODEL_PATH.replace(".pth", "_actor.pth")):
                planner_instance.load_model(TRAINED_MODEL_PATH)
            else:
                print(f"FATAL: Trained model for {planner_name} not found at {TRAINED_MODEL_PATH}")
                continue
            planners_to_test[planner_name] = planner_instance
        elif planner_name == "DeepACO":
             # This assumes your planners/aco_planner.py contains the original frontier-based ACO
             planners_to_test[planner_name] = OriginalACOPlanner(dummy_env, **ORIGINAL_ACO_EVAL_PARAMS)
    
    all_results = {} 

    for map_size in EVAL_MAP_SIZES:
        for obs_perc in EVAL_OBSTACLE_PERCENTAGES:
            for map_type in EVAL_MAP_TYPES:
                map_config_str = f"Size_{map_size[0]}x{map_size[1]}_Obs_{obs_perc}_Type_{map_type}"
                print(f"\n--- Evaluating Map Configuration: {map_config_str} ---")

                for planner_name, planner_instance in planners_to_test.items():
                    print(f"  Testing Planner: {planner_name}...")
                    config_key = (planner_name, f"{map_size[0]}x{map_size[1]}", obs_perc, map_type)
                    all_results[config_key] = {'steps': [], 'explored': []}

                    for i in range(NUM_RUNS_PER_CONFIG):
                        seed = MASTER_SEED_EVAL + i
                        env_config = {'width': map_size[0], 'height': map_size[1], 'obs_perc': obs_perc, 'map_type': map_type}
                        
                        exploration_history = run_single_evaluation_episode(planner_name, planner_instance, env_config, seed)
                        steps, explored_perc = zip(*exploration_history)
                        all_results[config_key]['steps'].append(list(steps))
                        all_results[config_key]['explored'].append(list(explored_perc))
                        print(f"    Run {i+1} finished. Steps: {steps[-1]}, Final Exploration: {explored_perc[-1]:.2f}%")

    if not all_results: print("No results to plot."); return

    # --- Plotting logic (using average_results, same as your original main_simulation.py) ---
    def average_results(list_of_explored_runs, list_of_steps_runs, debug_name="UNSPECIFIED"):
        # ... (This is the same robust average_results function from your main_simulation.py)
        valid_explored_runs = [r for r in list_of_explored_runs if r and len(r) > 0]
        valid_steps_runs = [r for r in list_of_steps_runs if r and len(r) > 0]
        if not valid_explored_runs or not valid_steps_runs or len(valid_explored_runs) != len(valid_steps_runs):
            return np.array([]), np.array([]), np.array([])
        max_step_all = 0
        for r_steps in valid_steps_runs: 
            if r_steps : max_step_all = max(max_step_all, r_steps[-1])
        if max_step_all == 0:
            if all(len(r) == 1 and r[0] == 0 for r in valid_steps_runs) and all(len(r) == 1 for r in valid_explored_runs):
                vals = [r[0] for r in valid_explored_runs if r]
                return np.array([0]), np.array([np.mean(vals) if vals else np.nan]), np.array([np.std(vals) if vals else np.nan])
            return np.array([]), np.array([]), np.array([])
        common_steps_axis_avg = np.arange(max_step_all + 1)
        aligned_explored_data_avg = []
        for i_avg_res in range(len(valid_explored_runs)):
            s_avg, e_avg = valid_steps_runs[i_avg_res], valid_explored_runs[i_avg_res]
            if len(s_avg) != len(e_avg) or not s_avg: continue 
            l_avg = e_avg[0] if s_avg[0] == 0 else np.nan 
            us_avg, ui_avg = np.unique(s_avg, return_index=True); ue_avg = np.array(e_avg)[ui_avg]
            if not common_steps_axis_avg.any():
                if us_avg.size == 1 and us_avg[0] == 0 : aligned_explored_data_avg.append(ue_avg)
                continue
            interp_vals_avg = np.interp(common_steps_axis_avg, us_avg, ue_avg, left=l_avg, right=ue_avg[-1] if ue_avg.size > 0 else np.nan)
            aligned_explored_data_avg.append(interp_vals_avg)
        if not aligned_explored_data_avg: 
            if common_steps_axis_avg.size == 1 and common_steps_axis_avg[0] == 0:
                vals_init = []
                for r_init_list_main in valid_explored_runs: 
                    if r_init_list_main and len(r_init_list_main) > 0 : 
                        vals_init.append(r_init_list_main[0]) 
                if vals_init: return np.array([0]), np.array([np.mean(vals_init)]), np.array([np.std(vals_init)])
            return np.array([]), np.array([]), np.array([])
        return common_steps_axis_avg, np.nanmean(np.array(aligned_explored_data_avg), axis=0), np.nanstd(np.array(aligned_explored_data_avg), axis=0)


    for map_size in EVAL_MAP_SIZES:
        for obs_perc in EVAL_OBSTACLE_PERCENTAGES:
            for map_type in EVAL_MAP_TYPES:
                plt.figure(figsize=(12, 8))
                map_size_str = f"{map_size[0]}x{map_size[1]}"
                fig_title = (f"Planner Comparison on {map_type} Map\n"
                             f"Size: {map_size_str}, Obstacle Rate: {obs_perc:.2f}, Sensor Range: {ROBOT_SENSOR_RANGE_EVAL}")
                plt.title(fig_title, fontsize=12)
                
                plotted_anything = False
                for planner_name in PLANNERS_TO_EVALUATE:
                    config_key = (planner_name, map_size_str, obs_perc, map_type)
                    if config_key in all_results and all_results[config_key]['steps']:
                        plot_data = all_results[config_key]
                        x, y_mean, y_std = average_results(plot_data['explored'], plot_data['steps'])
                        if x.size > 0:
                            plot_colors = {'DeepACOSimple': 'magenta', 'FBE': 'blue', 'IGE': 'green', 'URE': 'purple', 'OriginalACO': 'red'}
                            plt.plot(x, y_mean, label=planner_name, color=plot_colors.get(planner_name, 'black'))
                            plt.fill_between(x, y_mean-y_std, y_mean+y_std, color=plot_colors.get(planner_name, 'black'), alpha=0.1)
                            plotted_anything = True

                if plotted_anything:
                    plt.xlabel("Physical Steps"); plt.ylabel("Explored Area (%)")
                    plt.legend(); plt.grid(True); plt.ylim(0, 105); plt.xlim(left=0)
                    plot_filename = os.path.join(RESULTS_DIR_EVAL, f"comparison_{map_size_str}_{map_type}_obs{obs_perc:.2f}.png")
                    plt.savefig(plot_filename)
                    print(f"Comparison plot saved to: {plot_filename}")
                    plt.close()

if __name__ == "__main__":
    if VISUALIZE_EVALUATION and not pygame.get_init():
        pygame.init()
    try:
        evaluate_all_planners()
    except Exception as e:
        print(f"An error occurred during evaluation script: {e}")
        
    finally:
        if pygame.get_init():
            pygame.quit()