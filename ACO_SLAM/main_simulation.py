# main_simulation.py
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time

from environment import Environment, UNKNOWN
from robot import Robot
from planners.fbe_planner import FBEPlanner
from planners.aco_planner import ACOPlanner
from visualizer import Visualizer

# Simulation Parameters
ENV_WIDTH = 40
ENV_HEIGHT = 30
OBSTACLE_PERCENTAGE = 0.25
ROBOT_START_POS = (ENV_HEIGHT // 2, ENV_WIDTH // 2)
ROBOT_SENSOR_RANGE = 4 # Half-width of square sensor
MAX_STEPS = 200
CELL_SIZE = 15 # For visualization
SIM_DELAY = 0.05 # Seconds, for visualization speed

# ACO Parameters (can be tuned)
N_ANTS = 15
N_ITERATIONS = 25
ALPHA = 1.0  # Pheromone influence
BETA = 3.0   # Heuristic influence (higher beta gives more weight to distance/IG)
EVAPORATION_RATE = 0.3


def run_simulation(planner_type, env_seed=None, show_visualization=True):
    """Runs a single simulation with the specified planner."""
    if env_seed is not None:
        np.random.seed(env_seed) # For reproducible environments

    env = Environment(ENV_WIDTH, ENV_HEIGHT, OBSTACLE_PERCENTAGE)
    robot = Robot(ROBOT_START_POS, ROBOT_SENSOR_RANGE)

    if planner_type == "FBE":
        planner = FBEPlanner(env)
    elif planner_type == "ACO":
        planner = ACOPlanner(env, n_ants=N_ANTS, n_iterations=N_ITERATIONS,
                             alpha=ALPHA, beta=BETA, evaporation_rate=EVAPORATION_RATE)
    else:
        raise ValueError("Unknown planner type")

    if show_visualization:
        visualizer = Visualizer(ENV_WIDTH, ENV_HEIGHT, CELL_SIZE)

    # Initial sense
    robot.sense(env)

    explored_percentages = []
    steps_taken = []
    total_explorable_area = env.get_total_explorable_area()
    if total_explorable_area == 0: total_explorable_area = 1 # Avoid division by zero

    current_target = None

    for step in range(MAX_STEPS):
        if show_visualization:
            if not visualizer.handle_events(): # Handle QUIT event
                break
            visualizer.clear_screen()
            visualizer.draw_environment(env.grid_known)
            
            # Optionally show frontiers for debugging
            # frontiers = planner.find_frontiers(env.get_known_map_for_planner())
            # visualizer.draw_frontiers(frontiers)

            visualizer.draw_path(robot.path_taken)
            visualizer.draw_robot(robot.get_position())
            if current_target:
                visualizer.draw_target(current_target)
            
            explored_area = env.get_explored_area()
            percent_explored = (explored_area / total_explorable_area) * 100
            visualizer.draw_text(f"Planner: {planner_type}", (10, 10))
            visualizer.draw_text(f"Step: {step}/{MAX_STEPS}", (10, 30))
            visualizer.draw_text(f"Explored: {percent_explored:.2f}%", (10, 50))
            visualizer.update_display()
            time.sleep(SIM_DELAY)

        # Planning
        known_map_for_planner = env.get_known_map_for_planner()
        current_target = planner.plan_next_action(robot.get_position(), known_map_for_planner)

        if current_target is None:
            print(f"{planner_type}: No more frontiers or cannot plan. Exploration ended at step {step}.")
            break
        
        # Movement
        robot.move_to(current_target)
        
        # Sensing
        robot.sense(env)

        # Record metrics
        explored_area = env.get_explored_area()
        percent_explored = (explored_area / total_explorable_area) * 100
        explored_percentages.append(percent_explored)
        steps_taken.append(step + 1)

        if percent_explored >= 99.9: # Consider fully explored
            print(f"{planner_type}: Environment almost fully explored at step {step}.")
            break
    
    if show_visualization:
        visualizer.quit()
    
    return steps_taken, explored_percentages


if __name__ == "__main__":
    num_runs = 3 # Number of runs for averaging results (due to randomness in env and ACO)
    env_master_seed = 42 # Use a master seed for reproducibility of environments across runs
    
    all_results_fbe = {'steps': [], 'explored': []}
    all_results_aco = {'steps': [], 'explored': []}

    print("Running FBE simulations...")
    for i in range(num_runs):
        current_env_seed = env_master_seed + i 
        print(f"  FBE Run {i+1}/{num_runs} with env_seed {current_env_seed}")
        steps, explored = run_simulation("FBE", env_seed=current_env_seed, show_visualization=(i==0)) # Show viz for first run
        all_results_fbe['steps'].append(steps)
        all_results_fbe['explored'].append(explored)
        if not steps: # If exploration ended prematurely (e.g. no frontiers)
            print(f"    Warning: FBE Run {i+1} ended prematurely or found no explorable area.")


    print("\nRunning ACO simulations...")
    for i in range(num_runs):
        current_env_seed = env_master_seed + i
        print(f"  ACO Run {i+1}/{num_runs} with env_seed {current_env_seed}")
        steps, explored = run_simulation("ACO", env_seed=current_env_seed, show_visualization=(i==0)) # Show viz for first run
        all_results_aco['steps'].append(steps)
        all_results_aco['explored'].append(explored)
        if not steps:
            print(f"    Warning: ACO Run {i+1} ended prematurely or found no explorable area.")

    # --- Plotting Results ---
    plt.figure(figsize=(12, 7))

    # Helper function to average results, handling lists of different lengths
    def average_results(results_list):
        if not any(results_list): # If all runs were empty
            return [], []
            
        max_len = 0
        for r in results_list:
            if r: max_len = max(max_len, len(r))
        
        if max_len == 0: return [], [] # Still no data

        # Pad shorter runs with their last value
        padded_results = []
        for r_item in results_list:
            if not r_item: # Skip empty runs
                # Or fill with zeros up to max_len if preferred
                # padded_results.append(np.zeros(max_len)) 
                continue
            
            last_val = r_item[-1] if r_item else 0
            padded_item = np.pad(r_item, (0, max_len - len(r_item)), 'constant', constant_values=last_val)
            padded_results.append(padded_item)
        
        if not padded_results: return [], np.arange(1,1) # No valid data after padding

        avg_values = np.mean(padded_results, axis=0)
        std_values = np.std(padded_results, axis=0)
        steps_axis = np.arange(1, max_len + 1)
        return steps_axis, avg_values, std_values


    # FBE Plotting
    if any(all_results_fbe['explored']):
        steps_fbe, avg_explored_fbe, std_explored_fbe = average_results(all_results_fbe['explored'])
        if len(steps_fbe) > 0:
            plt.plot(steps_fbe, avg_explored_fbe, label=f"FBE (Avg over {num_runs} runs)", color='blue')
            plt.fill_between(steps_fbe, avg_explored_fbe - std_explored_fbe, avg_explored_fbe + std_explored_fbe,
                             color='blue', alpha=0.2)
    else:
        print("No valid data for FBE to plot.")


    # ACO Plotting
    if any(all_results_aco['explored']):
        steps_aco, avg_explored_aco, std_explored_aco = average_results(all_results_aco['explored'])
        if len(steps_aco) > 0:
            plt.plot(steps_aco, avg_explored_aco, label=f"ACO (Avg over {num_runs} runs)", color='red')
            plt.fill_between(steps_aco, avg_explored_aco - std_explored_aco, avg_explored_aco + std_explored_aco,
                             color='red', alpha=0.2)
    else:
        print("No valid data for ACO to plot.")


    plt.xlabel("Number of Steps")
    plt.ylabel("Percentage of Area Explored (%)")
    plt.title("Exploration Performance: ACO vs. FBE")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 105) # Exploration percentage up to 100%
    plt.xlim(0, MAX_STEPS)
    plt.tight_layout()
    plt.savefig("exploration_comparison.png")
    print("\nComparison plot saved as exploration_comparison.png")
    plt.show()