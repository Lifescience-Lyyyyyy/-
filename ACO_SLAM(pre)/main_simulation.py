# main_simulation.py
import pygame
import numpy as np
import matplotlib
# --- NEW: Set matplotlib backend for non-GUI environments ---
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os # Import os for creating directories

from environment import Environment, UNKNOWN, OBSTACLE, FREE
from robot import Robot
from planners.fbe_planner import FBEPlanner
from planners.aco_planner import ACOPlanner
from planners.ige_planner import IGEPlanner
from planners.ure_planner import UREPlanner
from visualizer import Visualizer

# --- NEW: Global switch for cluster mode ---
LINUX_CLUSTER_MODE = False # SET TO True TO RUN ON A CLUSTER (NO GUI)

# --- Simulation Mode Control (only relevant if not in cluster mode) ---
PAINT_MODE = True 
LOAD_MAP_FROM_FILE = False
MAP_FILENAME = "custom_map.npy"

# --- Simulation Parameters ---
ENV_WIDTH = 300
ENV_HEIGHT = 200
OBSTACLE_PERCENTAGE = 0.30
ROBOT_START_POS = (ENV_HEIGHT // 2, ENV_WIDTH // 2)
ROBOT_SENSOR_RANGE = 5
MAX_PHYSICAL_STEPS = 1500
MAX_PLANNING_CYCLES = 500
CELL_SIZE = 5
# Delays are now conditional
SIM_DELAY_PHYSICAL_STEP = 0.01 if not LINUX_CLUSTER_MODE else 0
SIM_DELAY_PLANNING_PHASE = 0.02 if not LINUX_CLUSTER_MODE else 0
VISUALIZE_ANT_ITERATIONS = False 

# --- ACO Parameters ---
ACO_N_ANTS = 20; ACO_N_ITERATIONS = 30; ACO_ALPHA = 1.0; ACO_BETA = 3.0
ACO_EVAPORATION_RATE = 0.25; ACO_Q0 = 0.7; ACO_IG_WEIGHT_HEURISTIC = 2.5
ACO_SENSOR_HEURISTIC_RANGE = ROBOT_SENSOR_RANGE

# The painter and ant visualization are disabled in cluster mode
def create_map_with_painter(width, height, cell_size, filename):
    if LINUX_CLUSTER_MODE:
        print("Painter mode is disabled on Linux cluster.")
        return None
    # ... (rest of the function is the same, but will only run if not in cluster mode)
    viz = Visualizer(width, height, cell_size)
    custom_map = np.full((height, width), FREE)
    start_r, start_c = ROBOT_START_POS
    if 0 <= start_r < height and 0 <= start_c < width: custom_map[start_r, start_c] = FREE
    painting = True
    print("\n--- MAP PAINTER MODE ---")
    while painting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return None
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN: painting = False
        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] or mouse_buttons[2]:
            try:
                mx, my = pygame.mouse.get_pos()
                c, r = mx // cell_size, my // cell_size
                if (r, c) != ROBOT_START_POS:
                    if mouse_buttons[0]: custom_map[r, c] = OBSTACLE
                    elif mouse_buttons[2]: custom_map[r, c] = FREE
            except IndexError: pass
        viz.clear_screen(); viz.draw_environment(custom_map); viz.draw_robot(ROBOT_START_POS)
        viz.draw_text("PAINT MODE", (10, 10), color=(255,0,0))
        viz.update_display(); time.sleep(0.01)
    np.save(filename, custom_map)
    print(f"Map saved to '{filename}'.")
    return custom_map

def visualize_ant_paths_for_aco(robot_pos, known_map, all_ant_paths_iterations, environment_obj, visualizer_obj):
    # This function is entirely disabled in cluster mode
    if LINUX_CLUSTER_MODE or not VISUALIZE_ANT_ITERATIONS: return
    # ... (rest of the function is the same)
    if visualizer_obj:
        for i, paths_this_iter in enumerate(all_ant_paths_iterations):
            ret = visualizer_obj.handle_events()
            if not ret: return
            if ret == "SKIP_ANTS": return
            visualizer_obj.draw_ant_paths_iteration(robot_pos, known_map, paths_this_iter, environment_obj, i, len(all_ant_paths_iterations))

def run_simulation(planner_type, env, show_visualization=True, run_index=0, visualizer=None):
    # Disable visualization if in cluster mode
    if LINUX_CLUSTER_MODE:
        show_visualization = False

    robot = Robot(env.robot_start_pos_ref, ROBOT_SENSOR_RANGE)
    ant_viz_callback = None
    if planner_type == "ACO" and show_visualization and visualizer:
        ant_viz_callback = lambda rp, km, api, eo: visualize_ant_paths_for_aco(rp, km, api, eo, visualizer)
    
    planner = {"FBE": FBEPlanner(env), "ACO": ACOPlanner(env, n_ants=ACO_N_ANTS, n_iterations=ACO_N_ITERATIONS, alpha=ACO_ALPHA, beta=ACO_BETA, evaporation_rate=ACO_EVAPORATION_RATE, q0=ACO_Q0, ig_weight_heuristic=ACO_IG_WEIGHT_HEURISTIC, robot_sensor_range_for_heuristic=ACO_SENSOR_HEURISTIC_RANGE, visualize_ants_callback=ant_viz_callback), "IGE": IGEPlanner(env, ROBOT_SENSOR_RANGE), "URE": UREPlanner(env, ROBOT_SENSOR_RANGE)}.get(planner_type)
    if not planner: raise ValueError(f"Unknown planner type: {planner_type}")

    explored_percentages, physical_steps = [], []
    total_phys_steps, total_plan_cycles = 0, 0
    total_explorable = env.get_total_explorable_area() or 1.0
    sim_active = True
    path_segment, target = [], None
    robot.sense(env)
    if planner_type == "URE": planner.update_observation_counts(robot.get_position(), env.grid_known)
    explored_percentages.append((env.get_explored_area() / total_explorable) * 100)
    physical_steps.append(0)

    while total_phys_steps < MAX_PHYSICAL_STEPS and total_plan_cycles < MAX_PLANNING_CYCLES and sim_active:
        if (env.get_explored_area() / total_explorable) * 100 >= 99.9: break
        
        # --- Conditional visualization block ---
        if show_visualization and visualizer:
            if not visualizer.handle_events(): sim_active = False; break
            visualizer.clear_screen(); visualizer.draw_environment(env.grid_known)
            visualizer.draw_robot_path(robot.actual_pos_history)
            visualizer.draw_robot(robot.get_position())
            if target: visualizer.draw_target(target)
            visualizer.draw_text(f"{planner_type}", (10, 10))
            visualizer.update_display()
            time.sleep(SIM_DELAY_PHYSICAL_STEP)
        
        if not path_segment:
            if show_visualization and planner_type == "ACO" and visualizer:
                time.sleep(SIM_DELAY_PLANNING_PHASE)
            target, path = planner.plan_next_action(robot.get_position(), env.get_known_map_for_planner())
            total_plan_cycles +=1
            if target is None or not path: sim_active = False; break
            path_segment = path[1:]
        
        if path_segment:
            moved = robot.attempt_move_one_step(path_segment[0], env)
            total_phys_steps += 1
            if moved: path_segment.pop(0)
            else: path_segment = []
            robot.sense(env)
            if planner_type == "URE": planner.update_observation_counts(robot.get_position(), env.grid_known)
            explored_percentages.append((env.get_explored_area() / total_explorable) * 100)
            physical_steps.append(total_phys_steps)
        else: sim_active = False

    return physical_steps, explored_percentages

def average_results(list_of_explored_runs, list_of_steps_runs):
    # ... (this function remains unchanged)
    if not list_of_explored_runs or not list_of_steps_runs: return np.array([]), np.array([]), np.array([])
    max_step_all = 0
    for steps_run in list_of_steps_runs:
        if steps_run: max_step_all = max(max_step_all, steps_run[-1])
    common_steps_axis = np.arange(max_step_all + 1)
    if len(common_steps_axis) == 0: return np.array([]), np.array([]), np.array([])
    aligned_explored_data = []
    for i in range(len(list_of_explored_runs)):
        steps_run, explored_run = list_of_steps_runs[i], list_of_explored_runs[i]
        if len(steps_run) != len(explored_run) or not steps_run: continue
        interpolated_values = np.interp(common_steps_axis, steps_run, explored_run, left=explored_run[0], right=explored_run[-1])
        aligned_explored_data.append(interpolated_values)
    if not aligned_explored_data: return np.array([]), np.array([]), np.array([])
    aligned_array = np.array(aligned_explored_data)
    avg_explored = np.nanmean(aligned_array, axis=0)
    std_explored = np.nanstd(aligned_array, axis=0)
    return common_steps_axis, avg_explored, std_explored

def plot_performance_graph(all_results, num_runs, map_info_title):
    plt.figure(figsize=(15, 10))
    plotted_anything = False
    plot_colors = {'FBE': 'blue', 'ACO': 'red', 'IGE': 'green', 'URE': 'purple'}
    
    for planner_name, results_data in all_results.items():
        if results_data and results_data['explored']:
            x_axis, avg_explored, std_explored = average_results(results_data['explored'], results_data['steps'])
            if x_axis.size > 0 and avg_explored.size > 0:
                plt.plot(x_axis, avg_explored, label=planner_name, color=plot_colors.get(planner_name, 'black'))
                if num_runs > 1:
                    plt.fill_between(x_axis, avg_explored - std_explored, avg_explored + std_explored, color=plot_colors.get(planner_name, 'black'), alpha=0.1)
                plotted_anything = True
    
    if plotted_anything:
        plt.xlabel("Total Physical Steps Taken by Robot", fontsize=12)
        plt.ylabel("Percentage of Area Explored (%)", fontsize=12)
        title_suffix = f"Avg of {num_runs} Runs" if num_runs > 1 else "Single Run"
        title = f"Exploration Performance on {map_info_title}\n({title_suffix})"
        plt.title(title, fontsize=14)
        plt.legend(loc='best'); plt.grid(True, linestyle=':', alpha=0.6)
        plt.ylim(0, 105); plt.xlim(left=0)
        
        # --- Save plot to 'results' directory ---
        if not os.path.exists('results'):
            os.makedirs('results')
        plot_filename = f"results/exploration_comparison_{map_info_title.replace(' ', '_').lower()}.png"
        
        plt.savefig(plot_filename)
        print(f"\nPlot saved to '{plot_filename}'")
        
        # In cluster mode, we don't want to call plt.show() as it can block
        if not LINUX_CLUSTER_MODE:
            plt.show()
        
        plt.close() # Close the figure to free up memory
    else:
        print("\nWarning: No data was plotted.")

if __name__ == "__main__":
    # --- Conditional Pygame Initialization ---
    visualizer = None
    if not LINUX_CLUSTER_MODE:
        pygame.init()
        # In non-cluster mode, interactive modes are possible
        if PAINT_MODE:
            # Painter mode logic (runs with GUI)
            visualizer = Visualizer(ENV_WIDTH, ENV_HEIGHT, CELL_SIZE)
            custom_map_grid = create_map_with_painter(ENV_WIDTH, ENV_HEIGHT, CELL_SIZE, MAP_FILENAME)
            if custom_map_grid is not None:
                env = Environment(ENV_WIDTH, ENV_HEIGHT, map_type="custom", robot_start_pos_ref=ROBOT_START_POS, custom_grid=custom_map_grid)
                run_simulation(planner_type="FBE", env=env, show_visualization=True, visualizer=visualizer)

        elif LOAD_MAP_FROM_FILE:
            # Load map mode logic (runs with GUI and generates a plot)
            try:
                loaded_map_grid = np.load(MAP_FILENAME)
                all_results = {}
                planners_to_run = ["FBE", "ACO", "IGE", "URE"]
                visualizer = Visualizer(ENV_WIDTH, ENV_HEIGHT, CELL_SIZE)
                for planner_name in planners_to_run:
                    all_results[planner_name] = {'steps': [], 'explored': []}
                    env_for_run = Environment(ENV_WIDTH, ENV_HEIGHT, map_type="custom", custom_grid=np.copy(loaded_map_grid), robot_start_pos_ref=ROBOT_START_POS)
                    print(f"\n>>> Running with {planner_name.upper()} on loaded map <<<")
                    steps, explored = run_simulation(planner_name, env=env_for_run, show_visualization=True, visualizer=visualizer)
                    if steps and explored:
                        all_results[planner_name]['steps'].append(steps)
                        all_results[planner_name]['explored'].append(explored)
                plot_performance_graph(all_results, num_runs=1, map_info_title=f"Custom Map ({MAP_FILENAME})")
            except FileNotFoundError:
                print(f"ERROR: Map file '{MAP_FILENAME}' not found.")

        else:
            # Standard GUI mode (multiple runs with plot)
            num_runs = 3
            env_master_seed = 42
            all_results = {}
            planners_to_run = ["FBE", "ACO", "IGE", "URE"]
            visualizer = Visualizer(ENV_WIDTH, ENV_HEIGHT, CELL_SIZE)
            for planner_name in planners_to_run:
                all_results[planner_name] = {'steps': [], 'explored': []}
                for i in range(num_runs):
                    np.random.seed(env_master_seed + i)
                    env = Environment(ENV_WIDTH, ENV_HEIGHT, OBSTACLE_PERCENTAGE, map_type="random", robot_start_pos_ref=ROBOT_START_POS)
                    steps, explored = run_simulation(planner_name, env=env, show_visualization=True, run_index=i, visualizer=visualizer)
                    if steps and explored:
                        all_results[planner_name]['steps'].append(steps)
                        all_results[planner_name]['explored'].append(explored)
            map_details = f"Random Maps ({ENV_WIDTH}x{ENV_HEIGHT})"
            plot_performance_graph(all_results, num_runs=num_runs, map_info_title=map_details)

    else:
        # --- LINUX CLUSTER MODE LOGIC ---
        print("--- Running in Linux Cluster Mode (No GUI) ---")
        num_runs = 10 # Increase runs for better statistics on the cluster
        env_master_seed = 42
        all_results = {}
        planners_to_run = ["FBE", "ACO", "IGE", "URE"]
        
        for planner_name in planners_to_run:
            all_results[planner_name] = {'steps': [], 'explored': []}
            print(f"\n>>> Running {planner_name.upper()} for {num_runs} episodes <<<")
            for i in range(num_runs):
                current_env_seed = env_master_seed + i
                np.random.seed(current_env_seed) # Ensure reproducible maps
                print(f"  Run {i + 1}/{num_runs} with env_seed {current_env_seed}...")
                
                env_for_run = Environment(ENV_WIDTH, ENV_HEIGHT, OBSTACLE_PERCENTAGE, map_type="random", robot_start_pos_ref=ROBOT_START_POS)
                
                # `run_simulation` is called with show_visualization=False internally
                physical_steps, explored_over_phys_steps = run_simulation(planner_name, env=env_for_run)
                
                if physical_steps and explored_over_phys_steps:
                    all_results[planner_name]['steps'].append(physical_steps)
                    all_results[planner_name]['explored'].append(explored_over_phys_steps)

        # Plotting is still done at the end
        map_details = f"Random Maps ({ENV_WIDTH}x{ENV_HEIGHT}, {num_runs} runs)"
        plot_performance_graph(all_results, num_runs=num_runs, map_info_title=map_details)


    # --- Conditional Pygame Quit ---
    if not LINUX_CLUSTER_MODE:
        pygame.quit()
    
    print("\n--- Simulation Finished ---")