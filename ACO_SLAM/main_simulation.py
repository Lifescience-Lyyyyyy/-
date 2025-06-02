# main_simulation.py
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time

from environment import Environment, UNKNOWN, OBSTACLE, FREE
from robot import Robot
from planners.fbe_planner import FBEPlanner
from planners.aco_planner import ACOPlanner
from planners.ige_planner import IGEPlanner # Ensure this planner file has robust fallbacks
from planners.ure_planner import UREPlanner   # Ensure this planner file has robust fallbacks
from visualizer import Visualizer

# Simulation Parameters (Adjust as needed for different scenarios)
ENV_WIDTH = 50
ENV_HEIGHT = 40
OBSTACLE_PERCENTAGE = 0.30
ROBOT_START_POS = (ENV_HEIGHT // 2, ENV_WIDTH // 2)
ROBOT_SENSOR_RANGE = 5
MAX_PHYSICAL_STEPS = 1500
MAX_PLANNING_CYCLES = 500
CELL_SIZE = 12
SIM_DELAY_PHYSICAL_STEP = 0.01
SIM_DELAY_PLANNING_PHASE = 0.02
VISUALIZE_ANT_ITERATIONS = False

# ACO Parameters
ACO_N_ANTS = 20; ACO_N_ITERATIONS = 30; ACO_ALPHA = 1.0; ACO_BETA = 3.0
ACO_EVAPORATION_RATE = 0.25; ACO_Q0 = 0.7; ACO_IG_WEIGHT_HEURISTIC = 2.5
ACO_SENSOR_HEURISTIC_RANGE = ROBOT_SENSOR_RANGE

_skip_ant_visualization_for_this_run = False

def visualize_ant_paths_for_aco(robot_pos, known_map, all_ant_paths_iterations, environment_obj, visualizer_obj):
    global _skip_ant_visualization_for_this_run
    if not VISUALIZE_ANT_ITERATIONS or _skip_ant_visualization_for_this_run: return
    if visualizer_obj:
        total_iters = len(all_ant_paths_iterations)
        for i, paths_this_iter in enumerate(all_ant_paths_iterations):
            ret = visualizer_obj.handle_events();
            if not ret: pygame.quit(); exit()
            if ret == "SKIP_ANTS": _skip_ant_visualization_for_this_run = True; print("Skipping ACO ant viz."); return
            visualizer_obj.draw_ant_paths_iteration(robot_pos, known_map, paths_this_iter, environment_obj, i, total_iters)

def run_simulation(planner_type, env_seed=None, show_visualization=True, run_index=0):
    global _skip_ant_visualization_for_this_run
    _skip_ant_visualization_for_this_run = False
    if env_seed is not None: np.random.seed(env_seed)

    _actual_start_pos_to_use = ROBOT_START_POS
    map_type_to_use = "random"
    env = Environment(ENV_WIDTH, ENV_HEIGHT, OBSTACLE_PERCENTAGE,
                      map_type=map_type_to_use,
                      robot_start_pos_ref=_actual_start_pos_to_use)

    if env.grid_true[_actual_start_pos_to_use[0], _actual_start_pos_to_use[1]] == OBSTACLE:
        found_free_start = False
        for r_off in range(-3, 4):
            for c_off in range(-3, 4):
                if not found_free_start:
                    nr, nc = _actual_start_pos_to_use[0] + r_off, _actual_start_pos_to_use[1] + c_off
                    if env.is_within_bounds(nr, nc) and env.grid_true[nr, nc] == FREE:
                        _actual_start_pos_to_use = (nr, nc); found_free_start = True; break
        if not found_free_start: print(f"ERROR: No free start near {ROBOT_START_POS} for {planner_type}."); return [], []
        # print(f"Adjusted start for {planner_type} to {_actual_start_pos_to_use}")

    robot = Robot(_actual_start_pos_to_use, ROBOT_SENSOR_RANGE)
    visualizer = None
    if show_visualization: visualizer = Visualizer(ENV_WIDTH, ENV_HEIGHT, CELL_SIZE)

    ant_viz_callback = None
    if planner_type == "ACO" and show_visualization and VISUALIZE_ANT_ITERATIONS:
        ant_viz_callback = lambda rp, km, api, eo: visualize_ant_paths_for_aco(rp, km, api, eo, visualizer)

    planner = None
    if planner_type == "FBE": planner = FBEPlanner(env)
    elif planner_type == "ACO":
        planner = ACOPlanner(env, n_ants=ACO_N_ANTS, n_iterations=ACO_N_ITERATIONS, alpha=ACO_ALPHA, beta=ACO_BETA,
                             evaporation_rate=ACO_EVAPORATION_RATE, q0=ACO_Q0,
                             ig_weight_heuristic=ACO_IG_WEIGHT_HEURISTIC,
                             robot_sensor_range_for_heuristic=ACO_SENSOR_HEURISTIC_RANGE,
                             visualize_ants_callback=ant_viz_callback)
    elif planner_type == "IGE": planner = IGEPlanner(env, ROBOT_SENSOR_RANGE)
    elif planner_type == "URE": planner = UREPlanner(env, ROBOT_SENSOR_RANGE)
    else: raise ValueError(f"Unknown planner type: {planner_type}")

    explored_percentages_over_physical_steps = []
    physical_step_counts = []
    total_physical_steps_taken = 0
    total_planning_cycles = 0
    total_explorable_area = env.get_total_explorable_area()
    if total_explorable_area == 0: total_explorable_area = 1.0

    sim_active = True
    current_intended_path_segment = []
    current_ultimate_frontier_target = None

    robot.sense(env)
    if planner_type == "URE" and planner:
        planner.update_observation_counts(robot.get_position(), env.grid_known)
    
    initial_explored = (env.get_explored_area() / total_explorable_area) * 100
    explored_percentages_over_physical_steps.append(initial_explored)
    physical_step_counts.append(0)

    while total_physical_steps_taken < MAX_PHYSICAL_STEPS and \
          total_planning_cycles < MAX_PLANNING_CYCLES and \
          sim_active:

        current_explored_percentage = (env.get_explored_area() / total_explorable_area) * 100
        if current_explored_percentage >= 99.9: break

        if show_visualization:
            sim_event_ret = visualizer.handle_events()
            if not sim_event_ret: sim_active = False; break
            if sim_event_ret == "SKIP_ANTS" and planner_type == "ACO" and not current_intended_path_segment:
                _skip_ant_visualization_for_this_run = True
            
            visualizer.clear_screen(); visualizer.draw_environment(env.grid_known)
            visualizer.draw_robot_path(robot.actual_pos_history) # Draw actual walked path
            visualizer.draw_robot(robot.get_position())
            if current_ultimate_frontier_target: visualizer.draw_target(current_ultimate_frontier_target)
            
            visualizer.draw_text(f"{planner_type}", (10, 10))
            visualizer.draw_text(f"Phys Steps: {total_physical_steps_taken}/{MAX_PHYSICAL_STEPS}", (10, 30))
            visualizer.draw_text(f"Plan Cycles: {total_planning_cycles}/{MAX_PLANNING_CYCLES}", (10,50))
            visualizer.draw_text(f"Explored: {current_explored_percentage:.2f}%", (10, 70))
            if planner_type == "ACO" and VISUALIZE_ANT_ITERATIONS and not current_intended_path_segment:
                visualizer.draw_text("SPACE: skip ant viz", (10, visualizer.screen_height - 50), small=True)
            visualizer.update_display()
            if not (planner_type == "ACO" and VISUALIZE_ANT_ITERATIONS and not _skip_ant_visualization_for_this_run and not current_intended_path_segment):
                time.sleep(SIM_DELAY_PHYSICAL_STEP)

        if not current_intended_path_segment: 
            if show_visualization and planner_type == "ACO" and VISUALIZE_ANT_ITERATIONS and not _skip_ant_visualization_for_this_run:
                time.sleep(SIM_DELAY_PLANNING_PHASE)

            known_map_for_planner = env.get_known_map_for_planner()
            target_pos, path_segment = None, None
            
            # Uniform way to call planners, FBE's specific history logic is internal to its plan_next_action
            planner_kwargs = {}
            if planner_type == "FBE" and hasattr(planner, 'plan_next_action') and 'robot_path_history' in planner.plan_next_action.__code__.co_varnames:
                 planner_kwargs['robot_path_history'] = [tuple(p) for p in reversed(robot.actual_pos_history)]
            
            target_pos, path_segment = planner.plan_next_action(
                robot.get_position(), known_map_for_planner, **planner_kwargs
            )

            total_planning_cycles +=1
            _skip_ant_visualization_for_this_run = False

            if target_pos is None or not path_segment:
                print(f"SIM END ({planner_type}): Planner returned NO valid target/path. Phys_step: {total_physical_steps_taken}, Plan_cycle: {total_planning_cycles}")
                sim_active = False; break 
            
            current_ultimate_frontier_target = target_pos 
            current_intended_path_segment = path_segment[1:]

        if current_intended_path_segment:
            next_step_candidate = current_intended_path_segment[0]
            moved_one_cell = robot.attempt_move_one_step(next_step_candidate, env)
            total_physical_steps_taken += 1

            if moved_one_cell:
                current_intended_path_segment.pop(0) 
                robot.sense(env)
                if planner_type == "URE" and planner:
                    planner.update_observation_counts(robot.get_position(), env.grid_known)
            else: 
                # print(f"SIM DEBUG ({planner_type}): Robot hit obstacle at/before {next_step_candidate}. Invalidating path.")
                current_intended_path_segment = [] 
                current_ultimate_frontier_target = None 
                robot.sense(env) 
                if planner_type == "URE" and planner:
                     planner.update_observation_counts(robot.get_position(), env.grid_known)
            
            new_explored = (env.get_explored_area() / total_explorable_area) * 100
            explored_percentages_over_physical_steps.append(new_explored)
            physical_step_counts.append(total_physical_steps_taken)
        else: 
            print(f"SIM CRITICAL ({planner_type}): No path segment to follow. Phys_step: {total_physical_steps_taken}")
            sim_active = False; break
            
    final_explored_percentage = (env.get_explored_area() / total_explorable_area) * 100
    if sim_active :
        if total_physical_steps_taken >= MAX_PHYSICAL_STEPS :
             print(f"SIM END ({planner_type}): Max phys steps {MAX_PHYSICAL_STEPS} reached. Explored: {final_explored_percentage:.2f}%")
        elif total_planning_cycles >= MAX_PLANNING_CYCLES:
             print(f"SIM END ({planner_type}): Max plan cycles {MAX_PLANNING_CYCLES} reached. Explored: {final_explored_percentage:.2f}%")
        else: 
             print(f"SIM END ({planner_type}): Natural exploration end. Phys_steps: {total_physical_steps_taken}. Explored: {final_explored_percentage:.2f}%")

    if sim_active and (not physical_step_counts or physical_step_counts[-1] != total_physical_steps_taken):
        explored_percentages_over_physical_steps.append(final_explored_percentage)
        physical_step_counts.append(total_physical_steps_taken)

    if show_visualization and visualizer:
        if sim_active:
            finish_text_y = visualizer.screen_height // 2
            if planner_type == "ACO" and VISUALIZE_ANT_ITERATIONS: finish_text_y = visualizer.screen_height // 2 - 20
            visualizer.draw_text(f"FINISHED: {planner_type}", (visualizer.screen_width // 2 - 70, finish_text_y))
            visualizer.update_display(); time.sleep(0.3) # Shorter delay
        if visualizer: visualizer.quit()
    return physical_step_counts, explored_percentages_over_physical_steps


if __name__ == "__main__":
    num_runs = 1 
    env_master_seed = 48 
    all_results = {}
    
    planners_to_run = ["FBE", "ACO", "IGE", "URE"] 
    
    plot_colors = {
        'FBE': 'blue', 'ACO': 'red', 'IGE': 'green', 'URE': 'purple'
    }

    print("--- STARTING SIMULATION RUNS ---")
    for planner_name in planners_to_run:
        all_results[planner_name] = {'steps': [], 'explored': []}
        print(f"\n>>> Running {planner_name.upper()} simulations <<<")
        for i in range(num_runs):
            current_env_seed = env_master_seed + i 
            print(f"  {planner_name} Run {i + 1}/{num_runs} with env_seed {current_env_seed}")
            show_viz_this_run = True 

            physical_steps, explored_over_phys_steps = run_simulation(
                planner_name, env_seed=current_env_seed,
                show_visualization=show_viz_this_run, run_index=i
            )
            all_results[planner_name]['steps'].append(physical_steps)
            all_results[planner_name]['explored'].append(explored_over_phys_steps)
            
            print(f"  {planner_name} Run {i + 1} FINISHED. Produced {len(physical_steps)} data points.")
            if len(physical_steps) <= 1 and not (len(physical_steps) == 1 and physical_steps[0] == 0) :
                print(f"    !!!! CRITICAL WARNING (run_simulation): {planner_name} Run {i + 1} got stuck or produced very little data. !!!!")

    print("\n\n--- AGGREGATED RAW DATA BEFORE AVERAGING ---")
    # ... (Raw data printing as before, it's helpful) ...
    for planner_name_debug in planners_to_run:
        if planner_name_debug in all_results:
            print(f"\nData for {planner_name_debug}:")
            if all_results[planner_name_debug]['steps'] and all_results[planner_name_debug]['steps'][0]:
                print(f"  Run 0 Steps ({len(all_results[planner_name_debug]['steps'][0])} pts): {all_results[planner_name_debug]['steps'][0][:5]}...{all_results[planner_name_debug]['steps'][0][-5:] if len(all_results[planner_name_debug]['steps'][0]) > 10 else all_results[planner_name_debug]['steps'][0]}")
            else: print(f"  Run 0 Steps: NO DATA or EMPTY LIST")
            if all_results[planner_name_debug]['explored'] and all_results[planner_name_debug]['explored'][0]:
                e_list_formatted = ['{:.1f}'.format(x) for x in all_results[planner_name_debug]['explored'][0]]
                print(f"  Run 0 Explored ({len(all_results[planner_name_debug]['explored'][0])} pts): {e_list_formatted[:5]}...{e_list_formatted[-5:] if len(e_list_formatted) > 10 else e_list_formatted}")
            else: print(f"  Run 0 Explored: NO DATA or EMPTY LIST")
        else: print(f"No results structure found for {planner_name_debug}")
    print("--- END AGGREGATED RAW DATA ---\n\n")

    plt.figure(figsize=(15, 10))

    def average_results(list_of_explored_runs, list_of_steps_runs, planner_name_for_debug="UNSPECIFIED"):
        # ... (average_results function from your last complete main_simulation.py, it seems robust enough) ...
        # print(f"DEBUG average_results CALLED FOR: {planner_name_for_debug}")
        valid_explored_runs = [r for r in list_of_explored_runs if r and len(r) > 0]
        valid_steps_runs = [r for r in list_of_steps_runs if r and len(r) > 0]
        if not valid_explored_runs or not valid_steps_runs or len(valid_explored_runs) != len(valid_steps_runs):
            # print(f"AverageResults WARN ({planner_name_for_debug}): Not enough valid/matching run data.")
            return np.array([]), np.array([]), np.array([])
        max_step_all = 0
        for steps_run in valid_steps_runs: max_step_all = max(max_step_all, steps_run[-1])
        common_steps_axis = np.arange(max_step_all + 1)
        if len(common_steps_axis) == 0: return np.array([]), np.array([]), np.array([])
        aligned_explored_data = []
        for i in range(len(valid_explored_runs)):
            steps_run = valid_steps_runs[i]; explored_run = valid_explored_runs[i]
            if len(steps_run) != len(explored_run): continue 
            interpolated_values = np.interp(common_steps_axis, steps_run, explored_run, left=explored_run[0], right=explored_run[-1])
            aligned_explored_data.append(interpolated_values)
        if not aligned_explored_data: return np.array([]), np.array([]), np.array([])
        avg_explored = np.nanmean(np.array(aligned_explored_data), axis=0)
        std_explored = np.nanstd(np.array(aligned_explored_data), axis=0)
        # print(f"AverageResults OK ({planner_name_for_debug}): Axis len {len(common_steps_axis)}, AvgExpl len {len(avg_explored)}")
        return common_steps_axis, avg_explored, std_explored


    max_x_lim_plot = 0; plotted_anything = False
    print("--- STARTING PLOTTING LOOP ---")
    for planner_name in planners_to_run:
        print(f"Attempting to plot for: {planner_name}")
        results_data = all_results.get(planner_name)
        
        if results_data and results_data.get('explored') and any(r for r_list in results_data.get('explored', []) for r in r_list): # Check if any data exists
            print(f"  Data found for {planner_name}. Calling average_results.")
            x_axis, avg_explored, std_explored = average_results(results_data['explored'], results_data['steps'], planner_name)
            
            if len(x_axis) > 0 and len(avg_explored) > 0 and not np.all(np.isnan(avg_explored)):
                print(f"  Plotting {planner_name}: x_axis len {len(x_axis)}, avg_explored len {len(avg_explored)}. Max x_axis val: {x_axis[-1] if len(x_axis)>0 else 'N/A'}")
                if x_axis[-1] > max_x_lim_plot: max_x_lim_plot = x_axis[-1]
                
                color_to_use = plot_colors.get(planner_name)
                if color_to_use is None:
                    print(f"  CRITICAL WARNING: No color defined for {planner_name} in plot_colors! Will use black.")
                    color_to_use = 'black'

                # --- MODIFICATION FOR IGE VISIBILITY ---
                linestyle_to_use = '-'
                linewidth_to_use = 1.5
                zorder_to_use = 1 # Default z-order, FBE and ACO will be here

                if planner_name == "IGE":
                    color_to_use = 'lime' # Brighter green
                    linestyle_to_use = '--' # Dashed line
                    linewidth_to_use = 2.5  # Thicker line
                    zorder_to_use = 10    # Draw IGE on TOP of everything
                elif planner_name == "URE":
                    zorder_to_use = 5     # URE below IGE but above FBE/ACO if they overlap
                elif planner_name == "ACO":
                    zorder_to_use = 3
                elif planner_name == "FBE":
                    zorder_to_use = 2


                plt.plot(x_axis, avg_explored, label=f"{planner_name}",
                         color=color_to_use, 
                         linewidth=linewidth_to_use,
                         linestyle=linestyle_to_use,
                         zorder=zorder_to_use) # Apply zorder
                
                plt.fill_between(x_axis, 
                                 np.nan_to_num(avg_explored - std_explored, nan=avg_explored),
                                 np.nan_to_num(avg_explored + std_explored, nan=avg_explored),
                                 color=color_to_use, alpha=0.05, zorder=zorder_to_use-1) # very transparent fill
                plotted_anything = True
                print(f"  Successfully called plt.plot for {planner_name} with zorder {zorder_to_use}.")
            else:
                print(f"  SKIPPING PLOT for {planner_name}: No average data to plot.")
        else: 
            print(f"  SKIPPING PLOT for {planner_name}: No valid exploration data in all_results.")

    if not plotted_anything:
        print("CRITICAL: No data was plotted for ANY planner.")
        plt.text(0.5, 0.5, "No data to plot.", ha='center', va='center', transform=plt.gca().transAxes, fontsize=16, color='red')

    plt.xlabel("Total Physical Steps Taken by Robot", fontsize=12)
    plt.ylabel("Percentage of Area Explored (%)", fontsize=12)
    title = (f"Exploration Performance Comparison\n(Map: {ENV_WIDTH}x{ENV_HEIGHT}, Obst: {OBSTACLE_PERCENTAGE*100:.0f}%, "
             f"Sensor: {ROBOT_SENSOR_RANGE}, Start: {ROBOT_START_POS})\n"
             f"MAX_PHYSICAL_STEPS={MAX_PHYSICAL_STEPS}, Num Runs: {num_runs}")
    plt.title(title, fontsize=11)
    plt.legend(loc='best', fontsize=10); plt.grid(True, linestyle=':', alpha=0.6)
    plt.ylim(0, 105)
    plot_xlim_upper = MAX_PHYSICAL_STEPS
    if max_x_lim_plot > 0: plot_xlim_upper = min(MAX_PHYSICAL_STEPS + 20, max_x_lim_plot + int(max_x_lim_plot * 0.05) + 20)
    plt.xlim(left=-1, right=plot_xlim_upper if plot_xlim_upper > 0 else MAX_PHYSICAL_STEPS)
    plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tight_layout(pad=1.5)
    
    filename = "exploration_ige_visibility_test.png" 
    plt.savefig(filename)
    print(f"\nPlot saved: {filename}")
    plt.show()