# main_simulation.py
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

from environment import Environment, UNKNOWN, OBSTACLE, FREE
from robot import Robot
from planners.fbe_planner import FBEPlanner
from planners.aco_planner import ACOPlanner
from planners.ige_planner import IGEPlanner
from planners.ure_planner import UREPlanner
from visualizer import Visualizer

# --- Default Simulation Parameters (used if not provided via command line) ---
DEFAULT_MAP_SIZES_LIST = [(50, 50), (100,100)] 
DEFAULT_OBSTACLE_PERCENTAGES_LIST = [0.15, 0.30]
DEFAULT_MAP_TYPES_LIST = ["random", "deceptive_hallway"]
DEFAULT_PLANNERS_LIST = ["FBE", "ACO"]

ROBOT_SENSOR_RANGE_DEFAULT = 5
MAX_PHYSICAL_STEPS_DEFAULT = 1000
MAX_PLANNING_CYCLES_DEFAULT = 500
CELL_SIZE_DEFAULT = 10
SIM_DELAY_PHYSICAL_STEP_DEFAULT = 0.001
SIM_DELAY_PLANNING_PHASE_DEFAULT = 0.001
VISUALIZE_ANT_ITERATIONS_DEFAULT = False
SCREENSHOT_INTERVAL_DEFAULT = 0
NUM_RUNS_PER_CONFIG_DEFAULT = 1
MASTER_SEED_DEFAULT = 420

ACO_N_ANTS_DEFAULT = 15
ACO_N_ITERATIONS_DEFAULT = 20
ACO_ALPHA_DEFAULT = 1.0
ACO_BETA_DEFAULT = 2.0 
ACO_EVAPORATION_RATE_DEFAULT = 0.25
ACO_Q0_DEFAULT = 0.7
ACO_IG_WEIGHT_HEURISTIC_DEFAULT = 2.5

_skip_ant_visualization_for_this_run = False
BASE_OUTPUT_DIR_DEFAULT_FOR_DIRECT_RUN = "simulation_results_main_direct_run_en"

def visualize_ant_paths_for_aco(robot_pos, known_map, all_ant_paths_iterations, environment_obj, visualizer_obj, visualize_ants_flag):
    global _skip_ant_visualization_for_this_run
    if not visualize_ants_flag or _skip_ant_visualization_for_this_run: return
    if visualizer_obj:
        total_iters = len(all_ant_paths_iterations)
        for i, paths_this_iter in enumerate(all_ant_paths_iterations):
            ret = visualizer_obj.handle_events();
            if not ret: pygame.quit(); exit()
            if ret == "SKIP_ANTS": _skip_ant_visualization_for_this_run = True; print("Skipping ACO ant visualization for this planning cycle."); return
            visualizer_obj.draw_ant_paths_iteration(robot_pos, known_map, paths_this_iter, environment_obj, i, total_iters)

def run_simulation(
    planner_type_arg, 
    env_width_arg, env_height_arg,
    current_obstacle_percentage_arg,
    map_type_sim_arg,
    robot_sensor_range_arg,
    max_physical_steps_arg, max_planning_cycles_arg,
    cell_size_viz_arg, sim_delay_physical_arg, sim_delay_planning_arg,
    visualize_ants_flag_arg,
    aco_params_arg,
    env_seed_arg=None,
    show_visualization_arg=True, 
    run_index_arg=0,
    specific_output_dir_arg=".", 
    screenshot_interval_sim_arg=0):
    
    global _skip_ant_visualization_for_this_run
    _skip_ant_visualization_for_this_run = False
    if env_seed_arg is not None: np.random.seed(env_seed_arg)

    ROBOT_START_POS_RUNTIME = (env_height_arg // 2, 5 if env_width_arg > 10 else env_width_arg // 2)

    env = Environment(env_width_arg, env_height_arg, current_obstacle_percentage_arg,
                      map_type=map_type_sim_arg,
                      robot_start_pos_ref=ROBOT_START_POS_RUNTIME)

    actual_robot_start_pos = ROBOT_START_POS_RUNTIME
    if env.grid_true[ROBOT_START_POS_RUNTIME[0], ROBOT_START_POS_RUNTIME[1]] == OBSTACLE:
        found_free_start = False
        for r_off in range(-3, 4):
            for c_off in range(-3, 4):
                if not found_free_start:
                    nr, nc = ROBOT_START_POS_RUNTIME[0] + r_off, ROBOT_START_POS_RUNTIME[1] + c_off
                    if env.is_within_bounds(nr, nc) and env.grid_true[nr, nc] == FREE:
                        actual_robot_start_pos = (nr, nc); found_free_start = True; break
            if found_free_start: break
        if not found_free_start: 
            err_msg = (f"Error: P:{planner_type_arg}, Map:{map_type_sim_arg} ({env_width_arg}x{env_height_arg}), "
                       f"O:{current_obstacle_percentage_arg:.2f}, S:{env_seed_arg} - "
                       f"Could not find a free start position near {ROBOT_START_POS_RUNTIME}.")
            return {"error": err_msg}, {}
    
    robot = Robot(actual_robot_start_pos, robot_sensor_range_arg)
    visualizer = None
    can_visualize_and_screenshot = False
    if show_visualization_arg:
        try:
            visualizer = Visualizer(env_width_arg, env_height_arg, cell_size_viz_arg)
            can_visualize_and_screenshot = True
        except pygame.error as e:
            # print(f"Warning: Failed to initialize Pygame Visualizer (P:{planner_type_arg}, S:{env_seed_arg}): {e}")
            # print("         This might be due to no available display. Screenshots will not be possible.")
            visualizer = None
    
    ant_viz_callback = None
    if planner_type_arg == "ACO" and can_visualize_and_screenshot and visualize_ants_flag_arg:
        ant_viz_callback = lambda rp, km, api, eo: visualize_ant_paths_for_aco(rp, km, api, eo, visualizer, visualize_ants_flag_arg)

    planner = None
    if planner_type_arg == "FBE": planner = FBEPlanner(env)
    elif planner_type_arg == "ACO":
        planner = ACOPlanner(env, n_ants=aco_params_arg['n_ants'], n_iterations=aco_params_arg['n_iterations'],
                             alpha=aco_params_arg['alpha'], beta=aco_params_arg['beta'],
                             evaporation_rate=aco_params_arg['evaporation_rate'], q0=aco_params_arg['q0'],
                             ig_weight_heuristic=aco_params_arg['ig_weight_heuristic'],
                             robot_sensor_range_for_heuristic=aco_params_arg['sensor_heuristic_range'],
                             visualize_ants_callback=ant_viz_callback)
    elif planner_type_arg == "IGE": planner = IGEPlanner(env, robot_sensor_range_arg)
    elif planner_type_arg == "URE": planner = UREPlanner(env, robot_sensor_range_arg)
    else: raise ValueError(f"Unknown planner type: {planner_type_arg}")

    total_physical_steps_taken = 0
    total_planning_cycles = 0
    total_explorable_area = env.get_total_explorable_area()
    if total_explorable_area == 0: total_explorable_area = 1.0

    sim_active = True
    current_intended_path_segment = []
    current_ultimate_frontier_target = None

    robot.sense(env)
    if planner_type_arg == "URE" and planner:
        planner.update_observation_counts(robot.get_position(), env.grid_known)

    explored_percentages_over_physical_steps = [(env.get_explored_area() / total_explorable_area) * 100]
    physical_step_counts = [0]
    last_screenshot_step = -screenshot_interval_sim_arg

    while total_physical_steps_taken < max_physical_steps_arg and \
          total_planning_cycles < max_planning_cycles_arg and \
          sim_active:

        current_explored_percentage = (env.get_explored_area() / total_explorable_area) * 100
        if current_explored_percentage >= 99.9:
            sim_active = False; break

        if can_visualize_and_screenshot:
            sim_event_ret = visualizer.handle_events()
            if not sim_event_ret: sim_active = False; break
            if sim_event_ret == "SKIP_ANTS" and planner_type_arg == "ACO" and not current_intended_path_segment:
                _skip_ant_visualization_for_this_run = True

            visualizer.clear_screen(); visualizer.draw_environment(env.grid_known)
            visualizer.draw_robot_path(robot.actual_pos_history)
            visualizer.draw_robot(robot.get_position())
            if current_ultimate_frontier_target: visualizer.draw_target(current_ultimate_frontier_target)

            text_y = 10
            vis_title = (f"{planner_type_arg}-M:{map_type_sim_arg}({env_width_arg}x{env_height_arg})"
                         f"-O:{current_obstacle_percentage_arg:.2f}-S:{env_seed_arg}")
            visualizer.draw_text(vis_title, (10, text_y)); text_y += 20
            visualizer.draw_text(f"Steps: {total_physical_steps_taken}/{max_physical_steps_arg}", (10, text_y)); text_y += 20
            visualizer.draw_text(f"Cycles: {total_planning_cycles}/{max_planning_cycles_arg}", (10, text_y)); text_y += 20
            visualizer.draw_text(f"Explored: {current_explored_percentage:.2f}%", (10, text_y)); text_y += 20
            if planner_type_arg == "ACO" and visualize_ants_flag_arg and not current_intended_path_segment:
                visualizer.draw_text("SPACE: Skip Ant Viz", (10, visualizer.screen_height - 50), small=True)
            visualizer.update_display()

            if screenshot_interval_sim_arg > 0 and \
               (total_physical_steps_taken == 0 or total_physical_steps_taken >= last_screenshot_step + screenshot_interval_sim_arg):
                scr_fname = (f"P_{planner_type_arg}_M_{map_type_sim_arg}_Size{env_width_arg}x{env_height_arg}_O{current_obstacle_percentage_arg:.2f}_"
                             f"S{env_seed_arg}_R{run_index_arg}_Step{total_physical_steps_taken}.png")
                screenshots_dir = os.path.join(specific_output_dir_arg, "screenshots")
                visualizer.save_screenshot(screenshots_dir, scr_fname)
                last_screenshot_step = total_physical_steps_taken
            
            effective_delay_physical = sim_delay_physical_arg
            if screenshot_interval_sim_arg > 0 and sim_delay_physical_arg > 0.001 and not (planner_type_arg == "ACO" and visualize_ants_flag_arg):
                 effective_delay_physical = 0.001 

            if not (planner_type_arg == "ACO" and visualize_ants_flag_arg and \
                    not _skip_ant_visualization_for_this_run and not current_intended_path_segment):
                time.sleep(effective_delay_physical)
        elif sim_delay_physical_arg > 0:
            time.sleep(sim_delay_physical_arg)


        if not current_intended_path_segment:
            if can_visualize_and_screenshot and planner_type_arg == "ACO" and \
               visualize_ants_flag_arg and not _skip_ant_visualization_for_this_run:
                time.sleep(sim_delay_planning_arg)

            known_map_for_planner = env.get_known_map_for_planner()
            target_pos, path_segment = None, None
            planner_kwargs = {}
            if planner_type_arg == "FBE" and hasattr(planner, 'plan_next_action') and \
               'robot_path_history' in planner.plan_next_action.__code__.co_varnames:
                 planner_kwargs['robot_path_history'] = [tuple(p) for p in reversed(robot.actual_pos_history)]
            
            target_pos, path_segment = planner.plan_next_action(
                robot.get_position(), known_map_for_planner, **planner_kwargs)
            total_planning_cycles += 1
            if planner_type_arg == "ACO": _skip_ant_visualization_for_this_run = False

            if target_pos is None or not path_segment:
                sim_active = False; break
            current_ultimate_frontier_target = target_pos
            current_intended_path_segment = path_segment[1:] if len(path_segment) > 1 else []

        if current_intended_path_segment:
            next_step_candidate = current_intended_path_segment[0]
            moved_one_cell = robot.attempt_move_one_step(next_step_candidate, env)
            total_physical_steps_taken += 1

            if moved_one_cell:
                current_intended_path_segment.pop(0)
                robot.sense(env)
                if planner_type_arg == "URE" and planner:
                    planner.update_observation_counts(robot.get_position(), env.grid_known)
            else:
                current_intended_path_segment = []
                current_ultimate_frontier_target = None
                robot.sense(env)
                if planner_type_arg == "URE" and planner:
                     planner.update_observation_counts(robot.get_position(), env.grid_known)
            
            new_explored = (env.get_explored_area() / total_explorable_area) * 100
            if not physical_step_counts or physical_step_counts[-1] != total_physical_steps_taken:
                 explored_percentages_over_physical_steps.append(new_explored)
                 physical_step_counts.append(total_physical_steps_taken)
            elif explored_percentages_over_physical_steps:
                explored_percentages_over_physical_steps[-1] = new_explored
        else:
            sim_active = False; break
    
    final_explored_percentage = (env.get_explored_area() / total_explorable_area) * 100

    if not physical_step_counts or physical_step_counts[-1] != total_physical_steps_taken:
        explored_percentages_over_physical_steps.append(final_explored_percentage)
        physical_step_counts.append(total_physical_steps_taken)
    elif explored_percentages_over_physical_steps : 
         explored_percentages_over_physical_steps[-1] = final_explored_percentage

    if can_visualize_and_screenshot and screenshot_interval_sim_arg > 0 and sim_active :
        scr_fname = (f"P_{planner_type_arg}_M_{map_type_sim_arg}_Size{env_width_arg}x{env_height_arg}_O{current_obstacle_percentage_arg:.2f}_"
                     f"S{env_seed_arg}_R{run_index_arg}_Step{total_physical_steps_taken}_Final.png")
        screenshots_dir = os.path.join(specific_output_dir_arg, "screenshots")
        visualizer.save_screenshot(screenshots_dir, scr_fname)

    if can_visualize_and_screenshot and visualizer:
        visualizer.quit()

    results = {
        "planner_type": planner_type_arg,
        "map_size_str": f"{env_width_arg}x{env_height_arg}",
        "map_type": map_type_sim_arg,
        "obstacle_percentage": current_obstacle_percentage_arg,
        "env_seed": env_seed_arg,
        "run_index": run_index_arg,
        "total_physical_steps_taken": total_physical_steps_taken,
        "final_exploration_percentage": final_explored_percentage,
        "total_planning_cycles": total_planning_cycles,
        "max_steps_reached": total_physical_steps_taken >= max_physical_steps_arg,
        "max_planning_cycles_reached": total_planning_cycles >= max_planning_cycles_arg,
    }
    exploration_data = {
        "steps_history": physical_step_counts,
        "explored_history": explored_percentages_over_physical_steps
    }
    return results, exploration_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLAM exploration simulations with different configurations.")
    parser.add_argument("--map_sizes", type=str, default=",".join([f"{s[0]}x{s[1]}" for s in DEFAULT_MAP_SIZES_LIST]), 
                        help="Comma-separated list of map sizes, e.g., 50x50,100x100")
    parser.add_argument("--obs_percentages", type=str, default=",".join(map(str, DEFAULT_OBSTACLE_PERCENTAGES_LIST)),
                        help="Comma-separated list of obstacle percentages, e.g., 0.15,0.30,0.45")
    parser.add_argument("--map_types", type=str, default=",".join(DEFAULT_MAP_TYPES_LIST),
                        help="Comma-separated list of map types, e.g., random,deceptive_hallway")
    parser.add_argument("--planners", type=str, default=",".join(DEFAULT_PLANNERS_LIST),
                        help="Comma-separated list of planners, e.g., FBE,ACO,IGE,URE")
    
    parser.add_argument("--num_runs", type=int, default=NUM_RUNS_PER_CONFIG_DEFAULT,
                        help="Number of runs per configuration")
    parser.add_argument("--master_seed", type=int, default=MASTER_SEED_DEFAULT,
                        help="Master seed for environment generation")
    parser.add_argument("--base_output_dir", type=str, default=BASE_OUTPUT_DIR_DEFAULT_FOR_DIRECT_RUN,
                        help="Base directory for saving results for this specific configuration call")
    parser.add_argument("--screenshot_interval", type=int, default=SCREENSHOT_INTERVAL_DEFAULT,
                        help="Interval for taking screenshots (0 for no screenshots)")
    parser.add_argument("--no_viz", action='store_true',
                        help="Attempt to disable Pygame visualization window (screenshots might still be attempted if interval > 0 but may fail without display)")
    parser.add_argument("--viz_ants", action='store_true', default=VISUALIZE_ANT_ITERATIONS_DEFAULT,
                        help="Enable ACO ant visualization (effective if visualization window is shown)")
    
    parser.add_argument("--robot_sensor_range", type=int, default=ROBOT_SENSOR_RANGE_DEFAULT)
    parser.add_argument("--max_steps", type=int, default=MAX_PHYSICAL_STEPS_DEFAULT)
    parser.add_argument("--max_cycles", type=int, default=MAX_PLANNING_CYCLES_DEFAULT)
    parser.add_argument("--cell_size", type=int, default=CELL_SIZE_DEFAULT)
    parser.add_argument("--delay_physical", type=float, default=SIM_DELAY_PHYSICAL_STEP_DEFAULT)
    parser.add_argument("--delay_planning", type=float, default=SIM_DELAY_PLANNING_PHASE_DEFAULT)
    
    parser.add_argument("--aco_ants", type=int, default=ACO_N_ANTS_DEFAULT)
    parser.add_argument("--aco_iters", type=int, default=ACO_N_ITERATIONS_DEFAULT)
    parser.add_argument("--aco_alpha", type=float, default=ACO_ALPHA_DEFAULT)
    parser.add_argument("--aco_beta", type=float, default=ACO_BETA_DEFAULT)
    parser.add_argument("--aco_evaporation", type=float, default=ACO_EVAPORATION_RATE_DEFAULT)
    parser.add_argument("--aco_q0", type=float, default=ACO_Q0_DEFAULT)
    parser.add_argument("--aco_ig_heuristic_weight", type=float, default=ACO_IG_WEIGHT_HEURISTIC_DEFAULT)

    args = parser.parse_args()

    map_sizes_str_list = args.map_sizes.split(',')
    map_sizes_to_run_final = []
    for s_item in map_sizes_str_list:
        if 'x' not in s_item.lower():
            print(f"Warning: Invalid map size format '{s_item}'. Expected WxH. Skipping.")
            continue
        try:
            w_str, h_str = s_item.lower().split('x')
            map_sizes_to_run_final.append((int(w_str), int(h_str)))
        except ValueError:
            print(f"Warning: Could not parse map size '{s_item}'. Skipping.")
    if not map_sizes_to_run_final: 
        print("Warning: No valid map sizes provided. Using defaults or exiting if none.")
        map_sizes_to_run_final = DEFAULT_MAP_SIZES_LIST

    obstacle_percentages_to_run_final = []
    try:
        obstacle_percentages_to_run_final = [float(p.strip()) for p in args.obs_percentages.split(',') if p.strip()]
    except ValueError:
        print(f"Warning: Invalid obstacle percentage format '{args.obs_percentages}'. Using defaults.")
    if not obstacle_percentages_to_run_final: obstacle_percentages_to_run_final = DEFAULT_OBSTACLE_PERCENTAGES_LIST

    map_types_to_run_final = [mt.strip() for mt in args.map_types.split(',') if mt.strip()]
    if not map_types_to_run_final: map_types_to_run_final = DEFAULT_MAP_TYPES_LIST
    
    planners_to_run_final = [p.strip() for p in args.planners.split(',') if p.strip()]
    if not planners_to_run_final: planners_to_run_final = DEFAULT_PLANNERS_LIST

    num_runs_per_config_final = args.num_runs
    env_master_seed_final = args.master_seed
    
    CONFIG_SPECIFIC_BASE_OUTPUT_DIR = args.base_output_dir 
    os.makedirs(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, exist_ok=True)

    SCREENSHOT_INTERVAL_final = args.screenshot_interval
    attempt_visualization_globally = (SCREENSHOT_INTERVAL_final > 0) or (not args.no_viz)
    global_visualize_ants_final = args.viz_ants if attempt_visualization_globally else False

    all_run_summary_results_for_this_main_call = []
    exploration_data_for_this_main_call_plot = {} 

    aco_params_runtime = {
        'n_ants': args.aco_ants, 'n_iterations': args.aco_iters,
        'alpha': args.aco_alpha, 'beta': args.aco_beta,
        'evaporation_rate': args.aco_evaporation, 'q0': args.aco_q0,
        'ig_weight_heuristic': args.aco_ig_heuristic_weight,
        'sensor_heuristic_range': args.robot_sensor_range 
    }

    overall_start_time = time.time()
    print(f"--- main_simulation.py started (Output to: {CONFIG_SPECIFIC_BASE_OUTPUT_DIR}) ---")

    current_map_w_iter, current_map_h_iter = map_sizes_to_run_final[0]
    current_obs_perc_iter = obstacle_percentages_to_run_final[0]
    current_map_type_iter = map_types_to_run_final[0]
    map_size_str_iter = f"{current_map_w_iter}x{current_map_h_iter}"

    for planner_name_iter in planners_to_run_final:
        plot_key_for_current_planner = (planner_name_iter, map_size_str_iter, current_map_type_iter, current_obs_perc_iter)
        exploration_data_for_this_main_call_plot[plot_key_for_current_planner] = {'steps': [], 'explored': []}
        
        print(f"\n  Executing Planner: {planner_name_iter} "
              f"(Size={map_size_str_iter}, MapType={current_map_type_iter}, Obs%={current_obs_perc_iter:.2f})")
        
        # Ensure screenshots sub-directory exists for this planner's output
        planner_screenshots_dir = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, "screenshots")
        # This was: os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, planner_name_iter, "screenshots")
        # but batch_runner now passes a base_output_dir that already includes the planner name.
        # So, run_simulation just needs to append "screenshots" to the dir it receives.
        # The line below is just for main.py standalone runs if we want planner subdirs.
        # If batch_runner calls this, CONFIG_SPECIFIC_BASE_OUTPUT_DIR is already .../PlannerName
        # So specific_output_dir_arg in run_simulation IS the planner-specific dir.
        
        # For direct run of main_simulation.py, let's ensure planner sub-directory if multiple planners
        if len(planners_to_run_final) > 1 and args.base_output_dir == BASE_OUTPUT_DIR_DEFAULT_FOR_DIRECT_RUN:
            current_planner_output_for_run_sim = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, planner_name_iter)
            os.makedirs(current_planner_output_for_run_sim, exist_ok=True)
        else: # Called by batch_runner, or single planner direct run
            current_planner_output_for_run_sim = CONFIG_SPECIFIC_BASE_OUTPUT_DIR

        os.makedirs(os.path.join(current_planner_output_for_run_sim, "screenshots"), exist_ok=True)


        for i_run in range(num_runs_per_config_final):
            current_env_seed_iter = env_master_seed_final + i_run 
            print(f"    Run {i_run + 1}/{num_runs_per_config_final}, Seed {current_env_seed_iter}")
            
            show_actual_pygame_window_this_run = attempt_visualization_globally and \
                                                 (i_run == 0 or SCREENSHOT_INTERVAL_final > 0)
            
            current_visualize_ants_flag_iter = global_visualize_ants_final if show_actual_pygame_window_this_run and planner_name_iter == "ACO" else False

            summary_res, explor_data = run_simulation(
                planner_type_arg=planner_name_iter,
                env_width_arg=current_map_w_iter, env_height_arg=current_map_h_iter,
                current_obstacle_percentage_arg=current_obs_perc_iter,
                map_type_sim_arg=current_map_type_iter,
                robot_sensor_range_arg=args.robot_sensor_range,
                max_physical_steps_arg=args.max_steps,
                max_planning_cycles_arg=args.max_cycles,
                cell_size_viz_arg=args.cell_size,
                sim_delay_physical_arg=args.delay_physical if show_actual_pygame_window_this_run else 0,
                sim_delay_planning_arg=args.delay_planning if show_actual_pygame_window_this_run and current_visualize_ants_flag_iter else 0,
                visualize_ants_flag_arg=current_visualize_ants_flag_iter,
                aco_params_arg=aco_params_runtime,
                env_seed_arg=current_env_seed_iter,
                show_visualization_arg=show_actual_pygame_window_this_run,
                run_index_arg=i_run,
                specific_output_dir_arg=current_planner_output_for_run_sim, # Pass planner-specific path
                screenshot_interval_sim_arg=SCREENSHOT_INTERVAL_final if show_actual_pygame_window_this_run else 0
            )
            
            if "error" in summary_res:
                print(f"    Run Error: {summary_res['error']}. Skipping.")
                continue

            all_run_summary_results_for_this_main_call.append(summary_res)
            if explor_data["steps_history"]:
                exploration_data_for_this_main_call_plot[plot_key_for_current_planner]['steps'].append(explor_data["steps_history"])
                exploration_data_for_this_main_call_plot[plot_key_for_current_planner]['explored'].append(explor_data["explored_history"])
            
            print(f"    Completed. Steps: {summary_res['total_physical_steps_taken']}. "
                  f"Explored %: {summary_res['final_exploration_percentage']:.2f}")

    overall_end_time = time.time()
    total_duration_min = (overall_end_time - overall_start_time) / 60
    print(f"\n--- All planners for this main.py call completed. Duration: {total_duration_min:.2f} minutes ---")

    if not all_run_summary_results_for_this_main_call:
        print("No valid simulation results collected in this main.py call.")
    else:
        try:
            import pandas as pd
            df_results = pd.DataFrame(all_run_summary_results_for_this_main_call)
            
            summary_aggs = {
                'total_physical_steps_taken': 'mean',
                'final_exploration_percentage': 'mean',
                'total_planning_cycles': 'mean',
                'max_steps_reached': 'mean',
                'max_planning_cycles_reached': 'mean',
                'map_size_str': 'size' 
            }
            avg_summary_df = df_results.groupby(['planner_type', 'map_size_str', 'map_type', 'obstacle_percentage']).agg(summary_aggs).reset_index()
            avg_summary_df.rename(columns={'map_size_str_y': 'completed_runs'}, inplace=True) # Adjust if pandas version changes col name

            print("\nAverage Performance Metrics for this main.py call:")
            print(avg_summary_df.to_string(index=False))
            
            csv_filename = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, f"summary_avg_runs_S{env_master_seed_final}.csv")
            avg_summary_df.to_csv(csv_filename, index=False, float_format='%.2f')
            print(f"\nSummary CSV for this call saved to: {csv_filename}")

        except ImportError:
            print("Warning: pandas library not installed. Detailed table summary skipped.")
        except Exception as e:
            print(f"Error during pandas summary processing: {e}")

        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'sans-serif'] # Add more fallbacks
            plt.rcParams['axes.unicode_minus'] = False
            
            def average_results(list_of_explored_runs, list_of_steps_runs, debug_name="UNSPECIFIED"):
                valid_explored_runs = [r for r in list_of_explored_runs if r and len(r) > 0]
                valid_steps_runs = [r for r in list_of_steps_runs if r and len(r) > 0]
                if not valid_explored_runs or not valid_steps_runs or len(valid_explored_runs) != len(valid_steps_runs):
                    return np.array([]), np.array([]), np.array([])
                
                max_step_all = 0
                for steps_run_iter_avg in valid_steps_runs:
                    if steps_run_iter_avg: max_step_all = max(max_step_all, steps_run_iter_avg[-1])
                
                if max_step_all == 0:
                    is_only_initial_data_avg = all(len(r) == 1 and r[0] == 0 for r in valid_steps_runs)
                    if is_only_initial_data_avg and all(len(r) == 1 for r in valid_explored_runs):
                        initial_expl_values = [r[0] for r in valid_explored_runs]
                        if not initial_expl_values: return np.array([0]), np.array([np.nan]), np.array([np.nan]) # Handle no data
                        return np.array([0]), np.array([np.mean(initial_expl_values)]), np.array([np.std(initial_expl_values)])

                common_steps_axis_avg = np.arange(max_step_all + 1)
                if len(common_steps_axis_avg) == 0 and max_step_all > 0:
                     return np.array([]), np.array([]), np.array([])
                elif len(common_steps_axis_avg) == 0 and max_step_all == 0: 
                     if valid_explored_runs:
                        initial_expl_values = [r[0] for r in valid_explored_runs if r]
                        if initial_expl_values:
                            return np.array([0]), np.array([np.mean(initial_expl_values)]), np.array([np.std(initial_expl_values)])
                     return np.array([]), np.array([]), np.array([])

                aligned_explored_data_avg = []
                for i_avg in range(len(valid_explored_runs)):
                    steps_run_iter_avg, explored_run_iter_avg = valid_steps_runs[i_avg], valid_explored_runs[i_avg]
                    if len(steps_run_iter_avg) != len(explored_run_iter_avg) or not steps_run_iter_avg: continue 
                    
                    left_val_avg = explored_run_iter_avg[0] if steps_run_iter_avg and steps_run_iter_avg[0] == 0 else np.nan 
                    unique_steps_avg, unique_indices_avg = np.unique(steps_run_iter_avg, return_index=True)
                    unique_explored_avg = np.array(explored_run_iter_avg)[unique_indices_avg]

                    if not common_steps_axis_avg.size: 
                        if unique_steps_avg.size == 1 and unique_steps_avg[0] == 0 : 
                             aligned_explored_data_avg.append(unique_explored_avg)
                        continue

                    interpolated_values_avg = np.interp(common_steps_axis_avg, unique_steps_avg, unique_explored_avg, 
                                                    left=left_val_avg, right=unique_explored_avg[-1] if unique_explored_avg.size > 0 else np.nan)
                    aligned_explored_data_avg.append(interpolated_values_avg)
                
                if not aligned_explored_data_avg: 
                    if common_steps_axis_avg.size == 1 and common_steps_axis_avg[0] == 0:
                        all_initial_expl = []
                        for r_init in valid_explored_runs:
                            if r_init: all_initial_expl.append(r_init[0])
                        if all_initial_expl:
                            return np.array([0]), np.array([np.mean(all_initial_expl)]), np.array([np.std(all_initial_expl)])
                    return np.array([]), np.array([]), np.array([])

                avg_explored_val = np.nanmean(np.array(aligned_explored_data_avg), axis=0)
                std_explored_val = np.nanstd(np.array(aligned_explored_data_avg), axis=0)
                return common_steps_axis_avg, avg_explored_val, std_explored_val

            plot_colors = {'FBE': 'blue', 'ACO': 'red', 'IGE': 'green', 'URE': 'purple'}
            
            plt.figure(figsize=(12, 8))
            plotted_anything_on_this_fig_main = False
            max_x_lim_plot_fig_main = 0
            
            fig_title_main = (f"Map: {current_map_type_iter}, Size: {map_size_str_iter}, Obstacle Rate: {current_obs_perc_iter:.2f}\n"
                         f"Exploration Performance (Sensor: {args.robot_sensor_range}, Max Steps: {args.max_steps})\n"
                         f"Averaged over {num_runs_per_config_final} runs (Planners: {', '.join(planners_to_run_final)})")
            plt.title(fig_title_main, fontsize=11)

            for planner_name_plot_main in planners_to_run_final:
                plot_key_lookup_main = (planner_name_plot_main, map_size_str_iter, current_map_type_iter, current_obs_perc_iter)
                
                if plot_key_lookup_main not in exploration_data_for_this_main_call_plot or \
                   not exploration_data_for_this_main_call_plot[plot_key_lookup_main]['steps']:
                    continue
                
                results_data_main = exploration_data_for_this_main_call_plot[plot_key_lookup_main]
                if results_data_main['explored'] and any(run for run in results_data_main['explored'] if run):
                    x_axis_main, avg_explored_main, std_explored_main = average_results(
                        results_data_main['explored'], results_data_main['steps'], 
                        f"PlotData_{planner_name_plot_main}"
                    )
                    
                    if len(x_axis_main) > 0 and len(avg_explored_main) > 0 and not np.all(np.isnan(avg_explored_main)):
                        if len(x_axis_main) > 0 and x_axis_main[-1] > max_x_lim_plot_fig_main:
                            max_x_lim_plot_fig_main = x_axis_main[-1]
                        
                        color_to_use_main = plot_colors.get(planner_name_plot_main, 'black')
                        plt.plot(x_axis_main, avg_explored_main, label=f"{planner_name_plot_main}", color=color_to_use_main, linewidth=1.8)
                        plt.fill_between(x_axis_main, 
                                         np.nan_to_num(avg_explored_main - std_explored_main, nan=avg_explored_main),
                                         np.nan_to_num(avg_explored_main + std_explored_main, nan=avg_explored_main),
                                         color=color_to_use_main, alpha=0.15)
                        plotted_anything_on_this_fig_main = True
            
            if plotted_anything_on_this_fig_main:
                plt.xlabel("Total Physical Steps Taken by Robot", fontsize=12)
                plt.ylabel("Average Explored Area (%)", fontsize=12)
                plt.legend(loc='lower right', fontsize=10); plt.grid(True, linestyle=':', alpha=0.6)
                plt.ylim(0, 105)
                
                plot_xlim_upper_fig_main = args.max_steps
                if max_x_lim_plot_fig_main > 0:
                    plot_xlim_upper_fig_main = min(args.max_steps + 50, max_x_lim_plot_fig_main + int(max_x_lim_plot_fig_main * 0.05) + 50)
                
                plt.xlim(left=-max(1, int(0.01 * plot_xlim_upper_fig_main)), right=plot_xlim_upper_fig_main)
                plt.xticks(fontsize=10); plt.yticks(fontsize=10)
                plt.tight_layout(pad=1.8)
                
                plots_subdir_main = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, "_plots_summary_for_this_config_call")
                os.makedirs(plots_subdir_main, exist_ok=True)
                
                plot_filename_base_main = (f"EXPL_ComparePlanners_Size{map_size_str_iter}_MapT{current_map_type_iter}_"
                                           f"Obs{current_obs_perc_iter:.2f}_S{env_master_seed_final}.png")
                plot_filename_full_main = os.path.join(plots_subdir_main, plot_filename_base_main)

                plt.savefig(plot_filename_full_main)
                print(f"Comparison plot for this configuration saved: {plot_filename_full_main}")
                plt.close()
            else:
                print(f"No data to plot for configuration: Size:{map_size_str_iter}, Map:{current_map_type_iter}, Obs:{current_obs_perc_iter:.2f}")
        except Exception as e_plot:
            print(f"Error during plotting: {e_plot}")