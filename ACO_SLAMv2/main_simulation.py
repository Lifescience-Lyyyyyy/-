# main_simulation.py
import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

from environment import Environment, UNKNOWN, OBSTACLE, FREE
from robot import Robot # Assumes robot.py has LoS integrated if desired
from planners.fbe_planner import FBEPlanner
from planners.aco_planner import ACOPlanner # Assumes this and others have LoS in IG calc
from planners.ige_planner import IGEPlanner
from planners.ure_planner import UREPlanner
from visualizer import Visualizer
from planners.pathfinding import a_star_search # For backtrack planning

# --- Default Simulation Parameters ---
DEFAULT_MAP_SIZES_LIST = [(50, 50), (100,100)] 
DEFAULT_OBSTACLE_PERCENTAGES_LIST = [0.15, 0.30]
DEFAULT_MAP_TYPES_LIST = ["random", "deceptive_hallway"]
DEFAULT_PLANNERS_LIST = ["FBE", "ACO", "IGE", "URE"]

ROBOT_SENSOR_RANGE_DEFAULT = 5
MAX_PHYSICAL_STEPS_DEFAULT_FALLBACK = 2000 # Fallback if dynamic calculation is problematic
MAX_PLANNING_CYCLES_DEFAULT = 500
CELL_SIZE_DEFAULT = 10
SIM_DELAY_PHYSICAL_STEP_DEFAULT = 0.001
SIM_DELAY_PLANNING_PHASE_DEFAULT = 0.001
VISUALIZE_ANT_ITERATIONS_DEFAULT = False
SCREENSHOT_INTERVAL_DEFAULT = 0
NUM_RUNS_PER_CONFIG_DEFAULT = 1 
MASTER_SEED_DEFAULT = 420
MAX_BACKTRACK_ATTEMPTS_DEFAULT = 3
BACKTRACK_STEPS_DEFAULT = 5

ACO_N_ANTS_DEFAULT = 15
ACO_N_ITERATIONS_DEFAULT = 20
ACO_ALPHA_DEFAULT = 1.0
ACO_BETA_DEFAULT = 2.0 
ACO_EVAPORATION_RATE_DEFAULT = 0.25
ACO_Q0_DEFAULT = 0.7
ACO_IG_WEIGHT_HEURISTIC_DEFAULT = 2.5

_skip_ant_visualization_for_this_run = False
# Default base output directory when main_simulation.py is run directly
BASE_OUTPUT_DIR_DEFAULT_FOR_DIRECT_RUN = f"sim_direct_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def visualize_ant_paths_for_aco(robot_pos, known_map, all_ant_paths_iterations, environment_obj, visualizer_obj, visualize_ants_flag):
    global _skip_ant_visualization_for_this_run
    if not visualize_ants_flag or _skip_ant_visualization_for_this_run or not visualizer_obj: return
    # ... (rest of the function is the same as your previous version)
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
    max_physical_steps_arg, 
    max_planning_cycles_arg,
    cell_size_viz_arg, sim_delay_physical_arg, sim_delay_planning_arg,
    visualize_ants_flag_arg,
    aco_params_arg,
    env_seed_arg=None,
    show_visualization_arg=True, 
    run_index_arg=0, 
    specific_output_dir_arg=".", 
    screenshot_interval_sim_arg=0,
    max_backtrack_attempts_arg=3, 
    backtrack_steps_arg=5):      

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
        # Wider search for start position if default is blocked
        for r_off in range(-(env_height_arg//4), env_height_arg//4 + 1): 
            for c_off in range(-(env_width_arg//4), env_width_arg//4 + 1):
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
    can_visualize_and_screenshot = False # Flag to check if visualizer is successfully initialized
    if show_visualization_arg:
        try:
            visualizer = Visualizer(env_width_arg, env_height_arg, cell_size_viz_arg)
            can_visualize_and_screenshot = True
        except pygame.error as e:
            print(f"Warning: Pygame Visualizer initialization failed (P:{planner_type_arg}, S:{env_seed_arg}): {e}. "
                  "Continuing without visualization/screenshots for this run.")
            visualizer = None # Ensure it's None
    
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
    last_screenshot_step = -screenshot_interval_sim_arg if screenshot_interval_sim_arg > 0 else 0

    current_backtrack_attempts = 0
    backtracking_path = []

    while total_physical_steps_taken < max_physical_steps_arg and \
          total_planning_cycles < max_planning_cycles_arg and \
          sim_active:

        current_explored_percentage = (env.get_explored_area() / total_explorable_area) * 100
        if current_explored_percentage >= 99.9:
            sim_active = False; break

        if can_visualize_and_screenshot:
            sim_event_ret = visualizer.handle_events()
            if not sim_event_ret: sim_active = False; break
            if sim_event_ret == "SKIP_ANTS" and planner_type_arg == "ACO" and not current_intended_path_segment and not backtracking_path:
                _skip_ant_visualization_for_this_run = True

            visualizer.clear_screen(); visualizer.draw_environment(env.grid_known)
            visualizer.draw_robot_path(robot.actual_pos_history)
            visualizer.draw_robot(robot.get_position())
            if current_ultimate_frontier_target: visualizer.draw_target(current_ultimate_frontier_target)
            if backtracking_path and visualizer: 
                 for i_bp in range(len(backtracking_path) -1) : # Draw lines between points in backtrack_path
                    bp_start_r, bp_start_c = backtracking_path[i_bp]
                    # Next point in path is the end of the line segment
                    # Need to be careful if backtracking_path can have just one point
                    if i_bp + 1 < len(backtracking_path):
                        bp_end_r, bp_end_c = backtracking_path[i_bp+1]
                        bp_start_px = (bp_start_c * cell_size_viz_arg + cell_size_viz_arg // 2, bp_start_r * cell_size_viz_arg + cell_size_viz_arg // 2)
                        bp_end_px = (bp_end_c * cell_size_viz_arg + cell_size_viz_arg // 2, bp_end_r * cell_size_viz_arg + cell_size_viz_arg // 2)
                        pygame.draw.line(visualizer.screen, (255,0,255), bp_start_px, bp_end_px, 1) 


            text_y = 10; vis_title = (f"{planner_type_arg}-M:{map_type_sim_arg}({env_width_arg}x{env_height_arg})"
                                      f"-O:{current_obstacle_percentage_arg:.2f}-S:{env_seed_arg}-R{run_index_arg}")
            visualizer.draw_text(vis_title, (10, text_y)); text_y += 20
            visualizer.draw_text(f"Steps: {total_physical_steps_taken}/{max_physical_steps_arg}", (10, text_y)); text_y += 20
            visualizer.draw_text(f"Cycles: {total_planning_cycles}/{max_planning_cycles_arg}", (10, text_y)); text_y += 20
            visualizer.draw_text(f"Explored: {current_explored_percentage:.2f}%", (10, text_y)); text_y += 20
            if current_backtrack_attempts > 0:
                visualizer.draw_text(f"Backtracks: {current_backtrack_attempts}/{max_backtrack_attempts_arg}", (10,text_y)); text_y +=20
            if planner_type_arg == "ACO" and visualize_ants_flag_arg and not current_intended_path_segment and not backtracking_path:
                visualizer.draw_text("SPACE: Skip Ant Viz", (10, visualizer.screen_height - 50), small=True)
            visualizer.update_display()

            if screenshot_interval_sim_arg > 0 and \
               (total_physical_steps_taken == 0 or total_physical_steps_taken >= last_screenshot_step + screenshot_interval_sim_arg):
                scr_fname = (f"P_{planner_type_arg}_M_{map_type_sim_arg}_Size{env_width_arg}x{env_height_arg}_O{current_obstacle_percentage_arg:.2f}_"
                             f"S{env_seed_arg}_R{run_index_arg}_Step{total_physical_steps_taken}.png")
                screenshots_dir = os.path.join(specific_output_dir_arg, "screenshots")
                # visualizer.save_screenshot itself creates the directory if it doesn't exist
                visualizer.save_screenshot(screenshots_dir, scr_fname)
                last_screenshot_step = total_physical_steps_taken
            
            effective_delay_physical = sim_delay_physical_arg
            if screenshot_interval_sim_arg > 0 and sim_delay_physical_arg > 0.001 and not (planner_type_arg == "ACO" and visualize_ants_flag_arg):
                 effective_delay_physical = 0.0001 
            if not (planner_type_arg == "ACO" and visualize_ants_flag_arg and \
                    not _skip_ant_visualization_for_this_run and not current_intended_path_segment and not backtracking_path):
                time.sleep(effective_delay_physical)
        elif sim_delay_physical_arg > 0: 
            time.sleep(sim_delay_physical_arg)

        if backtracking_path:
            next_backtrack_step_tuple = backtracking_path.pop(0) 
            moved = robot.attempt_move_one_step(next_backtrack_step_tuple, env)
            total_physical_steps_taken += 1
            if not moved:
                backtracking_path = [] 
            robot.sense(env)
            if planner_type_arg == "URE" and planner: planner.update_observation_counts(robot.get_position(), env.grid_known)
            
            new_explored = (env.get_explored_area() / total_explorable_area) * 100
            if not physical_step_counts or physical_step_counts[-1] != total_physical_steps_taken:
                 explored_percentages_over_physical_steps.append(new_explored); physical_step_counts.append(total_physical_steps_taken)
            elif explored_percentages_over_physical_steps: explored_percentages_over_physical_steps[-1] = new_explored
            
            if not backtracking_path: current_intended_path_segment = [] 
            continue

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
            
            target_pos, path_segment = planner.plan_next_action(robot.get_position(), known_map_for_planner, **planner_kwargs)
            total_planning_cycles += 1
            if planner_type_arg == "ACO": _skip_ant_visualization_for_this_run = False

            if target_pos is None or not path_segment:
                if current_backtrack_attempts < max_backtrack_attempts_arg:
                    current_backtrack_attempts += 1
                    hist = robot.actual_pos_history
                    backtrack_target_pos_candidate = None
                    if len(hist) > 1:
                        unique_hist_reversed = []
                        if hist:
                            last_added_hist = None
                            for p_hist in reversed(hist): # Reversed to get [current, prev1, prev2...]
                                if list(p_hist) != last_added_hist: # Compare as list or tuple
                                    unique_hist_reversed.append(list(p_hist))
                                    last_added_hist = list(p_hist)
                        
                        if len(unique_hist_reversed) > backtrack_steps_arg :
                            backtrack_target_pos_candidate = tuple(unique_hist_reversed[backtrack_steps_arg])
                        elif len(unique_hist_reversed) > 1 :
                             backtrack_target_pos_candidate = tuple(unique_hist_reversed[1]) 

                    if backtrack_target_pos_candidate and backtrack_target_pos_candidate != robot.get_position():
                        path_to_backtrack_target = a_star_search(env.get_known_map_for_planner(), robot.get_position(), backtrack_target_pos_candidate)
                        if path_to_backtrack_target and len(path_to_backtrack_target) > 1:
                            backtracking_path = path_to_backtrack_target # Full path including start
                            # We pop current pos from backtracking_path if robot is already at path_to_backtrack_target[0]
                            if backtracking_path[0] == robot.get_position():
                                backtracking_path.pop(0)
                            
                            current_intended_path_segment = []
                            current_ultimate_frontier_target = None
                            if backtracking_path: # If there's actually a path to backtrack on
                                # print(f"    Starting backtrack to {backtrack_target_pos_candidate} via path of length {len(backtracking_path)}")
                                continue 
                            else: # Path to backtrack was just current pos, or empty
                                # print(f"    Backtrack target {backtrack_target_pos_candidate} is current/unreachable or path is trivial. Stuck.")
                                sim_active = False; break 
                        else: 
                            # print(f"    Could not find path to backtrack target {backtrack_target_pos_candidate}. Stuck.")
                            sim_active = False; break 
                    else: 
                        # print("    No suitable backtrack target found or already at target. Stuck.")
                        sim_active = False; break 
                else: 
                    # print(f"  Max backtrack attempts ({max_backtrack_attempts_arg}) reached. Simulation stuck.")
                    sim_active = False; break
            else: 
                current_backtrack_attempts = 0 
                current_ultimate_frontier_target = target_pos
                current_intended_path_segment = path_segment[1:] if len(path_segment) > 1 else []

        if current_intended_path_segment:
            next_step_candidate = current_intended_path_segment[0]
            moved_one_cell = robot.attempt_move_one_step(next_step_candidate, env)
            total_physical_steps_taken += 1
            if moved_one_cell:
                current_intended_path_segment.pop(0)
                robot.sense(env)
                if planner_type_arg == "URE" and planner: planner.update_observation_counts(robot.get_position(), env.grid_known)
            else: 
                current_intended_path_segment = []
                current_ultimate_frontier_target = None
                robot.sense(env)
                if planner_type_arg == "URE" and planner: planner.update_observation_counts(robot.get_position(), env.grid_known)
            
            new_explored = (env.get_explored_area() / total_explorable_area) * 100
            if not physical_step_counts or physical_step_counts[-1] != total_physical_steps_taken:
                 explored_percentages_over_physical_steps.append(new_explored); physical_step_counts.append(total_physical_steps_taken)
            elif explored_percentages_over_physical_steps: explored_percentages_over_physical_steps[-1] = new_explored
        elif not backtracking_path: 
            # This case: no intended path segment, and not currently backtracking.
            # This implies planning might have returned an empty path or just the current pos.
            # Or, a backtrack path just finished.
            # The loop will go to the top, and 'if not current_intended_path_segment' will trigger re-planning.
            # If planning consistently fails, the backtrack logic above will eventually stop sim_active.
            # So, no 'break' here is needed, allow re-planning.
            pass 
    
    final_explored_percentage = (env.get_explored_area() / total_explorable_area) * 100
    if not physical_step_counts or physical_step_counts[-1] != total_physical_steps_taken:
        explored_percentages_over_physical_steps.append(final_explored_percentage); physical_step_counts.append(total_physical_steps_taken)
    elif explored_percentages_over_physical_steps : explored_percentages_over_physical_steps[-1] = final_explored_percentage

    if can_visualize_and_screenshot and visualizer and screenshot_interval_sim_arg > 0:
        # Check sim_active removed from here, always take final screenshot if viz was on and interval set
        scr_fname = (f"P_{planner_type_arg}_M_{map_type_sim_arg}_Size{env_width_arg}x{env_height_arg}_O{current_obstacle_percentage_arg:.2f}_"
                     f"S{env_seed_arg}_R{run_index_arg}_Step{total_physical_steps_taken}_Final.png")
        screenshots_dir = os.path.join(specific_output_dir_arg, "screenshots")
        visualizer.save_screenshot(screenshots_dir, scr_fname)
        # print(f"    Final screenshot saved: {scr_fname}")

    if can_visualize_and_screenshot and visualizer: visualizer.quit()

    results = {
        "planner_type": planner_type_arg, "map_size_str": f"{env_width_arg}x{env_height_arg}",
        "map_type": map_type_sim_arg, "obstacle_percentage": current_obstacle_percentage_arg,
        "env_seed": env_seed_arg, "run_index": run_index_arg,
        "total_physical_steps_taken": total_physical_steps_taken,
        "final_exploration_percentage": final_explored_percentage,
        "total_planning_cycles": total_planning_cycles,
        "total_backtrack_attempts_made": current_backtrack_attempts,
        "max_steps_reached": total_physical_steps_taken >= max_physical_steps_arg,
        "max_planning_cycles_reached": total_planning_cycles >= max_planning_cycles_arg,
    }
    exploration_data = {"steps_history": physical_step_counts, "explored_history": explored_percentages_over_physical_steps}
    return results, exploration_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SLAM exploration simulations with different configurations.")
    parser.add_argument("--map_sizes", type=str, default=",".join([f"{s[0]}x{s[1]}" for s in DEFAULT_MAP_SIZES_LIST]))
    parser.add_argument("--obs_percentages", type=str, default=",".join(map(str, DEFAULT_OBSTACLE_PERCENTAGES_LIST)))
    parser.add_argument("--map_types", type=str, default=",".join(DEFAULT_MAP_TYPES_LIST))
    parser.add_argument("--planners", type=str, default=",".join(DEFAULT_PLANNERS_LIST))
    parser.add_argument("--num_runs", type=int, default=NUM_RUNS_PER_CONFIG_DEFAULT)
    parser.add_argument("--master_seed", type=int, default=MASTER_SEED_DEFAULT)
    parser.add_argument("--base_output_dir", type=str, default=f"sim_direct_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--screenshot_interval", type=int, default=SCREENSHOT_INTERVAL_DEFAULT)
    parser.add_argument("--no_viz", action='store_true', help="Attempt to disable Pygame visualization window")
    parser.add_argument("--viz_ants", action='store_true', default=VISUALIZE_ANT_ITERATIONS_DEFAULT)
    parser.add_argument("--robot_sensor_range", type=int, default=ROBOT_SENSOR_RANGE_DEFAULT)
    parser.add_argument("--max_steps_override", type=int, default=None, help="Override dynamic max steps. If None, dynamic calculation is used.")
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
    parser.add_argument("--max_backtrack_attempts", type=int, default=MAX_BACKTRACK_ATTEMPTS_DEFAULT)
    parser.add_argument("--backtrack_steps", type=int, default=BACKTRACK_STEPS_DEFAULT)
    args = parser.parse_args()

    map_sizes_str_list = args.map_sizes.split(',')
    map_sizes_to_run_final = []
    for s_item in map_sizes_str_list:
        if 'x' not in s_item.lower(): print(f"Warning: Invalid map size format '{s_item}'. Expected WxH. Skipping."); continue
        try: w_str, h_str = s_item.lower().split('x'); map_sizes_to_run_final.append((int(w_str), int(h_str)))
        except ValueError: print(f"Warning: Could not parse map size '{s_item}'. Skipping.")
    if not map_sizes_to_run_final: map_sizes_to_run_final = DEFAULT_MAP_SIZES_LIST

    obstacle_percentages_to_run_final = []
    try: obstacle_percentages_to_run_final = [float(p.strip()) for p in args.obs_percentages.split(',') if p.strip()]
    except ValueError: print(f"Warning: Invalid obstacle percentage format '{args.obs_percentages}'. Using defaults.")
    if not obstacle_percentages_to_run_final: obstacle_percentages_to_run_final = DEFAULT_OBSTACLE_PERCENTAGES_LIST

    map_types_to_run_final = [mt.strip() for mt in args.map_types.split(',') if mt.strip()]
    if not map_types_to_run_final: map_types_to_run_final = DEFAULT_MAP_TYPES_LIST
    
    planners_to_run_final = [p.strip() for p in args.planners.split(',') if p.strip()]
    if not planners_to_run_final: planners_to_run_final = DEFAULT_PLANNERS_LIST

    CONFIG_SPECIFIC_BASE_OUTPUT_DIR = args.base_output_dir 
    os.makedirs(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, exist_ok=True)
    
    attempt_visualization_globally = (args.screenshot_interval > 0) or (not args.no_viz)
    global_visualize_ants_final = args.viz_ants if attempt_visualization_globally else False
    
    aco_params_runtime = {
        'n_ants': args.aco_ants, 'n_iterations': args.aco_iters, 'alpha': args.aco_alpha, 
        'beta': args.aco_beta, 'evaporation_rate': args.aco_evaporation, 'q0': args.aco_q0,
        'ig_weight_heuristic': args.aco_ig_heuristic_weight, 'sensor_heuristic_range': args.robot_sensor_range 
    }

    all_run_summary_results_for_this_main_call = []
    exploration_data_for_this_main_call_plot = {} 

    overall_start_time = time.time()
    print(f"--- main_simulation.py started (Output to: {CONFIG_SPECIFIC_BASE_OUTPUT_DIR}) ---")

    # When main.py is called by batch_runner, these lists will contain a single element.
    # When main.py is run directly, they can contain multiple elements as per its defaults or command line.
    # The current structure of main.py's __main__ block is to pick the *first* map config
    # and then iterate through planners. This is fine if batch_runner calls it for each map config.
    current_map_w_iter, current_map_h_iter = map_sizes_to_run_final[0]
    current_obs_perc_iter = obstacle_percentages_to_run_final[0]
    current_map_type_iter = map_types_to_run_final[0]
    map_size_str_iter = f"{current_map_w_iter}x{current_map_h_iter}"
    current_sensor_range = args.robot_sensor_range

    current_max_steps_calculated = args.max_cycles # Default to max_cycles
    if args.max_steps_override is not None:
        current_max_steps_calculated = args.max_steps_override
        print(f"    Using overridden max steps: {current_max_steps_calculated}")
    else:
        effective_width = current_map_w_iter - 2 * current_sensor_range
        effective_height = current_map_h_iter - 2 * current_sensor_range
        if effective_width <= 0 or effective_height <= 0 or current_sensor_range <= 0:
            current_max_steps_calculated = max(MAX_PHYSICAL_STEPS_DEFAULT_FALLBACK, 
                                               int((current_map_w_iter * current_map_h_iter) * (1.5 + current_obs_perc_iter)))
            print(f"    Map too small for sensor range, using fallback max steps: {current_max_steps_calculated}")
        else:
            term1_denominator = 2 * current_sensor_range
            if term1_denominator == 0: term1_denominator = 1 
            term1 = (effective_width * effective_height) / term1_denominator
            term2 = effective_width 
            calculated_steps = int((term1 + term2) * (2 * current_obs_perc_iter + 1.5)) 
            current_max_steps_calculated = max(300, min(calculated_steps, 20000)) 
        print(f"    Dynamic max steps for {map_size_str_iter}, Obs {current_obs_perc_iter:.2f}, Sensor {current_sensor_range}: {current_max_steps_calculated}")


    for planner_name_iter in planners_to_run_final:
        plot_key_for_current_planner = (planner_name_iter, map_size_str_iter, current_map_type_iter, current_obs_perc_iter)
        exploration_data_for_this_main_call_plot[plot_key_for_current_planner] = {'steps': [], 'explored': []}
        
        print(f"\n  Executing Planner: {planner_name_iter} "
              f"(Size={map_size_str_iter}, MapType={current_map_type_iter}, Obs%={current_obs_perc_iter:.2f})")
        
        # Output dir for this specific planner's runs (if main.py called directly with multiple planners)
        # If called by batch_runner, CONFIG_SPECIFIC_BASE_OUTPUT_DIR is already planner-specific.
        if args.base_output_dir == BASE_OUTPUT_DIR_DEFAULT_FOR_DIRECT_RUN and len(planners_to_run_final) > 1:
            current_planner_output_for_run_sim = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, planner_name_iter)
            os.makedirs(current_planner_output_for_run_sim, exist_ok=True)
        else:
            current_planner_output_for_run_sim = CONFIG_SPECIFIC_BASE_OUTPUT_DIR
        
        # Ensure screenshots subdir exists within the planner's specific output directory
        os.makedirs(os.path.join(current_planner_output_for_run_sim, "screenshots"), exist_ok=True)

        for i_run in range(args.num_runs):
            current_env_seed_iter = args.master_seed + i_run 
            print(f"    Run {i_run + 1}/{args.num_runs}, Seed {current_env_seed_iter}")
            
            show_actual_pygame_window_this_run = attempt_visualization_globally and \
                                                 (i_run == 0 or args.screenshot_interval > 0)
            current_visualize_ants_flag_iter = global_visualize_ants_final if show_actual_pygame_window_this_run and planner_name_iter == "ACO" else False

            summary_res, explor_data = run_simulation(
                planner_type_arg=planner_name_iter,
                env_width_arg=current_map_w_iter, env_height_arg=current_map_h_iter,
                current_obstacle_percentage_arg=current_obs_perc_iter,
                map_type_sim_arg=current_map_type_iter,
                robot_sensor_range_arg=current_sensor_range,
                max_physical_steps_arg=current_max_steps_calculated, 
                max_planning_cycles_arg=args.max_cycles,
                cell_size_viz_arg=args.cell_size,
                sim_delay_physical_arg=args.delay_physical if show_actual_pygame_window_this_run else 0,
                sim_delay_planning_arg=args.delay_planning if show_actual_pygame_window_this_run and current_visualize_ants_flag_iter else 0,
                visualize_ants_flag_arg=current_visualize_ants_flag_iter,
                aco_params_arg=aco_params_runtime,
                env_seed_arg=current_env_seed_iter,
                show_visualization_arg=show_actual_pygame_window_this_run,
                run_index_arg=i_run,
                specific_output_dir_arg=current_planner_output_for_run_sim, 
                screenshot_interval_sim_arg=args.screenshot_interval if show_actual_pygame_window_this_run else 0,
                max_backtrack_attempts_arg=args.max_backtrack_attempts, 
                backtrack_steps_arg=args.backtrack_steps
            )
            
            if "error" in summary_res: print(f"    Run Error: {summary_res['error']}. Skipping."); continue
            all_run_summary_results_for_this_main_call.append(summary_res)
            if explor_data["steps_history"]:
                exploration_data_for_this_main_call_plot[plot_key_for_current_planner]['steps'].append(explor_data["steps_history"])
                exploration_data_for_this_main_call_plot[plot_key_for_current_planner]['explored'].append(explor_data["explored_history"])
            print(f"    Completed. Steps: {summary_res['total_physical_steps_taken']}. Explored %: {summary_res['final_exploration_percentage']:.2f}. Backtracks: {summary_res.get('total_backtrack_attempts_made', 0)}")

    overall_end_time = time.time()
    total_duration_min = (overall_end_time - overall_start_time) / 60
    print(f"\n--- All simulations for this main.py call completed. Duration: {total_duration_min:.2f} minutes ---")

    if not all_run_summary_results_for_this_main_call:
        print("No valid simulation results collected in this main.py call.")
    else:
        try:
            import pandas as pd
            df_results = pd.DataFrame(all_run_summary_results_for_this_main_call)
            summary_aggs = {
                'total_physical_steps_taken': 'mean', 'final_exploration_percentage': 'mean',
                'total_planning_cycles': 'mean', 'total_backtrack_attempts_made': 'mean',
                'max_steps_reached': 'mean', 'max_planning_cycles_reached': 'mean',
                'run_index': 'count' # Count runs per group
            }
            grouping_keys = ['planner_type', 'map_size_str', 'map_type', 'obstacle_percentage']
            avg_summary_df = df_results.groupby(grouping_keys).agg(summary_aggs).reset_index()
            avg_summary_df.rename(columns={'run_index': 'completed_runs'}, inplace=True)

            print("\nAverage Performance Metrics for this main.py call (config specific):")
            print(avg_summary_df.to_string(index=False))
            
            csv_filename = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, f"summary_avg_runs_S{args.master_seed}.csv")
            avg_summary_df.to_csv(csv_filename, index=False, float_format='%.2f')
            print(f"\nSummary CSV for this call saved to: {csv_filename}")
        except ImportError: print("Warning: pandas library not installed. Detailed table summary skipped.")
        except Exception as e: print(f"Error during pandas summary processing: {e}")

        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            def average_results(list_of_explored_runs, list_of_steps_runs, debug_name="UNSPECIFIED"):
                valid_explored_runs = [r for r in list_of_explored_runs if r and len(r) > 0]
                valid_steps_runs = [r for r in list_of_steps_runs if r and len(r) > 0]
                if not valid_explored_runs or not valid_steps_runs or len(valid_explored_runs) != len(valid_steps_runs):
                    return np.array([]), np.array([]), np.array([])
                max_step_all = 0
                for r in valid_steps_runs: 
                    if r : max_step_all = max(max_step_all, r[-1])
                if max_step_all == 0:
                    if all(len(r) == 1 and r[0] == 0 for r in valid_steps_runs) and all(len(r) == 1 for r in valid_explored_runs):
                        vals = [r[0] for r in valid_explored_runs if r]
                        return np.array([0]), np.array([np.mean(vals) if vals else np.nan]), np.array([np.std(vals) if vals else np.nan])
                    return np.array([]), np.array([]), np.array([])
                common_steps_axis_avg = np.arange(max_step_all + 1)
                aligned_explored_data_avg = []
                for i in range(len(valid_explored_runs)):
                    s, e = valid_steps_runs[i], valid_explored_runs[i]
                    if len(s) != len(e) or not s: continue 
                    l = e[0] if s[0] == 0 else np.nan 
                    us, ui = np.unique(s, return_index=True); ue = np.array(e)[ui]
                    if not common_steps_axis_avg.any():
                        if us.size == 1 and us[0] == 0 : aligned_explored_data_avg.append(ue)
                        continue
                    interp_vals = np.interp(common_steps_axis_avg, us, ue, left=l, right=ue[-1] if ue.size > 0 else np.nan)
                    aligned_explored_data_avg.append(interp_vals)
                if not aligned_explored_data_avg: 
                    if common_steps_axis_avg.size == 1 and common_steps_axis_avg[0] == 0:
                        vals = [r[0] for r_list in valid_explored_runs for r in r_list if r is not None] # Flatten and filter
                        if vals: return np.array([0]), np.array([np.mean(vals)]), np.array([np.std(vals)])
                    return np.array([]), np.array([]), np.array([])
                return common_steps_axis_avg, np.nanmean(np.array(aligned_explored_data_avg), axis=0), np.nanstd(np.array(aligned_explored_data_avg), axis=0)

            plot_colors = {'FBE': 'blue', 'ACO': 'red', 'IGE': 'green', 'URE': 'purple'}
            plt.figure(figsize=(12, 8))
            plotted_anything_on_this_fig_main = False
            max_x_lim_plot_fig_main = 0
            fig_title_main = (f"Map: {current_map_type_iter}, Size: {map_size_str_iter}, Obstacle Rate: {current_obs_perc_iter:.2f}\n"
                         f"Exploration Performance (Sensor: {args.robot_sensor_range}, Max Dyn Steps: {current_max_steps_calculated})\n"
                         f"Averaged over {args.num_runs} runs (Planners: {', '.join(planners_to_run_final)})")
            plt.title(fig_title_main, fontsize=11)

            for planner_name_plot_main in planners_to_run_final:
                plot_key_lookup_main = (planner_name_plot_main, map_size_str_iter, current_map_type_iter, current_obs_perc_iter)
                if plot_key_lookup_main not in exploration_data_for_this_main_call_plot or \
                   not exploration_data_for_this_main_call_plot[plot_key_lookup_main]['steps']:
                    continue
                results_data_main = exploration_data_for_this_main_call_plot[plot_key_lookup_main]
                if results_data_main['explored'] and any(run for run in results_data_main['explored'] if run):
                    x_axis_main, avg_explored_main, std_explored_main = average_results(
                        results_data_main['explored'], results_data_main['steps'], f"PlotData_{planner_name_plot_main}")
                    if len(x_axis_main) > 0 and len(avg_explored_main) > 0 and not np.all(np.isnan(avg_explored_main)):
                        if len(x_axis_main) > 0 and x_axis_main[-1] > max_x_lim_plot_fig_main: max_x_lim_plot_fig_main = x_axis_main[-1]
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
                plot_xlim_upper_fig_main = current_max_steps_calculated
                if max_x_lim_plot_fig_main > 0: plot_xlim_upper_fig_main = min(current_max_steps_calculated + 50, max_x_lim_plot_fig_main + int(max_x_lim_plot_fig_main * 0.05) + 50)
                plt.xlim(left=-max(1, int(0.01 * plot_xlim_upper_fig_main)), right=plot_xlim_upper_fig_main)
                plt.xticks(fontsize=10); plt.yticks(fontsize=10); plt.tight_layout(pad=1.8)
                plots_subdir_main = os.path.join(CONFIG_SPECIFIC_BASE_OUTPUT_DIR, "_plots_summary_for_this_config_call")
                os.makedirs(plots_subdir_main, exist_ok=True)
                plot_filename_base_main = (f"EXPL_ComparePlanners_Size{map_size_str_iter}_MapT{current_map_type_iter}_"
                                           f"Obs{current_obs_perc_iter:.2f}_S{args.master_seed}.png")
                plot_filename_full_main = os.path.join(plots_subdir_main, plot_filename_base_main)
                plt.savefig(plot_filename_full_main)
                print(f"Comparison plot for this configuration saved: {plot_filename_full_main}")
                plt.close()
            else: print(f"No data to plot for configuration: Size:{map_size_str_iter}, Map:{current_map_type_iter}, Obs:{current_obs_perc_iter:.2f}")
        except Exception as e_plot:
            print(f"Error during plotting: {e_plot}")