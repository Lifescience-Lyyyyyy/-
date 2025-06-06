# main_demo_aco_known_map.py
import pygame
import numpy as np
import time
import os
import sys 

try:
    from environment import Environment, FREE, OBSTACLE, UNKNOWN
    from visualizer import Visualizer
    from planners.aco_known_map_planner import ACOKnownMapPlanner
except ImportError as e:
    print(f"Import Error: {e}. Make sure all required files are in the correct paths.")
    exit(1)

# --- Demo Configuration (保持不变) ---
ENV_WIDTH_DEMO = 50 
ENV_HEIGHT_DEMO = 50
OBSTACLE_PERCENTAGE_DEMO = 0.15
MAP_TYPE_DEMO = "random" 
ROBOT_START_POS_DEMO = (2, 2) 
ROBOT_GOAL_POS_DEMO = (ENV_HEIGHT_DEMO - 3, ENV_WIDTH_DEMO - 3)
CELL_SIZE_DEMO = 20 
SIM_DELAY_ANT_ITERATION_MS_DEMO = 50 
VISUALIZE_ANT_PATHS_ITERATIONS = True 
ALLOW_DIAGONAL_DEMO = False 
ACO_ANTS_DEMO = 20
ACO_ITERATIONS_DEMO = 50 
ACO_ALPHA_DEMO = 1.0      
ACO_BETA_DEMO = 2.5      
ACO_EVAPORATION_DEMO = 0.3
ACO_Q_DEPOSIT_DEMO = 100.0
ACO_PHER_MIN_DEMO = 0.01
ACO_PHER_MAX_DEMO = 20.0
USE_ELITIST_ANT_SYSTEM_DEMO = True
ELITIST_WEIGHT_FACTOR_DEMO = 2.0


def per_iteration_visualization_callback(
    # 这些参数现在将从 kwargs 中提取
    start_node_arg=None,          
    map_grid_ref_arg=None,               
    ant_paths_this_iteration_arg=None, # 修改这里，接收单次迭代的路径
    environment_ref_arg=None,     
    visualizer_instance_arg=None, 
    pheromone_map_to_display_arg=None,
    current_iteration_num_arg=0,      
    total_iterations_arg=0             
    ):
    """
    Callback for ACOKnownMapPlanner to visualize a SINGLE ant iteration and current pheromones.
    """
    if not visualizer_instance_arg or not ant_paths_this_iteration_arg:
        # print("Debug: per_iteration_visualization_callback missing visualizer or ant_paths_this_iteration_arg.")
        return

    # print(f"Debug: Callback iter {current_iteration_num_arg+1}/{total_iterations_arg}, Paths: {len(ant_paths_this_iteration_arg)}")

    if not visualizer_instance_arg.handle_events_simple(): 
        pygame.quit()
        # print("Visualization quit by user during ant path display.")
        raise SystemExit("Visualization quit by user.")

    visualizer_instance_arg.draw_ant_paths_iteration_known_map(
        static_map_grid=map_grid_ref_arg,
        pheromone_map_for_display=pheromone_map_to_display_arg, 
        ant_paths_this_iter=ant_paths_this_iteration_arg, # Paths from current iteration
        start_pos=ROBOT_START_POS_DEMO, # Use global demo config
        goal_pos=ROBOT_GOAL_POS_DEMO,   # Use global demo config
        iteration_num=current_iteration_num_arg, 
        total_iterations_planner=total_iterations_arg, 
        pher_display_max=ACO_PHER_MAX_DEMO # Use global demo config
    )
    pygame.time.wait(SIM_DELAY_ANT_ITERATION_MS_DEMO)


def run_aco_known_map_demo():
    print("--- ACO Pathfinding on Known Map Demo (Iterative Visualization) ---")
    print(f"    Movement: {'8-directional' if ALLOW_DIAGONAL_DEMO else '4-directional (Cardinal)'}")
    print(f"    Elitist Ant System: {'Enabled' if USE_ELITIST_ANT_SYSTEM_DEMO else 'Disabled'}")

    env = Environment(ENV_WIDTH_DEMO, ENV_HEIGHT_DEMO, 
                      OBSTACLE_PERCENTAGE_DEMO, map_type=MAP_TYPE_DEMO,
                      robot_start_pos_ref=ROBOT_START_POS_DEMO)

    if env.grid_true[ROBOT_GOAL_POS_DEMO[0], ROBOT_GOAL_POS_DEMO[1]] == OBSTACLE:
        print(f"  Goal {ROBOT_GOAL_POS_DEMO} was obstacle, forcing FREE.")
        env.grid_true[ROBOT_GOAL_POS_DEMO[0], ROBOT_GOAL_POS_DEMO[1]] = FREE
    
    static_map_for_planner = np.copy(env.grid_true)

    visualizer = Visualizer(ENV_WIDTH_DEMO, ENV_HEIGHT_DEMO, CELL_SIZE_DEMO)
    
    aco_viz_callback_for_planner = None
    if VISUALIZE_ANT_PATHS_ITERATIONS: 
        # Lambda now accepts all arguments as keyword arguments from the planner's call
        # and passes them to the callback function.
        aco_viz_callback_for_planner = lambda **cb_kwargs_from_planner: \
            per_iteration_visualization_callback(
                visualizer_instance_arg=visualizer, # Pass the captured visualizer
                **cb_kwargs_from_planner # Pass all other keyword args received from planner
            )

    aco_planner = ACOKnownMapPlanner(
        static_known_map=static_map_for_planner,
        map_height=ENV_HEIGHT_DEMO, map_width=ENV_WIDTH_DEMO,
        start_pos=ROBOT_START_POS_DEMO, goal_pos=ROBOT_GOAL_POS_DEMO,
        n_ants=ACO_ANTS_DEMO, n_iterations=ACO_ITERATIONS_DEMO,
        alpha=ACO_ALPHA_DEMO, beta=ACO_BETA_DEMO,
        evaporation_rate=ACO_EVAPORATION_DEMO,
        q_deposit=ACO_Q_DEPOSIT_DEMO,
        pheromone_min=ACO_PHER_MIN_DEMO, pheromone_max=ACO_PHER_MAX_DEMO,
        visualize_ants_callback=aco_viz_callback_for_planner,
        allow_diagonal_movement=ALLOW_DIAGONAL_DEMO,
        use_elitist_ant_system=USE_ELITIST_ANT_SYSTEM_DEMO,
        elitist_weight_factor=ELITIST_WEIGHT_FACTOR_DEMO
    )
    print(f"  ACO Planner initialized. Start: {ROBOT_START_POS_DEMO}, Goal: {ROBOT_GOAL_POS_DEMO}")
    print(f"  Running {ACO_ITERATIONS_DEMO} iterations with {ACO_ANTS_DEMO} ants each...")

    start_time = time.time()
    final_aco_path = None 
    try:
        _target_goal_pos, final_aco_path = aco_planner.find_path()
    except SystemExit as e: 
        print(f"Exiting due to: {e}")
    except Exception as e:
        print(f"Error during ACO pathfinding: {e}")
        import traceback
        traceback.print_exc()

    end_time = time.time()
    print(f"  ACO pathfinding process completed in {end_time - start_time:.2f} seconds.")

    if final_aco_path:
        print(f"  Path found by ACO (length {len(final_aco_path)-1}).")
    else:
        print("  ACO could not find a path to the goal (or visualization was quit).")

    # Display final path and pheromone map if visualizer is available
    if visualizer:
        final_pheromone_map = aco_planner.pheromone_map 
        visualizer.draw_final_path(static_map_for_planner, final_pheromone_map, 
                                   final_aco_path, ROBOT_START_POS_DEMO, ROBOT_GOAL_POS_DEMO,
                                   pher_display_max=ACO_PHER_MAX_DEMO) 
        
        running_final_show = True
        print("Displaying final path. Press ESC or close window to exit.")
        while running_final_show:
            if not visualizer.handle_events_simple():
                running_final_show = False
            pygame.time.wait(100)
        visualizer.quit() # Quit visualizer here if it was used

if __name__ == "__main__":
    if not pygame.get_init():
        pygame.init()
    try:
        run_aco_known_map_demo()
    except Exception as e:
        print(f"An error occurred at the top level of the demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame.get_init(): 
            pygame.quit()