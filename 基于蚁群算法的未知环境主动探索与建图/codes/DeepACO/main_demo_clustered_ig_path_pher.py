import pygame
import numpy as np
import time
import os
import sys
import shutil

try:
    from environment import Environment, FREE, OBSTACLE, UNKNOWN
    from robot import Robot 
    from visualizer1 import Visualizer
    from planners.ig_pheromone_path_planner import IGPheromonePathPlanner # Using the modified planner
    from planners.pathfinding import a_star_search 
except ImportError as e:
    print(f"Import Error in main_demo_clustered_ig_path_pher.py: {e}")
    exit(1)

# --- Demo Configuration ---
ENV_WIDTH_DEMO = 50
ENV_HEIGHT_DEMO = 50
OBSTACLE_PERCENTAGE_DEMO = 0.15
MAP_TYPE_DEMO = "random" 
ROBOT_START_POS_DEMO = (5, 5)
ROBOT_SENSOR_RANGE_DEMO = 5

CELL_SIZE_DEMO = 16
SIM_DELAY_ROBOT_STEP_MS = 50  
SIM_DELAY_ANT_VIZ_MS = 40     

ENABLE_VISUALIZATION_DEMO = True  
VISUALIZE_ANTS_CALLBACK_DEMO = True 
SCREENSHOT_INTERVAL_DEMO = 0    
SELECT_CORE_IG_POINT_DEMO = True

# IGPheromonePathPlanner Parameters
N_ANTS_PLANNER_DEMO = 18
N_ITERATIONS_PLANNER_DEMO = 8           
ALPHA_PLANNER_DEMO = 1.0                
BETA_ANT_TO_TARGET_PLANNER_DEMO = 1.5   
EVAP_RATE_PLANNER_DEMO = 0.05           
PHER_INIT_PLANNER_DEMO = 0.1            
PHER_MIN_PLANNER_DEMO = 0.01            
PHER_MAX_PLANNER_DEMO = 7.0             
Q_DEPOSIT_FACTOR_PLANNER_DEMO = 1.0      
IG_CALC_SENSOR_RANGE_PLANNER_DEMO = ROBOT_SENSOR_RANGE_DEMO 
ANT_MAX_STEPS_PLANNER_DEMO = 60         
ROBOT_NAV_STEPS_PLANNER_DEMO = 50 # Note: This planner version might not use this directly for robot decision
ALLOW_DIAGONAL_ANTS_PLANNER_DEMO = False 
ENABLE_CLUSTERING_PLANNER_DEMO = True
FRONTIER_CLUSTER_DIST_PLANNER_DEMO = 1 
CLUSTER_IG_SCALE_FACTOR_PLANNER_DEMO = 0.5 

MAX_SIM_ROBOT_STEPS_DEMO = 234
MAX_SIM_PLANNING_CYCLES_DEMO = 400

RECORD_FRAMES_DEMO = False 
FRAMES_DIR_DEMO = "clustered_ig_pher_frames_v3" # Changed dir name
VIDEO_OUT_DEMO = "clustered_ig_pher_demo_v3.mp4"
VIDEO_FPS_DEMO = 20

global_video_frame_count = 0
ROBOT_PATH_TRAIL_COLOR = (100, 100, 255, 100)

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def save_frame_for_video(viz, frame_dir, frame_idx):
    if viz and viz.screen and RECORD_FRAMES_DEMO:
        ensure_dir(frame_dir)
        pygame.image.save(viz.screen, os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))
        return frame_idx + 1
    return frame_idx

# This callback now expects all ant paths from all *internal* iterations of ONE planning call
def ig_pher_path_visualization_callback(
    start_node_arg=None, map_grid_ref_arg=None, 
    ant_paths_all_iterations_data_arg=None, 
    environment_ref_arg=None, 
    visualizer_instance_arg=None, 
    pheromone_map_to_display_arg=None,
    current_iteration_num_arg=0, # This is the outer (robot) planning cycle number
    total_iterations_arg=0       # This is n_iterations_sim (planner's internal iterations)
    ):
    global global_video_frame_count 

    if not visualizer_instance_arg or not ant_paths_all_iterations_data_arg: return

    num_internal_planner_iters_for_viz = len(ant_paths_all_iterations_data_arg)
    
    for internal_iter_idx_viz, paths_of_one_internal_iter_viz in enumerate(ant_paths_all_iterations_data_arg):
        if not visualizer_instance_arg.handle_events_simple(): 
            pygame.quit(); raise SystemExit("Visualization quit by user.")

        # Displaying (Outer Cycle X, Inner Ant Iter Y/Z)
        iter_text = f"Planning Cycle {current_iteration_num_arg + 1}, Ant Sim Iter {internal_iter_idx_viz + 1}/{num_internal_planner_iters_for_viz}"

        visualizer_instance_arg.draw_ant_paths_iteration_known_map( 
            static_map_grid=map_grid_ref_arg, 
            pheromone_map_for_display=pheromone_map_to_display_arg, 
            ant_paths_this_iter=paths_of_one_internal_iter_viz, 
            start_pos=start_node_arg, 
            goal_pos=None, 
            iteration_num=internal_iter_idx_viz, # Use internal iter for display count
            total_iterations_planner=num_internal_planner_iters_for_viz, # Total internal iters
            pher_display_max=PHER_MAX_PLANNER_DEMO 
        )
        # Override the default text with more context
        visualizer_instance_arg.draw_text(iter_text, 
                       (10, visualizer_instance_arg.screen_height - 30 if visualizer_instance_arg.screen_height > 60 else 10), 
                       color=pygame.Color("black"))
        visualizer_instance_arg.update_display()


        if RECORD_FRAMES_DEMO: 
            global_video_frame_count = save_frame_for_video(visualizer_instance_arg, FRAMES_DIR_DEMO, global_video_frame_count)
        
        pygame.time.wait(SIM_DELAY_ANT_VIZ_MS)


def run_ig_pheromone_path_exploration_demo():
    global global_video_frame_count
    global_video_frame_count = 0

    print("--- Clustered IG Pheromone Path Planner Demo ---") 
    if RECORD_FRAMES_DEMO:
        print(f"    Recording frames to: {FRAMES_DIR_DEMO}")
        if os.path.exists(FRAMES_DIR_DEMO): shutil.rmtree(FRAMES_DIR_DEMO)
        ensure_dir(FRAMES_DIR_DEMO)

    env = Environment(ENV_WIDTH_DEMO, ENV_HEIGHT_DEMO, 
                      OBSTACLE_PERCENTAGE_DEMO, map_type=MAP_TYPE_DEMO,
                      robot_start_pos_ref=ROBOT_START_POS_DEMO)
    
    robot = Robot(ROBOT_START_POS_DEMO, ROBOT_SENSOR_RANGE_DEMO)
    visualizer = None
    if ENABLE_VISUALIZATION_DEMO:
        try:
            visualizer = Visualizer(ENV_WIDTH_DEMO, ENV_HEIGHT_DEMO, CELL_SIZE_DEMO)
        except pygame.error as e:
            print(f"Visualizer init error: {e}. Running headless for this part.")
            visualizer = None
    
    planner_visualization_callback = None 
    if VISUALIZE_ANTS_CALLBACK_DEMO and visualizer: 
        planner_visualization_callback = lambda **cb_kwargs: \
            ig_pher_path_visualization_callback( 
                visualizer_instance_arg=visualizer,
                **cb_kwargs
            )

    planner = IGPheromonePathPlanner(
        environment=env,
        n_ants=N_ANTS_PLANNER_DEMO,
        n_iterations=N_ITERATIONS_PLANNER_DEMO, 
        alpha=ALPHA_PLANNER_DEMO,
        beta_ant_to_target_frontier=BETA_ANT_TO_TARGET_PLANNER_DEMO,
        evaporation_rate=EVAP_RATE_PLANNER_DEMO, 
        pheromone_initial=PHER_INIT_PLANNER_DEMO,
        pheromone_min=PHER_MIN_PLANNER_DEMO,    
        pheromone_max=PHER_MAX_PLANNER_DEMO,    
        q_deposit_factor=Q_DEPOSIT_FACTOR_PLANNER_DEMO,
        ig_calculation_sensor_range=IG_CALC_SENSOR_RANGE_PLANNER_DEMO,
        ant_max_steps_to_frontier=ANT_MAX_STEPS_PLANNER_DEMO, 
        max_pheromone_nav_steps=ROBOT_NAV_STEPS_PLANNER_DEMO,
        visualize_ants_callback=planner_visualization_callback,
        allow_diagonal_movement=ALLOW_DIAGONAL_ANTS_PLANNER_DEMO,
        enable_frontier_clustering=ENABLE_CLUSTERING_PLANNER_DEMO,
        frontier_clustering_distance=FRONTIER_CLUSTER_DIST_PLANNER_DEMO,
        cluster_ig_scaling_factor=CLUSTER_IG_SCALE_FACTOR_PLANNER_DEMO,
        select_core_ig_point_in_cluster=SELECT_CORE_IG_POINT_DEMO
    )

    total_robot_steps = 0
    total_robot_planning_cycles = 0
    sim_active_main = True
    current_robot_path_to_execute = []
    current_target_frontier_viz = None
    latest_frontier_clusters_for_viz = None # Store last clusters for drawing
    latest_cluster_reps_for_viz = None   # Store last representatives

    robot.sense(env) 
    if hasattr(planner, '_update_pheromones_for_obstacles'):
         planner._update_pheromones_for_obstacles(env.grid_known)

    try:
        while sim_active_main and total_robot_steps < MAX_SIM_ROBOT_STEPS_DEMO and \
              total_robot_planning_cycles < MAX_SIM_PLANNING_CYCLES_DEMO:
            
            if visualizer and not visualizer.handle_events_simple(): 
                sim_active_main = False; break

            if not current_robot_path_to_execute: 
                # print(f"\nPlanning Cycle {total_robot_planning_cycles + 1}: Robot at {robot.get_position()}")
                kwargs_for_planner = {'total_planning_cycles': total_robot_planning_cycles}
                
                plan_result = planner.plan_next_action(robot.get_position(), env.grid_known, **kwargs_for_planner)
                
                # Check if plan_result includes the planning_info dictionary
                if isinstance(plan_result, tuple) and len(plan_result) == 3:
                    target_pos, path_to_target, planning_info = plan_result
                    if planning_info: # Ensure planning_info is not None
                        latest_frontier_clusters_for_viz = planning_info.get('frontier_clusters')
                        latest_cluster_reps_for_viz = planning_info.get('cluster_representatives')
                elif isinstance(plan_result, tuple) and len(plan_result) == 2: # Older planner not returning info
                    target_pos, path_to_target = plan_result
                    latest_frontier_clusters_for_viz = None
                    latest_cluster_reps_for_viz = None
                else: # Unexpected result
                    target_pos, path_to_target = None, None
                    latest_frontier_clusters_for_viz = None
                    latest_cluster_reps_for_viz = None


                total_robot_planning_cycles += 1

                if target_pos and path_to_target:
                    current_robot_path_to_execute = path_to_target[1:] 
                    current_target_frontier_viz = target_pos
                    if not current_robot_path_to_execute: 
                        # print("  Planner returned trivial path.")
                        pass # Allow replan in next cycle
                else:
                    print("  Planner could not find a new target/path. Ending exploration.")
                    sim_active_main = False
            
            if current_robot_path_to_execute and sim_active_main:
                next_step_robot = current_robot_path_to_execute.pop(0)
                moved = robot.attempt_move_one_step(next_step_robot, env); total_robot_steps += 1
                if moved:
                    robot.sense(env)
                    if hasattr(planner, '_update_pheromones_for_obstacles'): planner._update_pheromones_for_obstacles(env.grid_known)
                else: 
                    current_robot_path_to_execute = [] ; current_target_frontier_viz = None; robot.sense(env) 
                    if hasattr(planner, '_update_pheromones_for_obstacles'): planner._update_pheromones_for_obstacles(env.grid_known)
            
            if visualizer: 
                visualizer.clear_screen()
                visualizer.draw_grid_map(env.grid_known) # Changed from draw_grid_map
                
                if hasattr(planner, 'pheromone_map') and hasattr(visualizer, 'draw_pheromone_map_overlay'):
                     pher_map_display = planner.pheromone_map
                     max_pher_for_viz = np.max(pher_map_display[pher_map_display != float('inf')]) if np.any(pher_map_display != float('inf')) else PHER_MAX_PLANNER_DEMO
                     visualizer.draw_pheromone_map_overlay(pher_map_display, env.grid_known, pher_max_val_display=max(max_pher_for_viz, PHER_MIN_PLANNER_DEMO + 0.1))

                # Draw clusters and representatives IF available
                if latest_frontier_clusters_for_viz and hasattr(visualizer, 'draw_frontier_clusters'):
                    visualizer.draw_frontier_clusters(latest_frontier_clusters_for_viz, latest_cluster_reps_for_viz)


                visualizer.draw_positions(robot.actual_pos_history, ROBOT_PATH_TRAIL_COLOR, is_path=True, line_thickness_factor=0.08)
                visualizer.draw_positions([robot.get_position()], (255,0,0), radius_factor=0.4) 
                if current_target_frontier_viz:
                    visualizer.draw_positions([current_target_frontier_viz], (255,165,0, 200) , radius_factor=0.3)

                if current_robot_path_to_execute: 
                    full_viz_path_robot = [robot.get_position()] + current_robot_path_to_execute
                    visualizer.draw_positions(full_viz_path_robot, (0,200,200, 150), is_path=True, line_thickness_factor=0.06)

                explored_area = np.sum(env.grid_known != UNKNOWN); total_map_area = env.width * env.height
                explored_perc = (explored_area / total_map_area) * 100
                visualizer.draw_text(f"Robot Steps: {total_robot_steps}/{MAX_SIM_ROBOT_STEPS_DEMO}", (10,10))
                visualizer.draw_text(f"Explored: {explored_perc:.1f}%", (10,30))
                if hasattr(planner, 'current_phase'): # For structured planners
                     phase_text = f"Phase: {planner.current_phase}"
                     if hasattr(planner, 'current_ring_level'): phase_text += f", Ring: {planner.current_ring_level}"
                     elif hasattr(planner, '_boxing_current_dir_idx'): phase_text += f", BoxDir: {planner._boxing_current_dir_idx}"
                     visualizer.draw_text(phase_text, (10,50))


                visualizer.update_display()

            if SCREENSHOT_INTERVAL_DEMO > 0 and total_robot_steps > 0 and \
               total_robot_steps % SCREENSHOT_INTERVAL_DEMO == 0:
                global_video_frame_count = save_frame_for_video(visualizer, FRAMES_DIR_DEMO, global_video_frame_count)
            
            if visualizer: time.sleep(SIM_DELAY_ROBOT_STEP_MS / 1000.0)
            if explored_perc >= 99.0: sim_active_main = False; print("Exploration nearly complete.")

    except SystemExit: print("Demo exited by user from visualization callback.")
    except Exception as e: print(f"An error occurred during the demo main loop: {e}"); import traceback; traceback.print_exc()
    finally:
        print(f"Simulation ended. Total robot steps: {total_robot_steps}, Total planning cycles: {total_robot_planning_cycles}")
        if RECORD_FRAMES_DEMO and global_video_frame_count > 0:
            if visualizer and pygame.display.get_init():
                 visualizer.clear_screen(); visualizer.draw_static_map(env.grid_known)
                 if hasattr(planner, 'pheromone_map') and hasattr(visualizer, 'draw_pheromone_map_overlay'): visualizer.draw_pheromone_map_overlay(planner.pheromone_map, env.grid_known)
                 visualizer.draw_positions(robot.actual_pos_history, ROBOT_PATH_TRAIL_COLOR, is_path=True)
                 visualizer.draw_positions([robot.get_position()], (255,0,0), radius_factor=0.4)
                 if latest_frontier_clusters_for_viz and hasattr(visualizer, 'draw_frontier_clusters'): 
                    visualizer.draw_frontier_clusters(latest_frontier_clusters_for_viz, latest_cluster_reps_for_viz)
                 visualizer.update_display()
                 global_video_frame_count = save_frame_for_video(visualizer, FRAMES_DIR_DEMO, global_video_frame_count)
            print(f"\nFrames for video saved in '{FRAMES_DIR_DEMO}'. Total frames: {global_video_frame_count}")
            print(f"To create video: ffmpeg -framerate {VIDEO_FPS_DEMO} -i \"{FRAMES_DIR_DEMO}/frame_%06d.png\" -c:v libx264 -pix_fmt yuv420p \"{VIDEO_OUT_DEMO}\"")
        if visualizer and pygame.display.get_init():
            if sim_active_main or total_robot_steps < MAX_SIM_ROBOT_STEPS_DEMO : 
                print("Displaying final state. Press ESC or close window to exit.")
                running_final = True; clock = pygame.time.Clock()
                while running_final:
                    if not visualizer.handle_events_simple(): running_final = False
                    pygame.display.flip(); clock.tick(30) 
            visualizer.quit()

if __name__ == "__main__":
    if not pygame.get_init(): pygame.init()
    try:
        run_ig_pheromone_path_exploration_demo() 
    except Exception as e:
        print(f"An error occurred at the top level: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame.get_init(): pygame.quit()