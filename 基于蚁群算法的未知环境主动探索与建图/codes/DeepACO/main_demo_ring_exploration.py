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
    from planners.ring_exploration_planner import RingExplorationPlanner # NEW PLANNER
    from planners.pathfinding import a_star_search 
except ImportError as e:
    print(f"Import Error in main_demo_ring_exploration.py: {e}")
    exit(1)

# --- Demo Configuration ---
ENV_WIDTH_DEMO = 100  # Larger map to see ring effect
ENV_HEIGHT_DEMO = 100
OBSTACLE_PERCENTAGE_DEMO = 0.20 # Fewer obstacles to allow clearer rings
MAP_TYPE_DEMO = "random" 
# Start near a corner to make initial ring traversal more intuitive for demo
ROBOT_START_POS_DEMO = (5, 5) 
ROBOT_SENSOR_RANGE_DEMO = 5

CELL_SIZE_DEMO = 10
SIM_DELAY_ROBOT_STEP_MS = 50  
SIM_DELAY_PLANNING_MS = 10     # Planning for this planner is relatively fast

VISUALIZE_DEMO = True 
SCREENSHOT_INTERVAL_MAIN_STEPS = 0 # e.g., 20 robot steps for screenshots

# RingExplorationPlanner specific parameters (can be tuned)
FRONTIER_CLUSTERING_DIST_DEMO = ROBOT_SENSOR_RANGE_DEMO // 2 # For IGE-like part if used

MAX_SIM_ROBOT_STEPS = 1144
MAX_SIM_PLANNING_CYCLES = 800

RECORD_FRAMES_DEMO = False 
FRAMES_DIR_DEMO = "ring_explore_frames"
VIDEO_OUT_DEMO = "ring_explore_demo.mp4"
VIDEO_FPS_DEMO = 15

global_video_frame_count = 0

def ensure_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def save_frame_for_video(viz, frame_dir, frame_idx):
    if viz and viz.screen and RECORD_FRAMES_DEMO:
        ensure_dir(frame_dir)
        pygame.image.save(viz.screen, os.path.join(frame_dir, f"frame_{frame_idx:06d}.png"))
        return frame_idx + 1
    return frame_idx

def run_ring_exploration_demo():
    global global_video_frame_count
    global_video_frame_count = 0

    print("--- Ring Exploration Planner Demo ---")
    if RECORD_FRAMES_DEMO:
        print(f"    Recording frames to: {FRAMES_DIR_DEMO}")
        if os.path.exists(FRAMES_DIR_DEMO): shutil.rmtree(FRAMES_DIR_DEMO)
        ensure_dir(FRAMES_DIR_DEMO)

    env = Environment(ENV_WIDTH_DEMO, ENV_HEIGHT_DEMO, 
                      OBSTACLE_PERCENTAGE_DEMO, map_type=MAP_TYPE_DEMO,
                      robot_start_pos_ref=ROBOT_START_POS_DEMO)
    
    robot = Robot(ROBOT_START_POS_DEMO, ROBOT_SENSOR_RANGE_DEMO)
    visualizer = None
    if VISUALIZE_DEMO:
        try:
            visualizer = Visualizer(ENV_WIDTH_DEMO, ENV_HEIGHT_DEMO, CELL_SIZE_DEMO)
        except pygame.error as e:
            print(f"Visualizer init error: {e}. Running without viz.")
            visualizer = None # Fallback
    
    planner = RingExplorationPlanner(
        environment=env,
        robot_sensor_range=ROBOT_SENSOR_RANGE_DEMO
        # frontier_clustering_distance can be added if RingPlanner uses it internally
    )

    total_robot_steps = 0
    total_robot_planning_cycles = 0
    sim_active_main = True
    current_robot_path_to_execute = []
    current_target_frontier_viz = None
    
    robot.sense(env) 

    try:
        while sim_active_main and total_robot_steps < MAX_SIM_ROBOT_STEPS and \
              total_robot_planning_cycles < MAX_SIM_PLANNING_CYCLES:
            
            if visualizer and not visualizer.handle_events_simple(): 
                sim_active_main = False; break

            if not current_robot_path_to_execute: 
                # print(f"\nPlanning Cycle {total_robot_planning_cycles + 1}: Robot at {robot.get_position()}")
                
                target_pos, path_to_target = planner.plan_next_action(robot.get_position(), env.grid_known)
                total_robot_planning_cycles += 1

                if target_pos and path_to_target:
                    # print(f"  Robot's new target frontier: {target_pos}, A* path length: {len(path_to_target)-1}")
                    current_robot_path_to_execute = path_to_target[1:] 
                    current_target_frontier_viz = target_pos
                    if not current_robot_path_to_execute: 
                        print("  Planner returned trivial path. May be stuck or finished.")
                        # Allow one more planning cycle to see if it resolves (e.g. moves to next ring)
                        # If it repeats, it will likely hit max_cycles or no frontiers.
                else:
                    print("  Planner could not find a new target/path. Ending exploration.")
                    sim_active_main = False
            
            if current_robot_path_to_execute and sim_active_main:
                next_step_robot = current_robot_path_to_execute.pop(0)
                moved = robot.attempt_move_one_step(next_step_robot, env)
                total_robot_steps += 1
                
                if moved:
                    robot.sense(env)
                else: 
                    # print(f"  Collision at {next_step_robot}! Robot at {robot.get_position()}. Clearing path.")
                    current_robot_path_to_execute = [] # Force replan
                    current_target_frontier_viz = None
                    robot.sense(env) 
            
            if visualizer:
                visualizer.clear_screen()
                visualizer.draw_grid_map(env.grid_known) 
                
                # Draw current ring boundaries for debugging/visualization
                if hasattr(planner, 'current_ring_min_r'):
                    color_ring = (255, 100, 100, 100) # Light red transparent
                    # Top line
                    pygame.draw.line(visualizer.screen, color_ring[:3], 
                                     (planner.current_ring_min_c * CELL_SIZE_DEMO, planner.current_ring_min_r * CELL_SIZE_DEMO), 
                                     ((planner.current_ring_max_c+1) * CELL_SIZE_DEMO, planner.current_ring_min_r * CELL_SIZE_DEMO))
                    # Bottom line
                    pygame.draw.line(visualizer.screen, color_ring[:3],
                                     (planner.current_ring_min_c * CELL_SIZE_DEMO, (planner.current_ring_max_r+1) * CELL_SIZE_DEMO),
                                     ((planner.current_ring_max_c+1) * CELL_SIZE_DEMO, (planner.current_ring_max_r+1) * CELL_SIZE_DEMO))
                    # Left line
                    pygame.draw.line(visualizer.screen, color_ring[:3],
                                     (planner.current_ring_min_c * CELL_SIZE_DEMO, planner.current_ring_min_r * CELL_SIZE_DEMO),
                                     (planner.current_ring_min_c * CELL_SIZE_DEMO, (planner.current_ring_max_r+1) * CELL_SIZE_DEMO))
                    # Right line
                    pygame.draw.line(visualizer.screen, color_ring[:3],
                                     ((planner.current_ring_max_c+1) * CELL_SIZE_DEMO, planner.current_ring_min_r * CELL_SIZE_DEMO),
                                     ((planner.current_ring_max_c+1) * CELL_SIZE_DEMO, (planner.current_ring_max_r+1) * CELL_SIZE_DEMO))


                visualizer.draw_positions(robot.actual_pos_history, (100,100,255,100), is_path=True, line_thickness_factor=0.08)
                visualizer.draw_positions([robot.get_position()], (255,0,0), radius_factor=0.4) 
                if current_target_frontier_viz:
                    visualizer.draw_positions([current_target_frontier_viz], (255,165,0, 200) , radius_factor=0.3) # Target frontier

                if current_robot_path_to_execute: 
                    full_viz_path_robot = [robot.get_position()] + current_robot_path_to_execute
                    visualizer.draw_positions(full_viz_path_robot, (0,200,200, 150), is_path=True, line_thickness_factor=0.06)

                explored_area = np.sum(env.grid_known != UNKNOWN)
                total_map_area = env.width * env.height
                explored_perc = (explored_area / total_map_area) * 100
                visualizer.draw_text(f"Robot Steps: {total_robot_steps}/{MAX_SIM_ROBOT_STEPS}", (10,10))
                visualizer.draw_text(f"Explored: {explored_perc:.1f}%", (10,30))
                if hasattr(planner, 'current_ring_level'):
                     visualizer.draw_text(f"Ring Level: {planner.current_ring_level}", (10,50))

                visualizer.update_display()

            if SCREENSHOT_INTERVAL_MAIN_STEPS > 0 and total_robot_steps > 0 and total_robot_steps % SCREENSHOT_INTERVAL_MAIN_STEPS == 0:
                global_video_frame_count = save_frame_for_video(visualizer, FRAMES_DIR_DEMO, global_video_frame_count)
            
            if visualizer: time.sleep(SIM_DELAY_ROBOT_STEP_MS / 1000.0)
            if explored_perc >= 99.5: sim_active_main = False; print("Exploration nearly complete.")

    except SystemExit: 
        print("Demo exited by user from visualization callback.")
    except Exception as e:
        print(f"An error occurred during the demo main loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"Simulation ended. Total robot steps: {total_robot_steps}, Total planning cycles: {total_robot_planning_cycles}")
        if RECORD_FRAMES_DEMO and global_video_frame_count > 0:
            if visualizer and pygame.display.get_init(): # Save final frame if viz is still up
                 visualizer.clear_screen(); visualizer.draw_grid_map(env.grid_known)
                 if hasattr(planner, 'pheromone_map'): visualizer.draw_pheromone_map_overlay(planner.pheromone_map, env.grid_known)
                 visualizer.draw_positions(robot.actual_pos_history, PATH_TRAIL_COLOR, is_path=True)
                 visualizer.draw_positions([robot.get_position()], (255,0,0), radius_factor=0.4)
                 visualizer.update_display()
                 global_video_frame_count = save_frame_for_video(visualizer, FRAMES_DIR_DEMO, global_video_frame_count)

            print(f"\nFrames for video saved in '{FRAMES_DIR_DEMO}'. Total frames: {global_video_frame_count}")
            print(f"To create video: ffmpeg -framerate {VIDEO_FPS_DEMO} -i \"{FRAMES_DIR_DEMO}/frame_%06d.png\" -c:v libx264 -pix_fmt yuv420p \"{VIDEO_OUT_DEMO}\"")
        
        if visualizer and pygame.display.get_init():
            if sim_active_main or total_robot_steps < MAX_SIM_ROBOT_STEPS : 
                print("Displaying final state. Press ESC or close window to exit.")
                running_final = True; clock = pygame.time.Clock()
                while running_final:
                    if not visualizer.handle_events_simple(): running_final = False
                    # No need to redraw if static, but keep event loop responsive
                    # visualizer.update_display() # Already drawn by draw_final_path or last loop state
                    clock.tick(30) # Limit FPS for this simple wait loop
            visualizer.quit()

if __name__ == "__main__":
    if not pygame.get_init(): pygame.init()
    try:
        run_ring_exploration_demo()
    except Exception as e:
        print(f"An error occurred at the top level: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame.get_init(): pygame.quit()