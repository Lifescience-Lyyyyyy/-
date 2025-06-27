import pygame
import numpy as np
import time
import os
import random
import traceback

try:
    from environment_SLAM import Environment, FREE, OBSTACLE, LANDMARK, UNKNOWN
    from robot_SLAM import RobotSlamSimplified as Robot
    from visualizer_SLAM import Visualizer # Assumes this has draw_probabilistic_map and draw_pose_with_uncertainty
    # Use your original ACO exploration planner for path selection
    from planners.aco_planner import ACOPlanner 
except ImportError as e:
    print(f"Import Error: {e}. Ensure all modules are accessible."); exit(1)

# --- SLAM Demo Configuration ---
ENV_WIDTH_SLAM = 30
ENV_HEIGHT_SLAM = 30
NUM_LANDMARKS_SLAM = 8 # Number of landmarks to place
ROBOT_SENSOR_RANGE_SLAM = 8 
CELL_SIZE_SLAM = 15
SIM_DELAY_STEP_MS = 50

# Noise parameters
MOTION_NOISE_STD_DEV = 0.025      # Standard deviation of motion noise (as a fraction of step length)
OBSERVATION_POS_NOISE_STD = 0.03  # Standard deviation of landmark observation noise (in cells)
OBSERVATION_NOISE_PROB_FLIP = 0.001

# ACO Planner parameters (for exploration)
ACO_PARAMS_SLAM = {
    'n_ants': 15, 'n_iterations': 10,
    'alpha': 1.0, 'beta': 2.5, 'evaporation_rate': 0.1, 'q0': 0.8,
    'ig_weight_heuristic': 2.0, 
    'robot_sensor_range_for_heuristic': ROBOT_SENSOR_RANGE_SLAM
}

MAX_STEPS_SLAM = 100
MAX_CYCLES_SLAM = 700

def run_slam_aco_demo():
    print("--- Simplified SLAM Demo with Landmarks and ACO Planner ---")
    seed = 1025; np.random.seed(seed); random.seed(seed)

    env = Environment(ENV_WIDTH_SLAM, ENV_HEIGHT_SLAM, 
                      map_type="landmarks_only", # Use the new map type
                      num_landmarks=NUM_LANDMARKS_SLAM, env_seed=seed)
    
    robot = Robot(env.robot_start_pos_ref, ROBOT_SENSOR_RANGE_SLAM,
                                motion_noise_std=MOTION_NOISE_STD_DEV,
                                observation_pos_noise_std=OBSERVATION_POS_NOISE_STD,
                                observation_noise_prob_flip=OBSERVATION_NOISE_PROB_FLIP)
    
    visualizer = Visualizer(ENV_WIDTH_SLAM, ENV_HEIGHT_SLAM, CELL_SIZE_SLAM)
    
    # The exploration planner doesn't need to know it's a SLAM problem.
    # It just needs a discrete map to plan on.
    planner = ACOPlanner(env, **ACO_PARAMS_SLAM)

    total_steps = 0; total_cycles = 0; sim_active = True
    current_path = []

    # --- Main SLAM Loop ---
    while sim_active and total_steps < MAX_STEPS_SLAM and total_cycles < MAX_CYCLES_SLAM:
        if not visualizer.handle_events_simple(): sim_active = False; break

        # 1. Robot Senses its environment (updates probabilistic map, detects landmarks)
        observed_landmarks = robot.sense(env)

        # 2. SLAM: Data Association and Correction (Simplified)
        for obs in observed_landmarks:
            landmark_id = obs['id']       # True position used as ID
            observed_pos = obs['pos']   # Noisy world position observation

            if landmark_id in robot.landmark_database:
                # Loop Closure Detected! Correct robot and landmark poses.
                robot.correct_pose_and_landmarks(landmark_id, observed_pos)
            else: # First time seeing this landmark
                # Add to database with high initial uncertainty
                initial_landmark_cov = np.eye(2) * 1.5 
                robot.landmark_database[landmark_id] = {'pos': observed_pos, 'cov': initial_landmark_cov}

        # 3. Planning (based on robot's current belief of the map)
        if not current_path:
            # Planner needs a discrete map. Get it from the current probabilistic map.
            known_map_for_planner = env.get_map_for_planner()
            
            # Update planner's internal environment reference if it uses map dimensions
            planner.environment = env
            
            target_pos, path = planner.plan_next_action(robot.get_position(), known_map_for_planner)
            total_cycles += 1
            if not target_pos or not path: print("Planner failed. Ending."); break
            current_path = path[1:]
        
        # 4. Robot Execution (Move)
        if current_path:
            next_step = current_path.pop(0)
            moved = robot.attempt_move_one_step(next_step, env)
            total_steps += 1
            if not moved: # Collision with a landmark
                current_path = [] # Force replan

        # 5. Visualization
        visualizer.clear_screen()
        # Draw the probabilistic map first as a background
        visualizer.draw_probabilistic_map(env.get_occupancy_probability_map())
        
        # Draw true landmark positions for reference (e.g., small black dots)
        true_landmark_positions = np.argwhere(env.grid_true == LANDMARK)
        for r_lm, c_lm in true_landmark_positions:
             pygame.draw.circle(visualizer.screen, (0,0,0), 
                                (c_lm * CELL_SIZE_SLAM + CELL_SIZE_SLAM/2, r_lm * CELL_SIZE_SLAM + CELL_SIZE_SLAM/2), 
                                2)

        # Draw robot's believed pose and path history
        visualizer.draw_robot_path(robot.actual_pos_history, path_color=(0,0,200,150))
        visualizer.draw_pose_with_uncertainty(robot.believed_pos, robot.pose_covariance, (255,0,0), (255,100,100))

        # Draw estimated landmark positions with uncertainty
        for lm_id, lm_data in robot.landmark_database.items():
            visualizer.draw_pose_with_uncertainty(lm_data['pos'], lm_data['cov'], (255,215,0), (255,255,150))
        
        # Display stats
        explored_perc = np.sum(np.abs(env.log_odds_map) > 0.1) / (env.width * env.height) * 100
        visualizer.draw_text(f"Steps: {total_steps}/{MAX_STEPS_SLAM}", (10,10))
        visualizer.draw_text(f"Explored (Confidence > 10%): {explored_perc:.1f}%", (10,30))
        visualizer.update_display()
        time.sleep(SIM_DELAY_STEP_MS / 1000.0)

        if explored_perc >= 95.0 : sim_active = False; print("Exploration nearly complete.")

    print(f"SLAM Demo ended. Total physical steps: {total_steps}")
    
    # Final display loop
    running_final = True
    while running_final:
        if not visualizer.handle_events_simple(): running_final = False
        pygame.time.wait(100)
    visualizer.quit()

if __name__ == "__main__":
    if not pygame.get_init(): pygame.init()
    try:
        run_slam_aco_demo()
    except Exception as e:
        print(f"An error occurred: {e}"); traceback.print_exc()
    finally:
        if pygame.get_init(): pygame.quit()