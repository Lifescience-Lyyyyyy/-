import pygame
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt
import shutil
import traceback # For detailed error printing

# --- All imports ---
try:
    from environment import Environment, FREE, OBSTACLE, UNKNOWN
    from robot import Robot
    from visualizer1 import Visualizer
    from planners.deep_aco_simple_planner import DeepACOSimplePlanner
except ImportError as e:
    print(f"FATAL: Import Error: {e}")
    print("Please ensure all required python files are present and accessible.")
    exit(1)

# --- All configurations (same as before) ---
MAP_WIDTH_TRAIN = 40; MAP_HEIGHT_TRAIN = 30; OBSTACLE_PERCENTAGE_TRAIN_RANGE = (0.15, 0.35)
MAP_TYPE_TRAIN = "random"; ROBOT_SENSOR_RANGE_TRAIN = 5
NUM_EPISODES = 5000; MAX_STEPS_PER_EPISODE = 500; PPO_UPDATE_TIMESTEP = 2048
USE_TIMESTEP_UPDATE = True
DEEP_ACO_SIMPLE_PARAMS = {
    'n_ants': 8, 'n_iterations': 5, 'alpha': 1.0, 'beta': 1.5,
    'evaporation_rate': 0.1, 'pheromone_initial': 0.1, 'pheromone_min': 0.01,
    'pheromone_max': 5.0, 'local_frontier_ig_sum_radius': 5,
    'robot_sensor_range': ROBOT_SENSOR_RANGE_TRAIN,
    'learning_rate_actor': 3e-4, 'learning_rate_critic': 1e-3, 'gamma_ppo': 0.99,
    'eps_clip_ppo': 0.2, 'ppo_epochs': 4, 'ppo_batch_size': 64, 'is_training': True
}
MODEL_SAVE_PATH = "deep_aco_simple_ppo_model.pth"
PRETRAINED_HEURISTIC_PATH = "heuristic_net_pretrained_manual_eta.pth"
RESULTS_DIR = "training_results_deep_aco_simple_ppo"
os.makedirs(RESULTS_DIR, exist_ok=True)
VISUALIZE_MODULO = 0 # Set to 0 to run faster without any Pygame window
CELL_SIZE_VIZ = 15; SIM_DELAY_VIZ = 0.01

def calculate_reward(newly_explored_cells, path_cost):
    if path_cost <= 0: path_cost = 1
    efficiency_reward = newly_explored_cells / path_cost 
    time_penalty = -0.02
    return efficiency_reward + time_penalty

def train_deep_aco_simple():
    print("--- CHECKPOINT 1: train_deep_aco_simple() function started ---")
    
    # Initialize Pygame only if absolutely necessary
    if VISUALIZE_MODULO > 0:
        print("--- CHECKPOINT 2: Pygame visualization is enabled, initializing Pygame... ---")
        pygame.init()

    # Initialize planner
    print("--- CHECKPOINT 3: Initializing dummy environment for planner setup... ---")
    temp_env = Environment(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, 0.2, "random")
    print("--- CHECKPOINT 4: Initializing DeepACOSimplePlanner... ---")
    planner = DeepACOSimplePlanner(temp_env, **DEEP_ACO_SIMPLE_PARAMS)
    print("--- CHECKPOINT 5: DeepACOSimplePlanner initialized. ---")

    # Load pre-trained model if it exists
    if os.path.exists(PRETRAINED_HEURISTIC_PATH):
        print(f"--- CHECKPOINT 6: Found pre-trained model at {PRETRAINED_HEURISTIC_PATH}, attempting to load... ---")
        try:
            planner.actor.load_state_dict(torch.load(PRETRAINED_HEURISTIC_PATH, map_location=planner.device))
            planner.policy_old.load_state_dict(planner.actor.state_dict())
            print(f"--- CHECKPOINT 7: Successfully loaded pre-trained ACTOR model. ---")
        except Exception as e:
            print(f"--- CHECKPOINT 7 FAILED: Could not load pre-trained actor model: {e}. Starting with random weights. ---")
    else:
        print(f"--- CHECKPOINT 6: No pre-trained model found at {PRETRAINED_HEURISTIC_PATH}. Starting with random weights. ---")

    timestep_counter = 0
    ep_rewards, ep_lengths, all_final_explored = [], [], []

    print(f"--- CHECKPOINT 8: Starting main training loop for {NUM_EPISODES} episodes... ---")
    for episode in range(1, NUM_EPISODES + 1):
        # Setup environment for the episode
        env_seed = episode 
        np.random.seed(env_seed)
        env = Environment(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, 
                          torch.random.uniform(0.15, 0.35), 
                          MAP_TYPE_TRAIN)
        robot = Robot(env.robot_start_pos_ref, ROBOT_SENSOR_RANGE_TRAIN)
        robot.sense(env)
        
        current_ep_reward = 0
        last_explored_count = np.sum(env.grid_known != UNKNOWN)
        
        show_this_ep = VISUALIZE_MODULO > 0 and episode % VISUALIZE_MODULO == 0
        visualizer = None
        if show_this_ep:
            try:
                visualizer = Visualizer(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, CELL_SIZE_VIZ)
                pygame.display.set_caption(f"PPO Training - Episode {episode}")
            except pygame.error as e:
                print(f"--- EPISODE {episode} WARNING: Visualizer init failed: {e} ---")
                visualizer = None

        # Run one episode
        for t in range(1, MAX_STEPS_PER_EPISODE + 1):
            timestep_counter += 1
            
            # --- Agent takes an action ---
            target_pos, path = planner.plan_next_action(robot.get_position(), env.grid_known)
            
            if target_pos is None or not path:
                print(f"--- EPISODE {episode} END: Planner failed to find a path at step {t}. ---")
                planner.memory.rewards.append(-1.0) 
                planner.memory.is_terminals.append(True)
                break
            
            # --- Environment executes the path ---
            path_cost = len(path) - 1
            robot_crashed = False
            for step_pos in path[1:]:
                if not robot.attempt_move_one_step(step_pos, env):
                    robot_crashed = True; break
            
            robot.sense(env)
            
            # --- Calculate reward ---
            new_explored_count = np.sum(env.grid_known != UNKNOWN)
            newly_explored = new_explored_count - last_explored_count
            last_explored_count = new_explored_count
            reward = calculate_reward(newly_explored, path_cost)
            if robot_crashed: reward -= 0.5
            current_ep_reward += reward
            
            is_done = (t == MAX_STEPS_PER_EPISODE) or (robot_crashed and path_cost > 0)
            planner.memory.rewards.append(reward)
            planner.memory.is_terminals.append(is_done)

            # --- PPO Update ---
            if USE_TIMESTEP_UPDATE and timestep_counter >= PPO_UPDATE_TIMESTEP:
                print(f"--- EPISODE {episode}, Timestep {timestep_counter}: Updating PPO policy... ---")
                planner.update_ppo()
                timestep_counter = 0

            if visualizer:
                if not visualizer.handle_events_simple():
                    print(f"--- EPISODE {episode} END: Visualization quit by user. ---")
                    break
                visualizer.clear_screen(); visualizer.draw_grid_map(env.grid_known)
                visualizer.draw_robot_path(robot.actual_pos_history)
                # visualizer.draw_robots([robot]) # Assuming this method exists
                visualizer.update_display(); time.sleep(SIM_DELAY_VIZ)

            if is_done:
                # print(f"--- EPISODE {episode} END: Reached terminal state at step {t}. ---")
                break
        
        # --- End of Episode Logging ---
        ep_rewards.append(current_ep_reward)
        ep_lengths.append(t)
        all_final_explored.append(np.sum(env.grid_known != UNKNOWN) / (env.width * env.height) * 100)
        
        if visualizer: visualizer.quit()
        TRAIN_EVERY_N_EPISODES = 1 if USE_TIMESTEP_UPDATE else 10
        if not USE_TIMESTEP_UPDATE and episode % TRAIN_EVERY_N_EPISODES == 0:
            if len(planner.memory.actions) > 0:
                print(f"--- EPISODE {episode}: Updating PPO policy (end of episode trigger)... ---")
                planner.update_ppo()

        if episode % 10 == 0:
            avg_reward = np.mean(ep_rewards[-10:])
            avg_length = np.mean(ep_lengths[-10:])
            avg_explored = np.mean(all_final_explored[-10:])
            print(f"Episode {episode} | Avg Length: {avg_length:.1f} | Avg Reward: {avg_reward:.2f} | Avg Explored: {avg_explored:.1f}%")
        
        if episode % 100 == 0:
            print(f"--- Saving model checkpoint at Episode {episode}... ---")
            planner.save_model(MODEL_SAVE_PATH)
    
    print("--- CHECKPOINT 9: Finished main training loop. ---")
    planner.save_model(MODEL_SAVE_PATH) 
    print(f"\n--- Training Finished ---")
    print(f"Final model saved to base path: {MODEL_SAVE_PATH}")

    # --- Plotting training progress ---
    if ep_rewards:
        try:
            # ... (Plotting logic from your last full version - ensure variables are defined) ...
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            def smooth(y, box_pts):
                if not isinstance(y, (list, np.ndarray)) or len(y) < box_pts : return np.array(y) 
                box = np.ones(box_pts)/box_pts
                return np.convolve(y, box, mode='valid')
            smoothing_window = max(1, len(ep_rewards) // 20)
            if smoothing_window == 0: smoothing_window = 1

            axs[0].plot(ep_rewards, label="Episode Reward", color='gray', alpha=0.4)
            rewards_smoothed = smooth(ep_rewards, smoothing_window) # Define before use
            if len(rewards_smoothed) > 0:
                axs[0].plot(np.arange(len(rewards_smoothed)) + smoothing_window - 1, rewards_smoothed, 
                            label=f"Smoothed Reward (win={smoothing_window})", color='green')
            axs[0].set_ylabel("Total Reward"); axs[0].set_title("Reward per Episode"); axs[0].legend(); axs[0].grid(True)
            
            axs[1].plot(ep_lengths, label="Episode Length", color='gray', alpha=0.4)
            lengths_smoothed = smooth(ep_lengths, smoothing_window) # Define before use
            if len(lengths_smoothed) > 0:
                axs[1].plot(np.arange(len(lengths_smoothed)) + smoothing_window - 1, lengths_smoothed, 
                            label=f"Smoothed Length (win={smoothing_window})", color='purple')
            axs[1].set_ylabel("Steps"); axs[1].set_title("Episode Length"); axs[1].legend(); axs[1].grid(True)
            
            axs[2].plot(all_final_explored, label="Final Exploration %", color='gray', alpha=0.4)
            expl_smoothed = smooth(all_final_explored, smoothing_window) # Define before use
            if len(expl_smoothed) > 0:
                axs[2].plot(np.arange(len(expl_smoothed)) + smoothing_window - 1, expl_smoothed, 
                            label=f"Smoothed Explored % (win={smoothing_window})", color='darkblue')
            axs[2].set_ylabel("Exploration %"); axs[2].set_title("Final Exploration Percentage per Episode"); axs[2].legend(); axs[2].grid(True); axs[2].set_ylim(0,105)

            axs[-1].set_xlabel("Episode")
            plt.tight_layout()
            plot_filename = os.path.join(RESULTS_DIR, "ppo_training_summary.png")
            plt.savefig(plot_filename); print(f"Training summary plot saved to {plot_filename}")
            plt.close(fig) 
        except Exception as e_plt: print(f"Error plotting training progress: {e_plt}")

    if pygame.get_init(): 
        pygame.quit()


if __name__ == "__main__":
    print("--- CHECKPOINT 0: Script starting... ---")
    if VISUALIZE_MODULO > 0 and not pygame.get_init():
        print("Pygame display required for visualization.")
        # Note: If no display is available, pygame.init() might fail here.
        try:
            pygame.init()
        except pygame.error as e:
            print(f"FATAL: Pygame could not be initialized: {e}. Cannot run with visualization.")
            print("Try running with VISUALIZE_MODULO = 0 for headless training.")
            exit(1)

    try:
        train_deep_aco_simple()
    except Exception as e_train_top:
        print(f"FATAL ERROR during top-level training script: {e_train_top}")
        traceback.print_exc()
    finally:
        print("--- Script Finished. ---")
        if pygame.get_init(): 
            pygame.quit()