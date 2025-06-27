import pygame
import numpy as np
import time
import os
import torch
import matplotlib.pyplot as plt
import shutil
import traceback
import random

try:
    from environment import Environment, FREE, OBSTACLE, UNKNOWN
    from robot import Robot
    from visualizer1 import Visualizer
    from planners.deep_aco_simple_planner import DeepACOSimplePlanner
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

# --- Training Configuration ---
MAP_WIDTH_TRAIN = 30
MAP_HEIGHT_TRAIN = 30
OBSTACLE_PERCENTAGE_TRAIN_RANGE = (0.15, 0.35)
MAP_TYPE_TRAIN = "random"
ROBOT_SENSOR_RANGE_TRAIN = 5

NUM_EPISODES = 2000
# MAX_STEPS_PER_EPISODE is now the limit for PHYSICAL steps, not planning cycles
MAX_PHYSICAL_STEPS_PER_EPISODE = 100 # This is the maximum number of physical steps the robot can take in an episode 
MAX_PLANNING_CYCLES_PER_EPISODE = 300 # Separate limit for planning cycles to prevent infinite planning loops

PPO_UPDATE_TIMESTEP = 234
USE_TIMESTEP_UPDATE = False
TRAIN_EVERY_N_EPISODES = 10

DEEP_ACO_SIMPLE_PARAMS = {
    'n_ants': 8, 'n_iterations': 5, 'alpha': 1.0, 'beta': 2,
    'evaporation_rate': 0.1, 'pheromone_initial': 0.1, 'pheromone_min': 0.01,
    'pheromone_max': 5.0, 'local_frontier_ig_sum_radius': 5,
    'robot_sensor_range': ROBOT_SENSOR_RANGE_TRAIN,
    'learning_rate_actor': 3e-4, 'learning_rate_critic': 1e-3, 'gamma_ppo': 0.99,
    'eps_clip_ppo': 0.2, 'ppo_epochs': 4, 'ppo_batch_size': 64,
    'is_training': True 
}

MODEL_SAVE_PATH = "deep_aco_simple_ppo_model.pth"
PRETRAINED_HEURISTIC_PATH = "heuristic_net_pretrained_manual_eta.pth"
RESULTS_DIR = "training_results_deep_aco_simple_ppo_v2" # New output dir
os.makedirs(RESULTS_DIR, exist_ok=True)
VISUALIZE_MODULO = 100 
CELL_SIZE_VIZ = 15
SIM_DELAY_VIZ = 0.01

def calculate_reward(newly_explored_cells, path_taken, visited_cells_global):
    """
    Calculates reward based on the new formula:
    Reward = (newly observed cells) - 3 * (repeated cells in path) / (total cells in path)
    """
    # 1. 新观察到的格子数
    reward = float(newly_explored_cells)

    # 2. 走的格子数惩罚
    path_length_penalty = len(path_taken)
    

    # 3. 重复走的格子数惩罚
    repeated_cells_count = 0
    if path_taken:
        # The first cell of a path segment is the robot's current position,
        # which is always "repeated". We care more about new steps into repeated territory.
        # Let's count how many cells in the path *after the first one* were already visited.
        for cell in path_taken[1:]:
            if cell in visited_cells_global:
                repeated_cells_count += 1
    
    repeat_penalty = 3 * repeated_cells_count
    reward -= repeat_penalty
    reward /= path_length_penalty if path_length_penalty > 0 else 1.0 # Avoid division by zero

    return reward

def train_deep_aco_simple():
    print("--- Training DeepACOSimplePlanner with PPO ---")
    if VISUALIZE_MODULO > 0 and not pygame.get_init(): pygame.init()

    # Initialize planner with a dummy environment, but pass correct max steps for normalization
    temp_env = Environment(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, 0.15, "random")
    # # Add max_simulation_steps_for_norm to the params if your planner uses it
    # DEEP_ACO_SIMPLE_PARAMS['max_simulation_steps_for_norm'] = MAX_PHYSICAL_STEPS_PER_EPISODE
    planner = DeepACOSimplePlanner(temp_env, **DEEP_ACO_SIMPLE_PARAMS)

    # Load pre-trained model if it exists
    if os.path.exists(PRETRAINED_HEURISTIC_PATH):
        try:
            planner.actor.load_state_dict(torch.load(PRETRAINED_HEURISTIC_PATH, map_location=planner.device))
            if planner.policy_old: planner.policy_old.load_state_dict(planner.actor.state_dict())
            print(f"Successfully loaded pre-trained ACTOR model from: {PRETRAINED_HEURISTIC_PATH}")
        except Exception as e: print(f"Could not load pre-trained actor model: {e}. Starting with random weights.")
    else: print(f"No pre-trained model found at {PRETRAINED_HEURISTIC_PATH}. Starting with random weights.")

    timestep_counter = 0 # Tracks total physical steps across episodes for PPO update
    episode_rewards_history, episode_lengths_history, all_final_explored = [], [], []

    for episode in range(1, NUM_EPISODES + 1):
        # --- Episode Setup ---
        env_seed = 82
        np.random.seed(env_seed); random.seed(env_seed)
        env = Environment(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, 
                          0.15, MAP_TYPE_TRAIN)
        robot = Robot(env.robot_start_pos_ref, ROBOT_SENSOR_RANGE_TRAIN)
        visited_cells_in_episode = {robot.get_position()}
        robot.sense(env)
        
        current_ep_reward = 0
        total_physical_steps_this_episode = 0
        total_planning_cycles_this_episode = 0
        current_path_segment = []
        
        # --- Visualization Setup for this Episode ---
        show_this_ep = VISUALIZE_MODULO > 0 and episode % VISUALIZE_MODULO == 0
        visualizer = None
        if show_this_ep:
            try: visualizer = Visualizer(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, CELL_SIZE_VIZ)
            except pygame.error as e: visualizer = None; print(f"Viz init error ep {episode}: {e}")

        # --- Main Episode Loop ---
        while total_physical_steps_this_episode < MAX_PHYSICAL_STEPS_PER_EPISODE and \
              total_planning_cycles_this_episode < MAX_PLANNING_CYCLES_PER_EPISODE:

            if visualizer and not visualizer.handle_events_simple(): break
            
            # --- Planning Phase ---
            if not current_path_segment:
                last_explored_count = np.sum(env.grid_known != UNKNOWN)
                
                kwargs_for_planner = {'total_physical_steps_taken': total_physical_steps_this_episode}
                target_pos, path = planner.plan_next_action(robot.get_position(), env.grid_known, **kwargs_for_planner)
                total_planning_cycles_this_episode += 1

                if target_pos is None or not path or len(path) <= 1:
                    print(f"  Ep {episode}: Planner failed or returned trivial path. Ending episode.")
                    if len(planner.memory.actions) > len(planner.memory.rewards):
                        planner.memory.rewards.append(-2.0) # Planner failure penalty
                        planner.memory.is_terminals.append(True)
                    break
                
                current_path_segment = path[1:]
                path_cost_to_execute = len(current_path_segment)

            # --- Execution Phase ---
            if current_path_segment:
                path_taken_this_action = [robot.get_position()] # Track path for this specific action
                next_step = current_path_segment.pop(0)
                moved = robot.attempt_move_one_step(next_step, env)
                total_physical_steps_this_episode += 1 # A physical step is attempted
                timestep_counter += 1

                robot.sense(env) # Sense after every move attempt
                if moved:
                        path_taken_this_action.append(robot.get_position())
                # --- Reward Calculation and Memory Storage ---
                # Reward is given only after an action (a full path segment) is completed or fails.
                # An "action" in this RL context is choosing a frontier.
                # The "outcome" is the result of trying to follow the path to it.
                if not moved or not current_path_segment: # If collision or path finished
                    new_explored_count = np.sum(env.grid_known != UNKNOWN)
                    newly_explored = new_explored_count - last_explored_count
                    
                    # The cost is how many steps we *actually* took before collision or completion.
                    # path_cost_to_execute is the *planned* cost.
                    # actual_cost is how many steps we just took.
                    # This requires tracking steps since last plan. Let's simplify for now:
                    # Let's assume the reward is based on the outcome of the entire planned path segment.
                    actual_cost_this_segment = path_cost_to_execute - len(current_path_segment) if moved else path_cost_to_execute
                    
                    reward = calculate_reward(newly_explored, path_taken_this_action, visited_cells_in_episode)
                    current_ep_reward += reward
                    for cell in path_taken_this_action:
                        visited_cells_in_episode.add(cell)
                    is_done = not moved # Treat collision as a terminal state for this action's consequence
                    planner.memory.rewards.append(reward)
                    planner.memory.is_terminals.append(is_done)

                    last_explored_count = new_explored_count # Update for next reward calculation
                    if not moved: current_path_segment = [] # Ensure path is cleared on collision
            
            # --- Visualization ---
            if visualizer:
                # ... (visualization logic from previous version) ...
                visualizer.clear_screen(); visualizer.draw_grid_map(env.grid_known)
                visualizer.draw_robot_path(robot.actual_pos_history)
                visualizer.update_display(); time.sleep(SIM_DELAY_VIZ)

            # --- PPO Update ---
            if USE_TIMESTEP_UPDATE and len(planner.memory.actions) >= PPO_UPDATE_TIMESTEP:
                print(f"  Episode {episode}, Timestep Counter {timestep_counter}: Updating PPO policy...")
                planner.update_ppo()
                # Memory is cleared inside update_ppo

        # --- End of Episode ---
        if len(planner.memory.actions) > len(planner.memory.rewards):
            planner.memory.rewards.append(0.0) # Neutral reward for episode timeout
            planner.memory.is_terminals.append(True)
        PPO_UPDATE = 10*TRAIN_EVERY_N_EPISODES
        if not USE_TIMESTEP_UPDATE and episode % PPO_UPDATE == 0:
            if len(planner.memory.actions) > 0:
                print(f"--- Updating PPO policy after Episode {episode} (data from last {len(planner.memory.rewards)} steps) ---")
                p_loss, v_loss = planner.update_ppo()
                # Memory is cleared inside update_ppo()
                print(f"  Update complete. Last Mini-batch -> Policy Loss: {p_loss:.4f}, Value Loss: {v_loss:.4f}")

        
        episode_rewards_history.append(current_ep_reward)
        episode_lengths_history.append(total_physical_steps_this_episode) # Log physical steps
        all_final_explored.append(np.sum(env.grid_known != UNKNOWN) / (env.width * env.height) * 100)
        
        if visualizer: visualizer.quit()

        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards_history[-10:])
            avg_length = np.mean(episode_lengths_history[-10:])
            avg_explored = np.mean(all_final_explored[-10:])
            print(f"Episode {episode} | Avg Physical Steps: {avg_length:.1f} | Avg Reward: {avg_reward:.2f} | Avg Explored: {avg_explored:.1f}%")
        
        if episode % 100 == 0:
            print(f"--- Saving model checkpoint at Episode {episode}... ---")
            planner.save_model(MODEL_SAVE_PATH)
    
    print("--- CHECKPOINT 9: Finished main training loop. ---")
    planner.save_model(MODEL_SAVE_PATH) 
    print(f"\n--- Training Finished ---")
    # ... (Plotting logic as before, using episode_lengths_history for the length plot) ...
    # if ep_rewards:
    #     try:
    #         fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    #         def smooth(y, box_pts):
    #             if not isinstance(y, (list, np.ndarray)) or len(y) < box_pts : return np.array(y) 
    #             box = np.ones(box_pts)/box_pts
    #             return np.convolve(y, box, mode='valid')
    #         smoothing_window = max(1, len(ep_rewards) // 20)
    #         if smoothing_window == 0: smoothing_window = 1

    #         axs[0].plot(ep_rewards, label="Episode Reward", color='gray', alpha=0.4)
    #         rewards_smoothed = smooth(ep_rewards, smoothing_window) 
    #         if len(rewards_smoothed) > 0:
    #             axs[0].plot(np.arange(len(rewards_smoothed)) + smoothing_window - 1, rewards_smoothed, 
    #                         label=f"Smoothed Reward (win={smoothing_window})", color='green')
    #         axs[0].set_ylabel("Total Reward"); axs[0].set_title("Reward per Episode"); axs[0].legend(); axs[0].grid(True)
            
    #         axs[1].plot(episode_lengths_history, label="Episode Length", color='gray', alpha=0.4)
    #         lengths_smoothed = smooth(episode_lengths_history, smoothing_window)
    #         if len(lengths_smoothed) > 0:
    #             axs[1].plot(np.arange(len(lengths_smoothed)) + smoothing_window - 1, lengths_smoothed, 
    #                         label=f"Smoothed Length (win={smoothing_window})", color='purple')
    #         axs[1].set_ylabel("Physical Steps"); axs[1].set_title("Episode Length (Physical Steps)"); axs[1].legend(); axs[1].grid(True)
            
    #         axs[2].plot(all_final_explored, label="Final Exploration %", color='gray', alpha=0.4)
    #         expl_smoothed = smooth(all_final_explored, smoothing_window)
    #         if len(expl_smoothed) > 0:
    #             axs[2].plot(np.arange(len(expl_smoothed)) + smoothing_window - 1, expl_smoothed, 
    #                         label=f"Smoothed Explored % (win={smoothing_window})", color='darkblue')
    #         axs[2].set_ylabel("Exploration %"); axs[2].set_title("Final Exploration Percentage per Episode"); axs[2].legend(); axs[2].grid(True); axs[2].set_ylim(0,105)

    #         axs[-1].set_xlabel("Episode")
    #         plt.tight_layout()
    #         plot_filename = os.path.join(RESULTS_DIR, "ppo_training_summary.png")
    #         plt.savefig(plot_filename); print(f"Training summary plot saved to {plot_filename}")
    #         plt.close(fig) 
    #     except Exception as e_plt: print(f"Error plotting training progress: {e_plt}")

    # if pygame.get_init(): pygame.quit()

if __name__ == "__main__":
    # ... (The __main__ block remains the same, it just calls train_deep_aco_simple()) ...
    print("--- CHECKPOINT 0: Script starting... ---")
    if VISUALIZE_MODULO > 0 and not pygame.get_init():
        print("Pygame display required for visualization.")
        try: pygame.init()
        except pygame.error as e: print(f"FATAL: Pygame could not be initialized: {e}."); exit(1)
    try:
        train_deep_aco_simple()
    except Exception as e:
        print(f"FATAL ERROR during top-level training script: {e}")
        traceback.print_exc()
    finally:
        print("--- Script Finished. ---")
        if pygame.get_init(): pygame.quit()