import numpy as np
import time
import os
import torch 
import pygame 
import matplotlib.pyplot as plt 

from environment import Environment, UNKNOWN, FREE, OBSTACLE
from robot import Robot 
from planners.deep_aco_planner import DeepACOPlanner 
from visualizer import Visualizer 

# --- Training Configuration ---
MAP_WIDTH_TRAIN = 50
MAP_HEIGHT_TRAIN = 50
OBSTACLE_PERCENTAGE_TRAIN = 0.15
MAP_TYPE_TRAIN = "random"
ROBOT_SENSOR_RANGE_TRAIN = 5

MAX_EPISODE_STEPS = 234    
NUM_EPISODES_TRAIN = 1000 
PLANNING_CYCLES_PER_EPISODE_LIMIT = MAX_EPISODE_STEPS 
ENV_SEED_TRAIN = 42 # Seed for reproducibility in environment generation

# --- DeepACOPlanner Parameters ---
# These keys must match the __init__ args of DeepACOPlanner
# Ensure default values here match those in DeepACOPlanner's __init__ if not overridden by args
DEEP_ACO_PARAMS_TRAIN = {
    'n_ants': 8, 
    'n_iterations': 5, 
    'alpha': 1.0, 
    'beta_nn_heuristic': 2.0, 
    'evaporation_rate': 0.1, 
    'q0': 0.7,     
    'pheromone_min': 0.01, 
    'pheromone_max': 5.0,
    
    'local_patch_radius': 7, 
    'num_global_features_for_heuristic_nn': 5, 
    'nn_input_channels': 1, 
    'nn_cnn_fc_out_dim': 8,
    'nn_mlp_hidden1_heuristic': 64, 
    'nn_mlp_hidden2_heuristic': 32,
    
    'num_global_features_for_value_nn': 4, 
    'nn_mlp_hidden1_value': 64, 
    'nn_mlp_hidden2_value': 32,

    'learning_rate_heuristic': 3e-5, 
    'learning_rate_value': 1e-4,    
    'gamma_rl': 0.99, 
    
    'robot_sensor_range': ROBOT_SENSOR_RANGE_TRAIN 
}

TRAIN_EVERY_N_EPISODES = 1 
MODEL_SAVE_PATH_TRAIN = "heuristic_net_pretrained_manual_eta.pth" 
RESULTS_DIR_TRAIN = "training_results_deep_aco_reinforce_v2" # Changed output dir
os.makedirs(RESULTS_DIR_TRAIN, exist_ok=True)

VISUALIZE_TRAINING_EPISODE_MODULO = 0 # 0 to disable, e.g. 50 to viz every 50th episode
CELL_SIZE_TRAIN_VIZ = 15
SIM_DELAY_TRAIN_VIZ = 0.001 

MASTER_SEED_DEFAULT_TRAIN = 420 # Moved from main_simulation.py for this script
env_seed_arg = ENV_SEED_TRAIN
if env_seed_arg is not None: np.random.seed(env_seed_arg)
def calculate_reward(newly_explored_cells_count, path_length_taken,
                     exploration_bonus_factor=1.0, 
                     path_cost_penalty_factor=0.01, # Reduced penalty
                     step_penalty = -0.005): # Smaller step penalty
    exploration_reward = newly_explored_cells_count * exploration_bonus_factor
    path_penalty = path_length_taken * path_cost_penalty_factor if path_length_taken > 0 else 0
    reward = exploration_reward - path_penalty + step_penalty 
    return reward


def train_deep_aco_planner():
    print("--- Starting DeepACOPlanner Training (REINFORCE with Baseline) ---")
    
    if not pygame.get_init() and VISUALIZE_TRAINING_EPISODE_MODULO > 0:
        pygame.init() 

    # Create a dummy environment instance just for initializing the planner
    # The actual environment for each episode will be created inside the loop
    temp_env_for_init = Environment(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, 0.0, "random")
    planner = DeepACOPlanner(
        environment=temp_env_for_init, 
        **DEEP_ACO_PARAMS_TRAIN 
    )
    # Try to load a pre-trained model if it exists
    if os.path.exists(MODEL_SAVE_PATH_TRAIN.replace(".pth", "_heuristic.pth")): # Check for one part
        print(f"Loading existing model from base path: {MODEL_SAVE_PATH_TRAIN}")
        planner.load_model(MODEL_SAVE_PATH_TRAIN)


    episode_final_exploration_percentages = []
    episode_total_steps_list = [] 
    episode_total_rewards_list = [] 
    policy_losses_history = [] 
    value_losses_history = []  

    for episode_num in range(NUM_EPISODES_TRAIN):
        print(f"\n--- Episode {episode_num + 1}/{NUM_EPISODES_TRAIN} ---")
        
        current_episode_seed = MASTER_SEED_DEFAULT_TRAIN + episode_num 
        env = Environment(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, 
                          OBSTACLE_PERCENTAGE_TRAIN, MAP_TYPE_TRAIN, 
                          robot_start_pos_ref=None)# Let env pick default start) 
        
        start_r_ep, start_c_ep = env.robot_start_pos_ref 
        if env.grid_true[start_r_ep, start_c_ep] == OBSTACLE:
            env.grid_true[start_r_ep, start_c_ep] = FREE

        robot = Robot(env.robot_start_pos_ref, ROBOT_SENSOR_RANGE_TRAIN)
        
        # Reset planner's episode-specific states
        planner.pheromones = {} 
        planner.episode_log_probs = [] 
        planner.episode_rewards = []   
        planner.episode_state_values_for_baseline = [] 
        
        robot.sense(env) # Initial sense
        last_known_explored_count_for_reward = np.sum(env.grid_known != UNKNOWN) 
        
        visualizer_episode = None 
        show_this_episode_viz = VISUALIZE_TRAINING_EPISODE_MODULO > 0 and \
                                (episode_num + 1) % VISUALIZE_TRAINING_EPISODE_MODULO == 0

        if show_this_episode_viz:
            try:
                if not pygame.get_init(): pygame.init()
                visualizer_episode = Visualizer(MAP_WIDTH_TRAIN, MAP_HEIGHT_TRAIN, CELL_SIZE_TRAIN_VIZ)
                pygame.display.set_caption(f"DeepACO Training - Ep {episode_num+1}")
            except pygame.error as e: visualizer_episode = None

        current_episode_path_segment = []
        total_physical_steps_this_episode = 0
        total_planning_cycles_this_episode = 0
        sim_active_episode = True
        episode_cumulative_reward_val = 0.0

        # --- Main Episode Loop ---
        while total_physical_steps_this_episode < MAX_EPISODE_STEPS and \
              total_planning_cycles_this_episode < PLANNING_CYCLES_PER_EPISODE_LIMIT and \
              sim_active_episode:

            if visualizer_episode:
                if not visualizer_episode.handle_events_simple(): 
                    sim_active_episode = False; break # User quit visualization
            
            # --- Planning ---
            if not current_episode_path_segment: 
                planner_kwargs_train = {'total_physical_steps_taken': total_physical_steps_this_episode}
                
                # plan_next_action will store log_prob and V(S_t) for the chosen action
                target_pos, path_segment_from_planner = planner.plan_next_action(
                    robot.get_position(), env.grid_known, **planner_kwargs_train
                )
                total_planning_cycles_this_episode +=1

                if target_pos is None or not path_segment_from_planner:
                    # Planner failed, store a penalty for the last (S,A) that led to this
                    if len(planner.episode_log_probs) > len(planner.episode_rewards): # Check if an action was taken
                         planner.store_reward_for_last_action(calculate_reward(0,0,0,step_penalty=-1.0), True) # done=True
                         episode_cumulative_reward_val += planner.episode_rewards[-1]
                    sim_active_episode = False; break # End episode if planner fails
                
                current_episode_path_segment = path_segment_from_planner[1:] if len(path_segment_from_planner) > 1 else []
                if not current_episode_path_segment: # Path was trivial (e.g., to current pos)
                    # Store a small penalty for the (S,A) that led to this
                    if len(planner.episode_log_probs) > len(planner.episode_rewards):
                        planner.store_reward_for_last_action(calculate_reward(0,0,0,step_penalty=-0.1), False)
                        episode_cumulative_reward_val += planner.episode_rewards[-1]
                    sim_active_episode = False; break 

            # --- Execution ---
            path_cost_this_action_segment = 0
            if current_episode_path_segment and sim_active_episode:
                next_step = current_episode_path_segment.pop(0)
                path_cost_this_action_segment += 1
                moved = robot.attempt_move_one_step(next_step, env)
                
                if moved:
                    robot.sense(env)
                else: # Collision
                    current_episode_path_segment = [] # Path invalid
                    robot.sense(env) 
                
                # Store reward if this was the last step of the segment or path broke
                if not current_episode_path_segment or not moved:
                    newly_explored_count_after_move = np.sum(env.grid_known != UNKNOWN)
                    reward_this_segment = calculate_reward(
                        newly_explored_count_after_move - last_known_explored_count_for_reward, 
                        0, # Assuming simple count for now (no distinction between FREE/OBSTACLE reward)
                        path_cost_this_action_segment 
                    )
                    # This reward is for the *previous* choice of frontier that generated this segment
                    planner.store_reward_for_last_action(reward_this_segment, False) 
                    episode_cumulative_reward_val += reward_this_segment
                    last_known_explored_count_for_reward = newly_explored_count_after_move
                    path_cost_this_action_segment = 0 # Reset for next segment (if any)
                
                total_physical_steps_this_episode +=1
                
                current_explored_ratio = np.sum(env.grid_known != UNKNOWN) / (env.width * env.height)
                if current_explored_ratio >= 0.995: # End episode if highly explored
                    sim_active_episode = False 

            # --- Visualization (if active) ---
            if visualizer_episode:
                visualizer_episode.clear_screen(); visualizer_episode.draw_grid_map(env.grid_known)
                visualizer_episode.draw_robot_path(robot.actual_pos_history)
                visualizer_episode.draw_robots([robot]) # Pass as list
                current_explored_ratio_viz = np.sum(env.grid_known != UNKNOWN) / (env.width * env.height)
                visualizer_episode.draw_text(f"Ep: {episode_num+1}, Step: {total_physical_steps_this_episode}", (10,10))
                visualizer_episode.draw_text(f"Explored: {current_explored_ratio_viz*100:.1f}% Rew: {episode_cumulative_reward_val:.2f}", (10,30))
                visualizer_episode.update_display()
                if SIM_DELAY_TRAIN_VIZ > 0: time.sleep(SIM_DELAY_TRAIN_VIZ)
        
        # --- End of Episode Logic ---
        # If episode ended for reasons other than planner failure, the last action still needs a reward if pending
        if sim_active_episode: # Means loop ended by max_steps, max_cycles, or exploration %
            if len(planner.episode_log_probs) > len(planner.episode_rewards):
                final_reward_ep = 0.0 
                current_explored_ratio_final = np.sum(env.grid_known != UNKNOWN) / (env.width * env.height)
                if current_explored_ratio_final >= 0.995: final_reward_ep = 5.0 # Bonus for high exploration
                elif total_physical_steps_this_episode >= MAX_EPISODE_STEPS: final_reward_ep = -1.0 # Penalty for timeout
                planner.store_reward_for_last_action(final_reward_ep, True) # Mark as done
                episode_cumulative_reward_val += final_reward_ep

        if visualizer_episode: visualizer_episode.quit() 

        final_explored_perc_episode = np.sum(env.grid_known != UNKNOWN) / (env.width * env.height) * 100
        episode_final_exploration_percentages.append(final_explored_perc_episode)
        episode_total_steps_list.append(total_physical_steps_this_episode)
        episode_total_rewards_list.append(episode_cumulative_reward_val)

        print(f"  Ep {episode_num + 1} ended. Steps: {total_physical_steps_this_episode}, Explored: {final_explored_perc_episode:.2f}%, Total Reward: {episode_cumulative_reward_val:.2f}")

        # --- Train Network Periodically ---
        if (episode_num + 1) % TRAIN_EVERY_N_EPISODES == 0:
            if planner.episode_log_probs: 
                p_loss, v_loss = planner.finish_episode_and_train() 
                if p_loss is not None: policy_losses_history.append(p_loss)
                if v_loss is not None: value_losses_history.append(v_loss)  
                avg_p_loss = np.mean(policy_losses_history[-10:]) if policy_losses_history else float('nan')
                avg_v_loss = np.mean(value_losses_history[-10:]) if value_losses_history else float('nan')
                print(f"  NN training. Policy Loss: {avg_p_loss:.4f}, Value Loss: {avg_v_loss:.4f}")
            else:
                print(f"  Skipping training for episode {episode_num+1}, no data collected (e.g. planner failed on first step).")
            
            # Save model periodically
            if (episode_num + 1) % (TRAIN_EVERY_N_EPISODES * 20) == 0 : # Adjusted save frequency
                planner.save_model(MODEL_SAVE_PATH_TRAIN)

    # --- End of Training ---
    planner.save_model(MODEL_SAVE_PATH_TRAIN) 
    print(f"\n--- Training Finished ---")
    print(f"Final model saved to {MODEL_SAVE_PATH_TRAIN}")

    # Plotting training progress
    if episode_final_exploration_percentages:
        try:
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            axs[0].plot(episode_final_exploration_percentages, label="Exploration %")
            axs[0].set_ylabel("Final Exploration %"); axs[0].set_title("Exploration Percentage per Episode"); axs[0].grid(True)
            axs[1].plot(episode_total_rewards_list, label="Total Reward", color='green')
            axs[1].set_ylabel("Cumulative Reward"); axs[1].set_title("Total Reward per Episode"); axs[1].grid(True)
            if policy_losses_history and value_losses_history:
                ax_loss = axs[2]
                def smooth(y, box_pts):
                    if len(y) < box_pts: return np.array(y) 
                    box = np.ones(box_pts)/box_pts
                    return np.convolve(y, box, mode='valid')
                smoothing_window = min(10, max(1,len(policy_losses_history)//2) ) 
                if len(policy_losses_history) >= 1:
                    ax_loss.plot(smooth(policy_losses_history, smoothing_window), label=f"Policy Loss (smooth {smoothing_window})", color='red')
                if len(value_losses_history) >= 1:
                    ax_loss.plot(smooth(value_losses_history, smoothing_window), label=f"Value Loss (smooth {smoothing_window})", color='blue')
                ax_loss.set_ylabel("Loss"); ax_loss.set_title("Training Losses (Smoothed)"); ax_loss.legend(); ax_loss.grid(True)
            axs[-1].set_xlabel("Episode")
            plt.tight_layout()
            plot_filename = os.path.join(RESULTS_DIR_TRAIN, "deep_aco_training_summary.png")
            plt.savefig(plot_filename); print(f"Training summary plot saved to {plot_filename}")
            plt.close(fig) 
        except Exception as e_plt: print(f"Error plotting training progress: {e_plt}")

if __name__ == "__main__":
    if not pygame.get_init() and VISUALIZE_TRAINING_EPISODE_MODULO > 0 : 
        pygame.init()
    try:
        train_deep_aco_planner()
    except Exception as e_train_top:
        print(f"Error during top-level training script: {e_train_top}")
        import traceback
        traceback.print_exc()
    finally:
        if pygame.get_init(): pygame.quit()