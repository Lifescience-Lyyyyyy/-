import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import random
import time
import pygame # For visualization if needed, but not strictly required for this script
# Assume these modules are accessible from the project root or PYTHONPATH
from environment import Environment, UNKNOWN, FREE, OBSTACLE
from robot import Robot # Needed to simulate robot movement and sensing for data generation
from planners.base_planner import BasePlanner # For find_frontiers, _is_reachable_and_get_path
from planners.geometry_utils import check_line_of_sight
# Import the HeuristicNetworkCNN from deep_aco_planner
from planners.deep_aco_planner import HeuristicNetworkCNN, DeepACOPlanner # For utility methods

# --- Configuration for Supervised Pre-training ---
# Map Generation for Data Collection
NUM_MAPS_TO_GENERATE = 30 # Number of different random maps for data generation
MAP_WIDTH_PRETRAIN = 50
MAP_HEIGHT_PRETRAIN = 50
OBSTACLE_PERCENTAGE_PRETRAIN_RANGE = (0.15, 0.35) # Range of obstacle %
MAP_TYPE_PRETRAIN = "random"

# Data Collection Parameters (simulating robot exploration to find frontiers)
STEPS_PER_MAP_FOR_DATA_COLLECTION = 200 # How many steps a simple explorer runs on each map
ROBOT_SENSOR_RANGE_PRETRAIN = 5

# Heuristic Network Parameters (must match definition in DeepACOPlanner)
LOCAL_PATCH_RADIUS_PRETRAIN = 7 # For 15x15 patch
PATCH_SIDE_LENGTH_PRETRAIN = 2 * LOCAL_PATCH_RADIUS_PRETRAIN + 1
NN_INPUT_CHANNELS_PRETRAIN = 1 # Assuming only known_map patch

# Define the number of global features your MANUAL heuristic (and thus NN input) will use.
# This needs to be consistent with how features are extracted for NN input.
# Let's assume the manual heuristic target eta_F uses:
# 1. Robot normalized R
# 2. Robot normalized C
# 3. Path cost to F (normalized)
# 4. Global Explored Ratio
# 5. Total number of frontiers (normalized)
# 6. IG from F (normalized) - If IG itself is a feature for NN
NUM_GLOBAL_FEATURES_PRETRAIN = 5 # Example: Adjust this!

NN_CNN_FC_OUT_DIM_PRETRAIN = 8
NN_MLP_HIDDEN1_PRETRAIN = 64
NN_MLP_HIDDEN2_PRETRAIN = 32

# Training Parameters
BATCH_SIZE_SUPERVISED = 64
LEARNING_RATE_SUPERVISED = 1e-3
NUM_EPOCHS_SUPERVISED = 15 # Number of epochs to train on the collected dataset

# Parameters for the MANUAL heuristic function (target for NN to learn)
MANUAL_HEURISTIC_IG_WEIGHT = 3.0 # w_IG in (w_IG * IG) / Cost
MANUAL_HEURISTIC_COST_EPSILON = 0.1 # Epsilon for cost denominator

PRETRAINED_MODEL_SAVE_PATH = "heuristic_net_pretrained_manual_eta.pth"
DATASET_SAVE_PATH = "heuristic_pretrain_dataset.pth" # To save/load generated data

# --- Helper class to use BasePlanner's methods for data generation ---
class DataGenerationHelper(BasePlanner):
    def __init__(self, environment):
        super().__init__(environment)
    
    def plan_next_action(self, robot_pos, known_map, **kwargs):
        """
        Minimal implementation for plan_next_action as DataGenerationHelper
        is an abstract class due to BasePlanner.
        This method might not be directly used for data generation logic here,
        but needs to be defined.
        """
        # print("DataGenerationHelper.plan_next_action called (dummy implementation)")
        # For data generation, we usually select frontiers or moves differently.
        # If a planner logic was needed here, it would be implemented.
        # For now, as we only use find_frontiers and get_path, this can be simple.
        return None, None # Or raise NotImplementedError if it should never be called
    
    def get_frontiers(self, known_map):
        return self.find_frontiers(known_map)
        
    def get_path(self, start_pos, end_pos, known_map):
        return self._is_reachable_and_get_path(start_pos, end_pos, known_map)

    def calculate_manual_ig_los(self, frontier_pos, known_map, sensor_range):
        ig = 0
        r_f, c_f = frontier_pos
        for dr in range(-sensor_range, sensor_range + 1):
            for dc in range(-sensor_range, sensor_range + 1):
                nr, nc = r_f + dr, c_f + dc
                if self.environment.is_within_bounds(nr, nc) and \
                   known_map[nr, nc] == UNKNOWN and \
                   check_line_of_sight(r_f, c_f, nr, nc, known_map):
                    ig += 1
        return ig

def generate_supervised_data(num_maps, steps_per_map):
    """Generates (input_features_for_NN, target_manual_heuristic_value) pairs."""
    print(f"--- Generating supervised learning data from {num_maps} maps ---")
    all_map_patches = []
    all_global_features = []
    all_target_heuristics = []

    # Use a temporary DeepACOPlanner instance just for its feature extraction methods
    # Its internal NN won't be trained here, only its helper methods are used.
    dummy_env = Environment(MAP_WIDTH_PRETRAIN, MAP_HEIGHT_PRETRAIN, 0.2, "random")
    feature_extractor = DeepACOPlanner(dummy_env, 
                                       local_patch_radius=LOCAL_PATCH_RADIUS_PRETRAIN,
                                       num_global_features_for_heuristic_nn=NUM_GLOBAL_FEATURES_PRETRAIN,
                                       nn_input_channels=NN_INPUT_CHANNELS_PRETRAIN,
                                       robot_sensor_range=ROBOT_SENSOR_RANGE_PRETRAIN)
    data_gen_helper = DataGenerationHelper(dummy_env) # For find_frontiers, get_path

    for map_idx in range(num_maps):
        MASTER_SEED_DEFAULT = 42 # Default seed for reproducibility
        random.seed(MASTER_SEED_DEFAULT + map_idx * 100) # Seed for reproducibility
        map_seed = MASTER_SEED_DEFAULT + map_idx * 100 # Vary seed for diverse maps
        obs_perc = random.uniform(OBSTACLE_PERCENTAGE_PRETRAIN_RANGE[0], OBSTACLE_PERCENTAGE_PRETRAIN_RANGE[1])
        
        env = Environment(MAP_WIDTH_PRETRAIN, MAP_HEIGHT_PRETRAIN, 
                          obs_perc, MAP_TYPE_PRETRAIN)
        robot_start_pos = env.robot_start_pos_ref
        if env.grid_true[robot_start_pos[0], robot_start_pos[1]] == OBSTACLE:
            env.grid_true[robot_start_pos[0], robot_start_pos[1]] = FREE # Ensure start is free

        robot = Robot(robot_start_pos, ROBOT_SENSOR_RANGE_PRETRAIN)
        
        print(f"  Map {map_idx+1}/{num_maps}, Seed: {map_seed}, Obs%: {obs_perc:.2f}")

        # Simple random walk or basic FBE to explore and find frontiers
        current_known_map = np.copy(env.grid_known) # Robot starts with empty known map
        robot.sense(env) # Initial sense
        current_known_map = np.copy(env.grid_known)

        for step in range(steps_per_map):
            frontiers = data_gen_helper.get_frontiers(current_known_map)
            if not frontiers: break # No more frontiers on this map for this run

            # For data generation, robot can just pick a random reachable frontier
            random.shuffle(frontiers)
            chosen_frontier_for_robot_move = None
            path_to_chosen_frontier_for_robot_move = None

            # Evaluate ALL frontiers to generate data points for NN training
            for f_pos_cand in frontiers:
                path_to_f = data_gen_helper.get_path(robot.get_position(), f_pos_cand, current_known_map)
                if path_to_f:
                    path_cost_to_f = len(path_to_f) - 1
                    if path_cost_to_f < 0: path_cost_to_f = 0
                    
                    # 1. Calculate TARGET manual heuristic (Y_F)
                    ig_f = data_gen_helper.calculate_manual_ig_los(f_pos_cand, current_known_map, ROBOT_SENSOR_RANGE_PRETRAIN)
                    manual_heuristic_val = (MANUAL_HEURISTIC_IG_WEIGHT * ig_f + 0.1) / (path_cost_to_f + MANUAL_HEURISTIC_COST_EPSILON)
                    manual_heuristic_val = max(0, manual_heuristic_val) # Ensure non-negative target
        
                    # 2. Extract INPUT features for NN (X_F)
                    # Use the feature extraction methods from DeepACOPlanner
                    map_patch_np = feature_extractor._get_map_patch_2d(
                        f_pos_cand[0], f_pos_cand[1], current_known_map, LOCAL_PATCH_RADIUS_PRETRAIN)
                    
                    global_features_np = feature_extractor._get_global_features_for_heuristic_input(
                        robot.get_position(), current_known_map, path_cost_to_f, f_pos_cand,
                        len(frontiers), step
                    )
                    # Note: _get_global_features_for_heuristic_input might need adjustment
                    # if it expects a pre-calculated centroid and you're not using it for manual heuristic.
                    # For this supervised learning, the `direction_consistency_score` feature might be
                    # hard to define a target for if the manual heuristic doesn't use it.
                    # Consider creating a version of _get_global_features... specifically for supervised pretraining
                    # that matches the features the manual heuristic implicitly considers.
                    # For now, we assume it returns the NUM_GLOBAL_FEATURES_PRETRAIN features.

                    all_map_patches.append(map_patch_np)
                    all_global_features.append(global_features_np)
                    all_target_heuristics.append(manual_heuristic_val)

                    # For robot movement to generate more diverse states (simple FBE like)
                    if chosen_frontier_for_robot_move is None: # Pick first reachable for robot to move
                        chosen_frontier_for_robot_move = f_pos_cand
                        path_to_chosen_frontier_for_robot_move = path_to_f

            # Robot takes a step to a (randomly chosen reachable) frontier to change state
            if chosen_frontier_for_robot_move and path_to_chosen_frontier_for_robot_move:
                if len(path_to_chosen_frontier_for_robot_move) > 1:
                    robot.attempt_move_one_step(path_to_chosen_frontier_for_robot_move[1], env)
                robot.sense(env)
                current_known_map = np.copy(env.grid_known)
            else: # Robot stuck or no frontiers
                break
    
    print(f"--- Data generation complete. Total samples: {len(all_target_heuristics)} ---")
    if not all_target_heuristics:
        print("Error: No data generated. Check map generation or exploration logic.")
        return None, None, None
        
    return np.array(all_map_patches), np.array(all_global_features), np.array(all_target_heuristics, dtype=np.float32)


def train_heuristic_net_supervised(model, train_loader, optimizer, num_epochs, device):
    print("--- Starting Supervised Training for Heuristic Network ---")
    model.to(device)
    model.train() # Set model to training mode
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for map_patches_batch, global_features_batch, target_heuristics_batch in train_loader:
            map_patches_batch = map_patches_batch.to(device)
            global_features_batch = global_features_batch.to(device)
            target_heuristics_batch = target_heuristics_batch.to(device).unsqueeze(1) # Ensure target is [B,1]

            optimizer.zero_grad()
            predicted_heuristics = model(map_patches_batch, global_features_batch)
            loss = F.mse_loss(predicted_heuristics, target_heuristics_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

    print("--- Supervised Training Finished ---")
    model.eval() # Set model to evaluation mode after training


if __name__ == "__main__":
    if not pygame.get_init(): # Pygame needed for Visualizer if used, but not for this script directly
        pass # Pygame init not strictly required for data gen and training if no viz

    # 1. Generate or Load Data
    if os.path.exists(DATASET_SAVE_PATH):
        print(f"Loading pre-generated dataset from {DATASET_SAVE_PATH}...")
        data = torch.load(DATASET_SAVE_PATH)
        map_patches_all, global_features_all, target_heuristics_all = data['patches'], data['globals'], data['targets']
    else:
        map_patches_all, global_features_all, target_heuristics_all = generate_supervised_data(
            NUM_MAPS_TO_GENERATE, STEPS_PER_MAP_FOR_DATA_COLLECTION
        )
        if map_patches_all is not None:
            torch.save({
                'patches': map_patches_all, 
                'globals': global_features_all, 
                'targets': target_heuristics_all
            }, DATASET_SAVE_PATH)
            print(f"Generated dataset saved to {DATASET_SAVE_PATH}")
        else:
            print("Exiting due to data generation failure.")
            exit()

    # 2. Create DataLoader
    # Ensure map_patches_all is (N, C, H, W)
    if map_patches_all.ndim == 3: # If (N, H, W), add channel dim
        map_patches_all = np.expand_dims(map_patches_all, axis=1)
    elif map_patches_all.shape[1] != NN_INPUT_CHANNELS_PRETRAIN:
        # This might happen if _get_map_patch_2d was returning (H,W) and then stacked to (N,H,W)
        # And then converted to tensor. It should be (N, C, H, W) for PyTorch CNN.
        # The _get_map_patch_2d in DeepACOPlanner returns (C,H,W) numpy,
        # so np.array(list_of_patches) should give (N,C,H,W).
        print(f"Warning: map_patches_all shape {map_patches_all.shape} might be incorrect for CNN input channels {NN_INPUT_CHANNELS_PRETRAIN}")
        # Attempt to reshape if it's (N, H, W) and channels=1
        if map_patches_all.ndim == 3 and NN_INPUT_CHANNELS_PRETRAIN == 1:
             map_patches_all = map_patches_all[:, np.newaxis, :, :]
             print(f"Reshaped map_patches_all to: {map_patches_all.shape}")


    dataset = TensorDataset(
        torch.tensor(map_patches_all, dtype=torch.float32),
        torch.tensor(global_features_all, dtype=torch.float32),
        torch.tensor(target_heuristics_all, dtype=torch.float32)
    )
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE_SUPERVISED, shuffle=True)

    # 3. Initialize Network and Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    heuristic_nn_pretrain = HeuristicNetworkCNN(
        patch_side_length=PATCH_SIDE_LENGTH_PRETRAIN,
        num_input_channels=NN_INPUT_CHANNELS_PRETRAIN,
        num_global_features=NUM_GLOBAL_FEATURES_PRETRAIN,
        cnn_fc_out_dim=NN_CNN_FC_OUT_DIM_PRETRAIN,
        mlp_hidden1=NN_MLP_HIDDEN1_PRETRAIN,
        mlp_hidden2=NN_MLP_HIDDEN2_PRETRAIN
    ).to(device)
    
    optimizer_pretrain = optim.Adam(heuristic_nn_pretrain.parameters(), lr=LEARNING_RATE_SUPERVISED)

    # 4. Train the Network
    train_heuristic_net_supervised(heuristic_nn_pretrain, train_loader, optimizer_pretrain, NUM_EPOCHS_SUPERVISED, device)

    # 5. Save the Pre-trained Model
    torch.save(heuristic_nn_pretrain.state_dict(), PRETRAINED_MODEL_SAVE_PATH)
    print(f"Pre-trained heuristic model saved to {PRETRAINED_MODEL_SAVE_PATH}")

    # Optional: Test the pre-trained model on a few samples
    if len(dataset) > 0:
        heuristic_nn_pretrain.eval()
        with torch.no_grad():
            sample_map_patches, sample_global_features, sample_targets = next(iter(train_loader))
            sample_map_patches = sample_map_patches.to(device)
            sample_global_features = sample_global_features.to(device)
            predictions = heuristic_nn_pretrain(sample_map_patches, sample_global_features)
            for i in range(min(5, len(sample_targets))):
                print(f"  Sample {i}: Target={sample_targets[i].item():.4f}, Predicted={predictions[i].item():.4f}")