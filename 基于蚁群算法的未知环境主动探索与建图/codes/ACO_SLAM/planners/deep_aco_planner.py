import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import os 

from .base_planner import BasePlanner 
from environment import UNKNOWN, FREE, OBSTACLE 
try:
    from .geometry_utils import check_line_of_sight 
except ImportError:
    try:
        from geometry_utils import check_line_of_sight
    except ImportError:
        print("Warning (DeepACOPlanner): geometry_utils.py not found or check_line_of_sight missing.")
        def check_line_of_sight(r1,c1,r2,c2,m): return True 

class HeuristicNetworkCNN(nn.Module):
    def __init__(self, patch_side_length, num_input_channels, num_global_features, 
                 cnn_channels1=16, cnn_channels2=32, cnn_fc_out_dim=32, 
                 mlp_hidden1=64, mlp_hidden2=32):
        super(HeuristicNetworkCNN, self).__init__()
        self.patch_side = patch_side_length
        self.num_channels = num_input_channels

        self.conv1 = nn.Conv2d(self.num_channels, cnn_channels1, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(cnn_channels1, cnn_channels2, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        dummy_input = torch.zeros(1, self.num_channels, self.patch_side, self.patch_side)
        dummy_out = self.pool2(F.relu(self.conv2(self.pool1(F.relu(self.conv1(dummy_input))))))
        self.cnn_flat_size = dummy_out.numel()

        self.cnn_to_fc = nn.Linear(self.cnn_flat_size, cnn_fc_out_dim)

        mlp_input_dim = cnn_fc_out_dim + num_global_features
        self.fc1_mlp = nn.Linear(mlp_input_dim, mlp_hidden1)
        self.fc2_mlp = nn.Linear(mlp_hidden1, mlp_hidden2)
        self.fc3_mlp = nn.Linear(mlp_hidden2, 1) 

    def forward(self, local_map_patch_batch, global_features_batch): 
        x_patch = F.relu(self.conv1(local_map_patch_batch))
        x_patch = self.pool1(x_patch)
        x_patch = F.relu(self.conv2(x_patch))
        x_patch = self.pool2(x_patch)
        x_patch_flat = x_patch.view(x_patch.size(0), -1)
        cnn_processed_features = F.relu(self.cnn_to_fc(x_patch_flat))

        if global_features_batch.numel() == 0:
            combined_x = cnn_processed_features
        else:
            if global_features_batch.shape[0] == 1 and cnn_processed_features.shape[0] > 1:
                 global_features_batch = global_features_batch.repeat(cnn_processed_features.shape[0], 1)
            elif global_features_batch.shape[0] != cnn_processed_features.shape[0]:
                 raise ValueError(f"Batch size mismatch: CNN features {cnn_processed_features.shape[0]}, Global features {global_features_batch.shape[0]}")
            combined_x = torch.cat((cnn_processed_features, global_features_batch), dim=1)
        
        x = F.relu(self.fc1_mlp(combined_x))
        x = F.relu(self.fc2_mlp(x))
        heuristic_value = self.fc3_mlp(x)
        return F.softplus(heuristic_value) + 1e-6 

class ValueNetwork(nn.Module):
    def __init__(self, num_global_state_features, hidden_dim1=64, hidden_dim2=32):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(num_global_state_features, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

    def forward(self, global_state_features_vec):
        if global_state_features_vec.ndim == 1 and global_state_features_vec.shape[0] == self.fc1.in_features:
            global_state_features_vec = global_state_features_vec.unsqueeze(0)
        elif global_state_features_vec.ndim == 2 and global_state_features_vec.shape[1] != self.fc1.in_features:
            raise ValueError(f"ValueNetwork input feature mismatch. Expected {self.fc1.in_features}, got {global_state_features_vec.shape[1]}")

        x = F.relu(self.fc1(global_state_features_vec))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

class DeepACOPlanner(BasePlanner):

    def __init__(self, environment,
                 n_ants=8, n_iterations=5, 
                 alpha=1.0, beta_nn_heuristic=2.0, 
                 evaporation_rate=0.1, q0=0.7,     
                 pheromone_min=0.01, pheromone_max=5.0,
                 local_patch_radius=7, 
                 num_global_features_for_heuristic_nn=5, 
                 nn_input_channels=1, 
                 nn_cnn_fc_out_dim=8,
                 nn_mlp_hidden1_heuristic=64, nn_mlp_hidden2_heuristic=32,
                 num_global_features_for_value_nn=4, 
                 nn_mlp_hidden1_value=64, nn_mlp_hidden2_value=32,
                 learning_rate_heuristic=3e-5,
                 learning_rate_value=1e-4, 
                 gamma_rl=0.99, 
                 visualize_ants_callback=None,
                 robot_sensor_range = 5
                 ):
        super().__init__(environment)
        self.n_ants = n_ants; self.n_iterations = n_iterations
        self.alpha = alpha; self.beta_nn_heuristic = beta_nn_heuristic
        self.evaporation_rate = evaporation_rate; self.q0 = q0
        self.pheromones = {} 
        self.pheromone_min = pheromone_min; self.pheromone_max = pheromone_max
        
        self.patch_side_length = 2 * local_patch_radius + 1
        self.num_global_features_heuristic = num_global_features_for_heuristic_nn
        self.num_global_features_value = num_global_features_for_value_nn
        self.nn_input_channels = nn_input_channels
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DeepACOPlanner using device: {self.device}")

        self.heuristic_net = HeuristicNetworkCNN( 
            patch_side_length=self.patch_side_length, 
            num_input_channels=self.nn_input_channels,
            num_global_features=self.num_global_features_heuristic,
            cnn_fc_out_dim=nn_cnn_fc_out_dim,
            mlp_hidden1=nn_mlp_hidden1_heuristic, 
            mlp_hidden2=nn_mlp_hidden2_heuristic
        ).to(self.device)
        self.optimizer_heuristic = optim.Adam(self.heuristic_net.parameters(), lr=learning_rate_heuristic)

        self.value_net = ValueNetwork( 
            num_global_state_features=self.num_global_features_value,
            hidden_dim1=nn_mlp_hidden1_value,
            hidden_dim2=nn_mlp_hidden2_value
        ).to(self.device)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=learning_rate_value)
        
        self.gamma_rl = gamma_rl
        self.visualize_ants_callback = visualize_ants_callback
        self.robot_sensor_range = robot_sensor_range

        self.episode_log_probs = []
        self.episode_rewards = []
        self.episode_state_values_for_baseline = []


    def _get_map_patch_2d(self, center_r, center_c, map_data, patch_radius):
        """Extracts a 2D square patch, returns (C, H, W) for PyTorch CNN."""
        patch_side = 2 * patch_radius + 1

        output_patch = np.full((self.nn_input_channels, patch_side, patch_side), 
                               float(UNKNOWN), dtype=np.float32) # Use float for tensor

        if self.nn_input_channels != 1:
            raise ValueError(f"DeepACOPlanner._get_map_patch_2d currently only supports nn_input_channels=1, but got {self.nn_input_channels}")

        r_start_map = max(0, center_r - patch_radius)
        r_end_map = min(map_data.shape[0], center_r + patch_radius + 1)
        c_start_map = max(0, center_c - patch_radius)
        c_end_map = min(map_data.shape[1], center_c + patch_radius + 1)

        r_start_patch = patch_radius - (center_r - r_start_map)
        c_start_patch = patch_radius - (center_c - c_start_map)
        
        h_copy = r_end_map - r_start_map
        w_copy = c_end_map - c_start_map

        if h_copy > 0 and w_copy > 0:
            output_patch[0, 
                         r_start_patch : r_start_patch + h_copy, 
                         c_start_patch : c_start_patch + w_copy] = \
                map_data[r_start_map:r_end_map, c_start_map:c_end_map].astype(np.float32)
        assert output_patch.shape == (self.nn_input_channels, self.patch_side_length, self.patch_side_length), \
        f"Patch shape mismatch! Expected {(self.nn_input_channels, self.patch_side_length, self.patch_side_length)}, got {output_patch.shape}"
        return output_patch 

    def _get_global_features_for_heuristic_input(self, robot_pos, known_map, path_cost_to_frontier, 
                                                 frontier_pos, all_frontiers_count, total_steps_taken):
        norm_rob_r = robot_pos[0] / (self.environment.height -1 + 1e-5)
        norm_rob_c = robot_pos[1] / (self.environment.width -1 + 1e-5)
        max_possible_cost = self.environment.height + self.environment.width 
        norm_path_cost = path_cost_to_frontier / (max_possible_cost + 1e-5)
        explored_ratio = np.sum(known_map != UNKNOWN) / (self.environment.height * self.environment.width + 1e-5)
        num_frontiers_norm = all_frontiers_count / (self.environment.height * self.environment.width * 0.2 + 1e-5)
        num_frontiers_norm = np.clip(num_frontiers_norm, 0, 1)
        features = np.array([norm_rob_r, norm_rob_c, norm_path_cost, explored_ratio, num_frontiers_norm], dtype=np.float32)
        if len(features) != self.num_global_features_heuristic:
            raise ValueError(f"Heuristic global feature length mismatch! Expected {self.num_global_features_heuristic}, got {len(features)}")
        return features

    def _get_global_state_features_for_value_net(self, robot_pos, known_map, all_frontiers_count, total_steps_taken):
        norm_rob_r = robot_pos[0] / (self.environment.height -1 + 1e-5)
        norm_rob_c = robot_pos[1] / (self.environment.width -1 + 1e-5)
        explored_ratio = np.sum(known_map != UNKNOWN) / (self.environment.height * self.environment.width + 1e-5)
        num_frontiers_norm = all_frontiers_count / (self.environment.height * self.environment.width * 0.2 + 1e-5)
        num_frontiers_norm = np.clip(num_frontiers_norm, 0, 1)
        features = np.array([norm_rob_r, norm_rob_c, explored_ratio, num_frontiers_norm], dtype=np.float32)
        if len(features) != self.num_global_features_value:
             raise ValueError(f"Value net global feature length mismatch! Expected {self.num_global_features_value}, got {len(features)}")
        return features

    def _get_nn_heuristic_for_frontier_batch(self, frontier_batch_map_patches_tensor, 
                                           frontier_batch_global_features_tensor):
        heuristic_nn_outputs = self.heuristic_net(frontier_batch_map_patches_tensor, 
                                                  frontier_batch_global_features_tensor)
        return heuristic_nn_outputs.squeeze(-1) 

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        total_physical_steps_taken = kwargs.get('total_physical_steps_taken', 0)
        
        frontiers = self.find_frontiers(known_map)
        if not frontiers: 
            if self.episode_rewards and len(self.episode_log_probs) == len(self.episode_rewards) + 1:
                 self.store_reward_for_last_action(0.0, True)
            return None, None

        initial_pheromone_value = (self.pheromone_min + self.pheromone_max) / 2.0
        known_map_snapshot = np.copy(known_map)
        all_frontiers_count_current = len(frontiers)

        batch_map_patches_for_nn_list = [] 
        batch_global_features_for_nn_list = [] 
        temp_frontier_path_cost_info_list = [] 

        for f_pos_cand in frontiers:
            path_to_f = self._is_reachable_and_get_path(robot_pos, f_pos_cand, known_map_snapshot)
            if path_to_f:
                path_cost_to_f = len(path_to_f) - 1; 
                if path_cost_to_f < 0: path_cost_to_f = 0
                
                map_patch_np = self._get_map_patch_2d(f_pos_cand[0], f_pos_cand[1], known_map_snapshot, (self.patch_side_length - 1) // 2)
                global_features_np = self._get_global_features_for_heuristic_input(
                    robot_pos, known_map_snapshot, path_cost_to_f, f_pos_cand, 
                    all_frontiers_count_current, total_physical_steps_taken)
                
                batch_map_patches_for_nn_list.append(torch.tensor(map_patch_np,dtype=torch.float32)) 
                batch_global_features_for_nn_list.append(global_features_np) 
                temp_frontier_path_cost_info_list.append({'pos': f_pos_cand, 'path': path_to_f, 'cost': path_cost_to_f})
                
                if f_pos_cand not in self.pheromones: self.pheromones[f_pos_cand] = initial_pheromone_value
        
        if not batch_map_patches_for_nn_list:
            if self.episode_rewards and len(self.episode_log_probs) == len(self.episode_rewards) + 1:
                 self.store_reward_for_last_action(0.0, True)
            return self._final_fallback_plan(robot_pos, known_map)

        self.heuristic_net.train() 
        
        map_patches_batch_tensor = torch.tensor(np.stack(batch_map_patches_for_nn_list, axis=0), dtype=torch.float32).to(self.device)
        global_features_batch_tensor = torch.tensor(np.array(batch_global_features_for_nn_list), dtype=torch.float32).to(self.device)

        eta_nn_values_tensor = self._get_nn_heuristic_for_frontier_batch(
            map_patches_batch_tensor, global_features_batch_tensor
        ) 
        
        candidate_frontiers_with_scores = []
        for i, f_data in enumerate(temp_frontier_path_cost_info_list):
            candidate_frontiers_with_scores.append({
                'pos': f_data['pos'], 'path': f_data['path'], 'cost': f_data['cost'],
                'eta_nn_tensor': eta_nn_values_tensor[i] 
            })
        
        current_frontiers_pos_set = {info['pos'] for info in candidate_frontiers_with_scores}
        for _ in range(self.n_iterations):
            ant_choices_this_iter = []
            for _i_ant in range(self.n_ants):
                scores_for_roulette = []; chosen_f_dict_for_ant = None
                for f_dict in candidate_frontiers_with_scores:
                    f_pos, eta_nn_tensor_val = f_dict['pos'], f_dict['eta_nn_tensor']
                    tau = self.pheromones.get(f_pos, initial_pheromone_value)
                    score = (tau ** self.alpha) * (eta_nn_tensor_val.item() ** self.beta_nn_heuristic) 
                    scores_for_roulette.append((f_dict, score))
                if not scores_for_roulette: continue
                if random.random() < self.q0: 
                    valid_cands = [s for s in scores_for_roulette if s[1] > 0] 
                    if valid_cands: chosen_f_dict_for_ant = max(valid_cands, key=lambda x: x[1])[0]
                    elif scores_for_roulette : chosen_f_dict_for_ant = max(scores_for_roulette, key=lambda x: x[1])[0]
                else:
                    valid_cands = [s for s in scores_for_roulette if s[1] > 0]
                    total_score_val = sum(s[1] for s in valid_cands) 
                    if total_score_val <= 1e-9:
                        if scores_for_roulette: chosen_f_dict_for_ant = random.choice(scores_for_roulette)[0]
                    else:
                        r_val = random.random() * total_score_val; cum_prob = 0.0
                        for f_d, s_val_roulette in valid_cands: 
                            cum_prob += s_val_roulette
                            if r_val <= cum_prob: chosen_f_dict_for_ant = f_d; break
                        if chosen_f_dict_for_ant is None and valid_cands: chosen_f_dict_for_ant = valid_cands[-1][0]
                if chosen_f_dict_for_ant is None and scores_for_roulette: chosen_f_dict_for_ant = random.choice(scores_for_roulette)[0]
                if chosen_f_dict_for_ant:
                    ant_choices_this_iter.append({'frontier_pos': chosen_f_dict_for_ant['pos'], 
                                                  'deposit_val': chosen_f_dict_for_ant['eta_nn_tensor'].item() })
            active_ph_keys = list(self.pheromones.keys())
            for f_key in active_ph_keys:
                if f_key in current_frontiers_pos_set:
                    self.pheromones[f_key] *= (1.0 - self.evaporation_rate)
                    self.pheromones[f_key] = max(self.pheromone_min, self.pheromones[f_key])
                else:
                    if f_key in self.pheromones: del self.pheromones[f_key]
            for choice in ant_choices_this_iter:
                f_pos_c, deposit = choice['frontier_pos'], choice['deposit_val']
                if f_pos_c in self.pheromones and deposit > 0: 
                    self.pheromones[f_pos_c] += deposit
                    self.pheromones[f_pos_c] = min(self.pheromone_max, self.pheromones[f_pos_c])

        if not candidate_frontiers_with_scores: 
             if self.episode_rewards and len(self.episode_log_probs) == len(self.episode_rewards) + 1:
                 self.store_reward_for_last_action(0.0, True)
             return self._final_fallback_plan(robot_pos, known_map)

        policy_logits_for_selection = torch.stack([f_dict['eta_nn_tensor'] for f_dict in candidate_frontiers_with_scores])
         
        action_probs_from_nn_policy = F.softmax(policy_logits_for_selection / 1.0, dim=0) # Temperature = 1.0
        
        chosen_idx_by_robot = 0
        try:
            if torch.isnan(action_probs_from_nn_policy).any() or torch.isinf(action_probs_from_nn_policy).any() or action_probs_from_nn_policy.sum().item() < 1e-6 :
                eta_nn_items = [f['eta_nn_tensor'].item() for f in candidate_frontiers_with_scores]
                if any(s > 0 for s in eta_nn_items): chosen_idx_by_robot = np.argmax(eta_nn_items)
                else: chosen_idx_by_robot = random.randrange(len(candidate_frontiers_with_scores)) if candidate_frontiers_with_scores else 0
            else:
                chosen_idx_by_robot = torch.multinomial(action_probs_from_nn_policy, 1).item()
        except RuntimeError:
            if candidate_frontiers_with_scores:
                eta_nn_items = [f['eta_nn_tensor'].item() for f in candidate_frontiers_with_scores]
                if any(s > 0 for s in eta_nn_items): chosen_idx_by_robot = np.argmax(eta_nn_items)
                else: chosen_idx_by_robot = random.randrange(len(candidate_frontiers_with_scores))
            else: 
                if self.episode_rewards and len(self.episode_log_probs) == len(self.episode_rewards) + 1: self.store_reward_for_last_action(0.0, True)
                return self._final_fallback_plan(robot_pos, known_map)


        final_chosen_frontier_full_info = candidate_frontiers_with_scores[chosen_idx_by_robot]
        
        log_prob_chosen_action = torch.log(action_probs_from_nn_policy[chosen_idx_by_robot] + 1e-9) 
        
        current_global_state_for_value_net_np = self._get_global_state_features_for_value_net(
            robot_pos, known_map, all_frontiers_count_current, total_physical_steps_taken)
        current_global_state_for_value_net_tensor = torch.tensor(current_global_state_for_value_net_np, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.value_net.train()
        state_value_tensor_for_baseline = self.value_net(current_global_state_for_value_net_tensor).squeeze() 
       
        
        self.episode_log_probs.append(log_prob_chosen_action)
        self.episode_state_values_for_baseline.append(state_value_tensor_for_baseline) # Store the TENSOR
            
        return final_chosen_frontier_full_info['pos'], final_chosen_frontier_full_info['path']
    
    def store_reward_for_last_action(self, reward, done):
        if self.episode_log_probs : 
             self.episode_rewards.append(reward)
            

    def finish_episode_and_train(self, final_reward_for_terminal_state=0.0):
        if not self.episode_log_probs or not self.episode_rewards or \
           len(self.episode_log_probs) != len(self.episode_rewards) or \
           len(self.episode_state_values_for_baseline) != len(self.episode_rewards) :
            self.episode_log_probs = []; self.episode_rewards = []; self.episode_state_values_for_baseline = []
            return 0.0, 0.0

        T = len(self.episode_rewards)
        if T == 0 : 
            self.episode_log_probs = []; self.episode_rewards = []; self.episode_state_values_for_baseline = []
            return 0.0, 0.0

        returns_g = np.zeros(T, dtype=np.float32)
        current_return_g = 0.0 
        
        R = torch.tensor(self.episode_rewards, dtype=torch.float32).to(self.device)
        log_probs = torch.stack(self.episode_log_probs).squeeze() # Ensure it's 1D
        state_values = torch.stack(self.episode_state_values_for_baseline).squeeze() # Ensure it's 1D

        returns_g_calc = torch.zeros_like(R)
        discounted_reward_sum = 0.0
        for t in reversed(range(len(R))):
            discounted_reward_sum = R[t] + self.gamma_rl * discounted_reward_sum
            returns_g_calc[t] = discounted_reward_sum
        returns_g_tensor = returns_g_calc.to(self.device)


        advantages = returns_g_tensor - 0 
        
        if log_probs.ndim == 0: log_probs = log_probs.unsqueeze(0)
        if advantages.ndim == 0: advantages = advantages.unsqueeze(0)
        
        policy_loss = (-log_probs * advantages).mean() 
        
        if state_values.ndim == 0: state_values = state_values.unsqueeze(0) 
        if returns_g_tensor.ndim == 0: returns_g_tensor = returns_g_tensor.unsqueeze(0)

        value_loss = F.mse_loss(state_values, returns_g_tensor) 

        self.heuristic_net.train(); self.value_net.train() 
        
        self.optimizer_heuristic.zero_grad()
        if policy_loss.requires_grad: 
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.heuristic_net.parameters(), 1.0) 
            self.optimizer_heuristic.step()

        self.optimizer_value.zero_grad()
        if value_loss.requires_grad: 
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1.0) 
            self.optimizer_value.step()

        self.episode_log_probs = []; self.episode_rewards = []; self.episode_state_values_for_baseline = []
        return policy_loss.item() if isinstance(policy_loss, torch.Tensor) else float(policy_loss), \
               value_loss.item() if isinstance(value_loss, torch.Tensor) else float(value_loss)


    def save_model(self, base_path):
        heuristic_path = base_path.replace(".pth", "_heuristic.pth")
        value_path = base_path.replace(".pth", "_value.pth")
        torch.save(self.heuristic_net.state_dict(), heuristic_path)
        torch.save(self.value_net.state_dict(), value_path)
        print(f"DeepACOPlanner models saved: {heuristic_path}, {value_path}")

    def load_model(self, base_path):
        heuristic_path = base_path.replace(".pth", "_heuristic.pth")
        value_path = base_path.replace(".pth", "_value.pth")
        loaded_heuristic = False; loaded_value = False
        if os.path.exists(heuristic_path):
            try:
                self.heuristic_net.load_state_dict(torch.load(heuristic_path, map_location=self.device))
                self.heuristic_net.eval() 
                print(f"HeuristicNet loaded from {heuristic_path}"); loaded_heuristic = True
            except Exception as e: print(f"Error loading HeuristicNet: {e}")
        else: print(f"HeuristicNet model path not found: {heuristic_path}")
        
        if os.path.exists(value_path):
            try:
                self.value_net.load_state_dict(torch.load(value_path, map_location=self.device))
                self.value_net.eval()
                print(f"ValueNet loaded from {value_path}"); loaded_value = True
            except Exception as e: print(f"Error loading ValueNet: {e}")
        else: print(f"ValueNet model path not found: {value_path}")
        
        if not (loaded_heuristic and loaded_value):
             print("Warning: Not all models loaded successfully. May start with fresh weights for some.")