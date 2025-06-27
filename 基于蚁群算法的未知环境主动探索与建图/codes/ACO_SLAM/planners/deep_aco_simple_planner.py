import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import os

from .base_planner import BasePlanner 
from environment import UNKNOWN, FREE, OBSTACLE 
try:
    from .geometry_utils import check_line_of_sight
    from .pathfinding import heuristic as manhattan_heuristic
except ImportError:
    def check_line_of_sight(r1,c1,r2,c2,m): return True 
    def manhattan_heuristic(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

class Actor(nn.Module):
    """ Policy Network (outputs heuristic scores) """
    def __init__(self, num_inputs=4, hidden_dim1=64, hidden_dim2=32):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
    def forward(self, state_features_batch):
        x = F.relu(self.fc1(state_features_batch))
        x = F.relu(self.fc2(x))
        return F.softplus(self.fc3(x)) + 1e-6

class Critic(nn.Module):
    """ Value Network """
    def __init__(self, num_inputs=2, hidden_dim1=64, hidden_dim2=32):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_inputs, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)
    def forward(self, global_state_features_batch):
        x = F.relu(self.fc1(global_state_features_batch))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPOMemory:
    """ Buffer to store a batch of trajectories for PPO update. """
    def __init__(self):
        self.actions, self.states, self.frontier_features = [], [], []
        self.log_probs, self.rewards, self.is_terminals = [], [], []
    def clear(self):
        del self.actions[:]; del self.states[:]; del self.frontier_features[:]
        del self.log_probs[:]; del self.rewards[:]; del self.is_terminals[:]

class DeepACOSimplePlanner(BasePlanner):
    def __init__(self, environment,
                 n_ants=10, n_iterations=5, 
                 alpha=1.0, beta=1.5, 
                 evaporation_rate=0.1, 
                 pheromone_initial=0.1, pheromone_min=0.01, pheromone_max=5.0,
                 local_frontier_ig_sum_radius=5,
                 robot_sensor_range=5,
                 learning_rate_actor=3e-4, learning_rate_critic=1e-3,
                 gamma_ppo=0.99, eps_clip_ppo=0.2, ppo_epochs=4,
                 ppo_batch_size=64, 
                 is_training=True,  
                 aco_softmax_temperature=1.0
                 ):
        super().__init__(environment)
        # ACO params
        self.n_ants = n_ants; self.n_iterations = n_iterations
        self.alpha = alpha; self.beta_nn_heuristic = beta
        self.evaporation_rate = evaporation_rate; 
        self.pheromones = {} 
        self.pheromone_min = pheromone_min; self.pheromone_max = pheromone_max
        self.pheromone_initial = pheromone_initial
        
        # Feature extraction params
        self.local_frontier_ig_sum_radius = local_frontier_ig_sum_radius
        self.robot_sensor_range = robot_sensor_range
        self.aco_softmax_temperature = aco_softmax_temperature
        # PPO params
        self.lr_actor = learning_rate_actor; self.lr_critic = learning_rate_critic
        self.gamma = gamma_ppo; self.eps_clip = eps_clip_ppo
        self.ppo_epochs = ppo_epochs; self.ppo_batch_size = ppo_batch_size
        
        self.is_training = is_training
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DeepACOSimplePlanner using device: {self.device}")

        self.actor = Actor(num_inputs=4).to(self.device)
        self.critic = Critic(num_inputs=2).to(self.device)
        
        if self.is_training:
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
            self.policy_old = Actor(num_inputs=4).to(self.device)
            self.policy_old.load_state_dict(self.actor.state_dict())
            self.memory = PPOMemory()
            self.MseLoss = nn.MSELoss()
        else:
            self.actor.eval()
            self.critic.eval()
            self.policy_old = None
            self.memory = None

    def _calculate_ig_with_los(self, frontier_pos, known_map, sensor_range):
        ig = 0; r_f, c_f = frontier_pos
        for dr in range(-sensor_range, sensor_range + 1):
            for dc in range(-sensor_range, sensor_range + 1):
                nr, nc = r_f + dr, c_f + dc
                if self.environment.is_within_bounds(nr, nc) and \
                   known_map[nr, nc] == UNKNOWN and \
                   check_line_of_sight(r_f, c_f, nr, nc, known_map):
                    ig += 1
        return ig

    def _get_features_for_frontier(self, robot_pos, frontier_pos, path_cost, known_map, all_frontiers):
        ig = self._calculate_ig_with_los(frontier_pos, known_map, self.robot_sensor_range)
        max_dist = self.environment.height + self.environment.width
        norm_cost = path_cost / (max_dist + 1e-5)
        explored_ratio = np.sum(known_map != UNKNOWN) / (self.environment.height * self.environment.width)
        local_ig_sum = 0
        for other_fr in all_frontiers:
            if manhattan_heuristic(frontier_pos, other_fr) <= self.local_frontier_ig_sum_radius:
                local_ig_sum += self._calculate_ig_with_los(other_fr, known_map, self.robot_sensor_range)
        max_local_ig = (np.pi * self.local_frontier_ig_sum_radius**2) * 5 
        norm_local_ig_sum = local_ig_sum / (max_local_ig + 1e-5)
        return np.array([ig, norm_cost, explored_ratio, norm_local_ig_sum], dtype=np.float32)

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        frontiers = self.find_frontiers(known_map)
        if not frontiers: return None, None
        self.pheromones = {} 
        reachable_frontiers_info = []; frontier_features_list = []
        for f_pos in frontiers:
            path = self._is_reachable_and_get_path(robot_pos, f_pos, known_map)
            if path:
                path_cost = len(path) - 1
                features = self._get_features_for_frontier(robot_pos, f_pos, path_cost, known_map, frontiers)
                frontier_features_list.append(features)
                reachable_frontiers_info.append({'pos': f_pos, 'path': path, 'cost': path_cost})
                self.pheromones[f_pos] = self.pheromone_initial
        if not reachable_frontiers_info: return self._final_fallback_plan(robot_pos, known_map)
        
        policy_to_use = self.policy_old if self.is_training else self.actor
        frontier_features_tensor = torch.tensor(np.array(frontier_features_list), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            eta_nn_values = policy_to_use(frontier_features_tensor).squeeze(-1)
        
        for i, info in enumerate(reachable_frontiers_info):
            info['eta_nn'] = eta_nn_values[i].item()

        for _ in range(self.n_iterations):
            ant_choices_this_iter = []
            
            scores_for_selection = []
            for f_info in reachable_frontiers_info:
                tau = self.pheromones[f_info['pos']]
                eta = f_info['eta_nn']
                score = (tau ** self.alpha) * (eta ** self.beta_nn_heuristic)
                scores_for_selection.append(score)
            
            scores_tensor = torch.tensor(scores_for_selection, dtype=torch.float32).to(self.device)
            probabilities_tensor = F.softmax(scores_tensor / self.aco_softmax_temperature, dim=0)
            
            probabilities = probabilities_tensor.cpu().numpy()
            

            if np.isnan(probabilities).any() or abs(probabilities.sum() - 1.0) > 1e-6:
                ant_choices_indices = np.random.choice(len(reachable_frontiers_info), size=self.n_ants)
            else:
                ant_choices_indices = np.random.choice(
                    len(reachable_frontiers_info),
                    size=self.n_ants,
                    p=probabilities
                )

            for idx in ant_choices_indices:
                ant_choices_this_iter.append(reachable_frontiers_info[idx])

            for f_pos in self.pheromones: self.pheromones[f_pos] *= (1.0 - self.evaporation_rate)
            
            for choice_info in ant_choices_this_iter:
                self.pheromones[choice_info['pos']] += choice_info['eta_nn'] 
            
            for f_pos in self.pheromones:
                self.pheromones[f_pos] = np.clip(self.pheromones[f_pos], self.pheromone_min, self.pheromone_max)

        final_scores = []
        for f_info in reachable_frontiers_info:
            final_tau = self.pheromones[f_info['pos']]; final_eta = f_info['eta_nn']
            final_scores.append((final_tau ** self.alpha) * (final_eta ** self.beta_nn_heuristic))
        final_scores_tensor = torch.tensor(final_scores, dtype=torch.float32).to(self.device)
        
        final_action_probs_dist = Categorical(logits=final_scores_tensor / self.aco_softmax_temperature) 
        
        if self.is_training:
            chosen_action_index = final_action_probs_dist.sample()
            
            policy_dist_for_logprob = Categorical(logits=eta_nn_values)
            log_prob_of_action = policy_dist_for_logprob.log_prob(chosen_action_index)
            
            global_state_features = np.array([np.sum(known_map != UNKNOWN) / (self.environment.height * self.environment.width),
                                              len(frontiers) / (self.environment.height * self.environment.width * 0.3)], dtype=np.float32)
            self.memory.states.append(global_state_features)
            self.memory.frontier_features.append(frontier_features_list) 
            self.memory.actions.append(chosen_action_index.item())
            self.memory.log_probs.append(log_prob_of_action)
        else: 
            chosen_action_index = torch.argmax(final_scores_tensor)

        final_chosen_frontier_info = reachable_frontiers_info[chosen_action_index.item()]
        return final_chosen_frontier_info['pos'], final_chosen_frontier_info['path']


    def update_ppo(self):
        """Full PPO update with mini-batching."""

        if not self.memory.rewards or len(self.memory.rewards) != len(self.memory.actions):
            print(f"Warning (update_ppo): Data mismatch or empty rewards. "
                  f"Actions: {len(self.memory.actions)}, Rewards: {len(self.memory.rewards)}. Skipping update.")
            self.memory.clear() 
            return 0.0, 0.0

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal: discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        num_valid_transitions = len(self.memory.rewards)
        old_log_probs = torch.stack(self.memory.log_probs[:num_valid_transitions]).to(self.device).detach()
        old_states = torch.tensor(np.array(self.memory.states[:num_valid_transitions]), dtype=torch.float32).to(self.device)

        for _ in range(self.ppo_epochs):
            indices = np.arange(num_valid_transitions) 
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_valid_transitions, self.ppo_batch_size):
                end_idx = start_idx + self.ppo_batch_size
                minibatch_indices = indices[start_idx:end_idx]

                batch_log_probs, batch_state_values, batch_dist_entropy = [], [], []
                for i in minibatch_indices:
                    frontier_features_for_state_i = torch.tensor(np.array(self.memory.frontier_features[i]), dtype=torch.float32).to(self.device)
                    new_eta_values = self.actor(frontier_features_for_state_i).squeeze(-1)
                    new_dist = Categorical(logits=new_eta_values)
                    
                    action_idx_tensor = torch.tensor(self.memory.actions[i]).to(self.device)
                    batch_log_probs.append(new_dist.log_prob(action_idx_tensor))
                    batch_dist_entropy.append(new_dist.entropy())

                    state_tensor = torch.tensor(self.memory.states[i], dtype=torch.float32).unsqueeze(0).to(self.device)
                    batch_state_values.append(self.critic(state_tensor))

                batch_log_probs = torch.stack(batch_log_probs)
                batch_state_values = torch.stack(batch_state_values).squeeze()
                batch_dist_entropy = torch.stack(batch_dist_entropy)

                with torch.no_grad(): 
                    old_state_values_for_advantage = self.critic(old_states[minibatch_indices]).squeeze()
                
                minibatch_advantages = rewards[minibatch_indices] - old_state_values_for_advantage

                ratios = torch.exp(batch_log_probs - old_log_probs[minibatch_indices])
                surr1 = ratios * minibatch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * minibatch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = self.MseLoss(batch_state_values, rewards[minibatch_indices])
                
                entropy_bonus = -0.01 * batch_dist_entropy.mean()
                
                loss = policy_loss + 0.5 * value_loss + entropy_bonus

                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()
                loss.backward()
                self.optimizer_actor.step()
                self.optimizer_critic.step()
        
        self.policy_old.load_state_dict(self.actor.state_dict())
        self.memory.clear()
        
        return policy_loss.item(), value_loss.item()
    
    def save_model(self, base_path):
        actor_path = base_path.replace(".pth", "_actor.pth")
        critic_path = base_path.replace(".pth", "_critic.pth")
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)
        print(f"DeepACOSimplePlanner models saved to base path: {base_path}")

    def load_model(self, base_path):
        actor_path = base_path.replace(".pth", "_actor.pth")
        critic_path = base_path.replace(".pth", "_critic.pth")
        loaded_actor = False
        if os.path.exists(actor_path):
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                if self.is_training: self.policy_old.load_state_dict(self.actor.state_dict())
                self.actor.eval() if not self.is_training else self.actor.train()
                print(f"ActorNet loaded from {actor_path}")
                loaded_actor = True
            except Exception as e: print(f"Error loading ActorNet: {e}")
        else: print(f"ActorNet model path not found: {actor_path}")
        
        if os.path.exists(critic_path):
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                self.critic.eval() if not self.is_training else self.critic.train()
                print(f"CriticNet loaded from {critic_path}")
            except Exception as e: print(f"Error loading CriticNet: {e}")
        
        if not loaded_actor: print("Warning: Actor model not loaded. Planner will use random weights.")