from .base_planner import BasePlanner
import numpy as np
import random
from environment import UNKNOWN
# 从新的 geometry_utils 模块导入 LoS 检查函数
from .geometry_utils import check_line_of_sight

class ACOPlanner(BasePlanner):
    def __init__(self, environment, n_ants=25, n_iterations=40,
                 alpha=1.2, beta=3.5, evaporation_rate=0.25, q0=0.7,
                 pheromone_min=0.05, pheromone_max=50.0,
                 ig_weight_heuristic=3.0,
                 robot_sensor_range_for_heuristic=3, # ACO用于计算IG的传感器范围
                 visualize_ants_callback=None):
        super().__init__(environment)
        self.n_ants = n_ants; self.n_iterations = n_iterations
        self.alpha = alpha; self.beta = beta
        self.evaporation_rate = evaporation_rate; self.q0 = q0
        self.pheromones = {}; 
        self.visualize_ants_callback = visualize_ants_callback
        self.pheromone_min = pheromone_min; self.pheromone_max = pheromone_max
        self.ig_weight_heuristic = ig_weight_heuristic
        self.robot_sensor_range_for_heuristic = robot_sensor_range_for_heuristic # 存储此参数

    def _calculate_ig_with_los_for_aco(self, 
                                       prospective_robot_pos, 
                                       current_known_map,
                                       sensor_range_for_ig): # 与IGE的IG计算函数签名一致
        """为ACO计算信息增益，考虑LoS。"""
        ig = 0
        r_prospective, c_prospective = prospective_robot_pos
        for dr in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
            for dc in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
                # if dr**2 + dc**2 > sensor_range_for_ig**2: continue
                r_target_cell, c_target_cell = r_prospective + dr, c_prospective + dc
                if self.environment.is_within_bounds(r_target_cell, c_target_cell):
                    if current_known_map[r_target_cell, c_target_cell] == UNKNOWN:
                        if check_line_of_sight(r_prospective, c_prospective, 
                                               r_target_cell, c_target_cell, 
                                               current_known_map):
                            ig += 1
        return ig

    def _get_heuristic_value_and_path(self, robot_pos, frontier_pos, known_map_snapshot):
        """计算到特定边界点的启发式值 (eta) 和路径，IG计算考虑LoS。"""
        # 路径规划基于快照
        path = self._is_reachable_and_get_path(robot_pos, frontier_pos, known_map_snapshot)
        if not path: 
            return -1.0, None 
        
        path_cost = len(path) - 1
        if path_cost < 0: path_cost = 0 # Should be 0 if start==frontier_pos

        # 使用带有LoS的新方法计算信息增益，基于快照
        # 使用ACO自己配置的 robot_sensor_range_for_heuristic
        ig = self._calculate_ig_with_los_for_aco(
            frontier_pos, # 机器人假设会到达这个边界点
            known_map_snapshot, # 基于当前的已知情况进行LoS判断
            self.robot_sensor_range_for_heuristic # 使用ACO自己的传感器范围参数
        )
        
        # 启发式值 = (加权信息增益) / (路径成本)
        heuristic_val = (self.ig_weight_heuristic * ig + 0.1) / (path_cost + 0.1)
        return heuristic_val, path

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        frontiers = self.find_frontiers(known_map)
        if not frontiers: 
            # print(f"ACO: No frontiers found from {robot_pos}.")
            return None, None

        initial_pheromone_value = (self.pheromone_min + self.pheromone_max) / 2.0
        reachable_frontiers_info = []
        
        known_map_snapshot = np.copy(known_map) # 创建快照用于本轮规划

        for f_pos in frontiers:
            # 使用快照计算启发式值（包含LoS的IG）和路径
            heuristic_val, path = self._get_heuristic_value_and_path(robot_pos, f_pos, known_map_snapshot)
            if path and heuristic_val >= 0: 
                reachable_frontiers_info.append({'pos': f_pos, 'heuristic': heuristic_val, 'path': path})
                if f_pos not in self.pheromones:
                    self.pheromones[f_pos] = initial_pheromone_value
        
        if not reachable_frontiers_info:
            # print(f"ACO: No reachable and valid frontiers. Falling back.")
            return self._final_fallback_plan(robot_pos, known_map) # 使用原始known_map回退

        current_frontiers_set = {info['pos'] for info in reachable_frontiers_info}
        all_ant_paths_for_viz = []

        for iteration in range(self.n_iterations):
            iteration_ant_paths = []; ant_choices_this_iteration = []
            for _ in range(self.n_ants): # Renamed ant_k to _
                candidate_scores = []
                for f_info in reachable_frontiers_info:
                    f_pos_loop = f_info['pos']; eta_loop = f_info['heuristic'] # 使用已计算好的启发式值
                    tau_loop = self.pheromones.get(f_pos_loop, initial_pheromone_value)
                    score = (tau_loop**self.alpha) * (eta_loop**self.beta)
                    candidate_scores.append({'frontier_info': f_info, 'selection_score': score})

                chosen_f_info = None
                if not candidate_scores: continue

                if random.random() < self.q0: 
                    positive_candidates = [cs for cs in candidate_scores if cs['selection_score'] > 0]
                    if positive_candidates: chosen_f_info = max(positive_candidates, key=lambda x: x['selection_score'])['frontier_info']
                    elif candidate_scores: chosen_f_info = max(candidate_scores, key=lambda x: x['selection_score'])['frontier_info'] # Fallback if no positive
                else: 
                    positive_score_candidates = [cs for cs in candidate_scores if cs['selection_score'] > 0]
                    prob_sum = sum(cs['selection_score'] for cs in positive_score_candidates)
                    if prob_sum <= 1e-9: 
                        if reachable_frontiers_info:                        
                             best_h_fallback = max(reachable_frontiers_info, key=lambda x: x['heuristic'], default=None)
                             if best_h_fallback and best_h_fallback['heuristic'] > 0 : 
                                 chosen_f_info = best_h_fallback
                             else: 
                                 chosen_f_info = random.choice(reachable_frontiers_info)
                    else:
                        r_val = random.random() * prob_sum; cumulative_prob = 0.0
                        for cs in positive_score_candidates:
                            cumulative_prob += cs['selection_score']
                            if r_val <= cumulative_prob: chosen_f_info = cs['frontier_info']; break
                        if chosen_f_info is None and positive_score_candidates: 
                            chosen_f_info = positive_score_candidates[-1]['frontier_info']
                
                if chosen_f_info is None and reachable_frontiers_info: 
                    chosen_f_info = random.choice(reachable_frontiers_info)

                if chosen_f_info:
                    deposit_value = chosen_f_info['heuristic'] 
                    ant_choices_this_iteration.append({'frontier_info': chosen_f_info, 'deposit_value': deposit_value})
                    if self.visualize_ants_callback: 
                        iteration_ant_paths.append(chosen_f_info['path'])
            
            if self.visualize_ants_callback and iteration_ant_paths:
                 all_ant_paths_for_viz.append(iteration_ant_paths)

            active_pheromone_keys = list(self.pheromones.keys())
            for f_pos_key in active_pheromone_keys:
                if f_pos_key in current_frontiers_set:
                    self.pheromones[f_pos_key] *= (1.0 - self.evaporation_rate)
                    self.pheromones[f_pos_key] = max(self.pheromone_min, self.pheromones[f_pos_key])
                else: 
                    if f_pos_key in self.pheromones:
                        del self.pheromones[f_pos_key]
            for choice in ant_choices_this_iteration:
                f_pos_loop = choice['frontier_info']['pos']; deposit_amount = choice['deposit_value']
                if f_pos_loop in self.pheromones and deposit_amount > 0 :
                    self.pheromones[f_pos_loop] += deposit_amount
                    self.pheromones[f_pos_loop] = min(self.pheromone_max, self.pheromones[f_pos_loop])

        if self.visualize_ants_callback and all_ant_paths_for_viz:
            self.visualize_ants_callback(robot_pos, known_map, all_ant_paths_for_viz, self.environment)

        best_score = -float('inf'); final_choice_pos = None; final_choice_path = None
        if not reachable_frontiers_info:
            return self._final_fallback_plan(robot_pos, known_map)

        for f_info in reachable_frontiers_info:
            f_pos_loop = f_info['pos']
            eta_loop = f_info['heuristic'] 
            tau_loop = self.pheromones.get(f_pos_loop, initial_pheromone_value)
            final_score = (tau_loop**self.alpha) * (eta_loop**self.beta)
            if final_score > best_score:
                best_score = final_score; final_choice_pos = f_pos_loop; final_choice_path = f_info['path']
        
        if final_choice_pos is not None:
            return final_choice_pos, final_choice_path
        else:
            return self._final_fallback_plan(robot_pos, known_map)