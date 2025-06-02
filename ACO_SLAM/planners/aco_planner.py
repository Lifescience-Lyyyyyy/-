# planners/aco_planner.py
from .base_planner import BasePlanner
import numpy as np
import random
from environment import UNKNOWN #, FREE, OBSTACLE (FREE/OBSTACLE not directly used here)

class ACOPlanner(BasePlanner):
    def __init__(self, environment, n_ants=25, n_iterations=40,
                 alpha=1.2, beta=3.5, evaporation_rate=0.25, q0=0.7,
                 pheromone_min=0.05, pheromone_max=50.0,
                 ig_weight_heuristic=3.0,
                 robot_sensor_range_for_heuristic=3,
                 visualize_ants_callback=None):
        super().__init__(environment)
        self.n_ants = n_ants; self.n_iterations = n_iterations
        self.alpha = alpha; self.beta = beta
        self.evaporation_rate = evaporation_rate; self.q0 = q0
        self.pheromones = {}; self.current_path_to_target = None
        self.visualize_ants_callback = visualize_ants_callback
        self.pheromone_min = pheromone_min; self.pheromone_max = pheromone_max
        self.ig_weight_heuristic = ig_weight_heuristic
        self.robot_sensor_range_for_heuristic = robot_sensor_range_for_heuristic

    def _get_heuristic_value_and_path(self, robot_pos, frontier_pos, known_map):
        path = self._is_reachable_and_get_path(robot_pos, frontier_pos, known_map)
        if not path: return -1.0, None
        path_cost = len(path) - 1
        ig = 0; r_f, c_f = frontier_pos
        for dr in range(-self.robot_sensor_range_for_heuristic, self.robot_sensor_range_for_heuristic + 1):
            for dc in range(-self.robot_sensor_range_for_heuristic, self.robot_sensor_range_for_heuristic + 1):
                nr, nc = r_f + dr, c_f + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    ig += 1
        heuristic_val = (self.ig_weight_heuristic * ig + 0.1) / (path_cost + 0.1)
        return heuristic_val, path

    def plan_next_action(self, robot_pos, known_map, **kwargs): # Added **kwargs
        frontiers = self.find_frontiers(known_map)
        # print(f"ACO Debug @ Robot {robot_pos}: Found {len(frontiers)} initial frontiers.")
        if not frontiers: 
            # print("ACO Debug: No frontiers found at all."); 
            return None, None

        initial_pheromone_value = (self.pheromone_min + self.pheromone_max) / 2.0
        reachable_frontiers_info = []
        for f_pos in frontiers:
            heuristic_val, path = self._get_heuristic_value_and_path(robot_pos, f_pos, known_map)
            if path and heuristic_val >= 0:
                reachable_frontiers_info.append({'pos': f_pos, 'heuristic': heuristic_val, 'path': path})
                if f_pos not in self.pheromones: self.pheromones[f_pos] = initial_pheromone_value
        
        # print(f"ACO Debug: Reachable frontiers count: {len(reachable_frontiers_info)}")
        if not reachable_frontiers_info:
            # print("ACO Debug: Frontiers found, but none valid/reachable for ACO.")
            # Even if ACO's heuristic makes all options bad, try final fallback
            return self._final_fallback_plan(robot_pos, known_map)


        current_frontiers_set = {info['pos'] for info in reachable_frontiers_info}
        all_ant_paths_for_viz = []

        for iteration in range(self.n_iterations):
            iteration_ant_paths = []; ant_choices_this_iteration = []
            for ant_k in range(self.n_ants):
                candidate_scores = []
                for f_info in reachable_frontiers_info:
                    f_pos = f_info['pos']; eta = f_info['heuristic']
                    tau = self.pheromones.get(f_pos, initial_pheromone_value)
                    score = (tau**self.alpha) * (eta**self.beta)
                    candidate_scores.append({'frontier_info': f_info, 'selection_score': score})

                chosen_f_info = None
                if not candidate_scores: continue

                if random.random() < self.q0 and candidate_scores:
                    positive_candidates = [cs for cs in candidate_scores if cs['selection_score'] > 0]
                    if positive_candidates: chosen_f_info = max(positive_candidates, key=lambda x: x['selection_score'])['frontier_info']
                    else: chosen_f_info = max(candidate_scores, key=lambda x: x['selection_score'])['frontier_info']
                else:
                    positive_score_candidates = [cs for cs in candidate_scores if cs['selection_score'] > 0]
                    prob_sum = sum(cs['selection_score'] for cs in positive_score_candidates)
                    if prob_sum <= 1e-9:
                        if reachable_frontiers_info:
                             best_h_fallback = max(reachable_frontiers_info, key=lambda x: x['heuristic'], default=None)
                             if best_h_fallback and best_h_fallback['heuristic'] > 0 : chosen_f_info = best_h_fallback
                             else: chosen_f_info = random.choice(reachable_frontiers_info)
                    else:
                        r_val = random.random() * prob_sum; cumulative_prob = 0.0
                        for cs in positive_score_candidates:
                            cumulative_prob += cs['selection_score']
                            if r_val <= cumulative_prob: chosen_f_info = cs['frontier_info']; break
                        if chosen_f_info is None and positive_score_candidates: chosen_f_info = positive_score_candidates[-1]['frontier_info']
                if chosen_f_info is None and reachable_frontiers_info: chosen_f_info = random.choice(reachable_frontiers_info)

                if chosen_f_info:
                    deposit_value = chosen_f_info['heuristic'] 
                    ant_choices_this_iteration.append({'frontier_info': chosen_f_info, 'deposit_value': deposit_value})
                    iteration_ant_paths.append(chosen_f_info['path'])
            
            all_ant_paths_for_viz.append(iteration_ant_paths)
            active_pheromone_keys = list(self.pheromones.keys())
            for f_pos_key in active_pheromone_keys:
                if f_pos_key in current_frontiers_set:
                    self.pheromones[f_pos_key] *= (1.0 - self.evaporation_rate)
                    self.pheromones[f_pos_key] = max(self.pheromone_min, self.pheromones[f_pos_key])
                else: del self.pheromones[f_pos_key]
            for choice in ant_choices_this_iteration:
                f_pos = choice['frontier_info']['pos']; deposit_amount = choice['deposit_value']
                if f_pos in self.pheromones and deposit_amount > 0 :
                    self.pheromones[f_pos] += deposit_amount
                    self.pheromones[f_pos] = min(self.pheromone_max, self.pheromones[f_pos])

        if self.visualize_ants_callback and all_ant_paths_for_viz:
            self.visualize_ants_callback(robot_pos, known_map, all_ant_paths_for_viz, self.environment)

        best_score = -float('inf'); final_choice_pos = None; final_choice_path = None
        if not reachable_frontiers_info: # Should be caught earlier
            return self._final_fallback_plan(robot_pos, known_map) 

        for f_info in reachable_frontiers_info:
            f_pos = f_info['pos']
            final_score = self.pheromones.get(f_pos, initial_pheromone_value)**self.alpha * (f_info['heuristic']**self.beta)
            if final_score > best_score:
                best_score = final_score; final_choice_pos = f_pos; final_choice_path = f_info['path']
        
        # --- Decision based on ACO logic or Final Fallback ---
        if final_choice_pos is not None: # ACO logic found a target
            # print(f"ACO Selected (Primary Logic): {final_choice_pos} with score {best_score:.2e}")
            self.current_path_to_target = final_choice_path
            return final_choice_pos, final_choice_path
        else: # ACO logic failed (e.g. all scores too low), try final fallback
            # print(f"ACO: Primary logic failed (best_score: {best_score}). Executing _final_fallback_plan.")
            fallback_target, fallback_path = self._final_fallback_plan(robot_pos, known_map)
            # if fallback_target:
            #     print(f"ACO Selected (Final Fallback): {fallback_target}")
            # else:
            #     print(f"ACO: Final Fallback also failed. No target.")
            self.current_path_to_target = fallback_path # May be None
            return fallback_target, fallback_path