# planners/ure_planner.py
from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN, FREE

class UREPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.observation_counts = np.zeros_like(environment.grid_known, dtype=int)
        self.exploration_weight = 0.2 
        self.consolidation_weight = 0.8
        self.min_utility_threshold = 1e-4 

    def update_observation_counts(self, robot_pos, known_map_after_sense):
        r_rob, c_rob = robot_pos
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr, nc = r_rob + dr, c_rob + dc
                if self.environment.is_within_bounds(nr, nc) and known_map_after_sense[nr, nc] == FREE:
                    self.observation_counts[nr, nc] += 1

    def _calculate_exploration_utility(self, path_cost, frontier_pos, known_map): # path and cost passed in
        ig = 0; r_f, c_f = frontier_pos
        for dr_s in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc_s in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr_s, nc_s = r_f + dr_s, c_f + dc_s
                if self.environment.is_within_bounds(nr_s, nc_s) and known_map[nr_s, nc_s] == UNKNOWN:
                    ig += 1
        if ig <= 0 and path_cost > 1e-4: return -1.0
        if ig <=0 and path_cost <= 1e-4 : return 0.1
        return ig / (path_cost + 1e-5) # ensure path_cost isn't exactly 0 for division

    def _find_consolidation_candidates(self, known_map):
        candidates = []; low_obs_threshold = 2 
        rows, cols = np.where((known_map == FREE) & (self.observation_counts <= low_obs_threshold))
        for r, c in zip(rows, cols): candidates.append({"pos": (r,c), "obs_count": self.observation_counts[r,c]})
        candidates.sort(key=lambda x: x["obs_count"])
        return candidates[:30]

    def _calculate_consolidation_utility(self, path_cost, obs_count): # path and cost passed in
        return (1.0 / (obs_count + 0.1)) / (path_cost + 0.1)

    def plan_next_action(self, robot_pos, known_map, **kwargs): # Added **kwargs
        best_overall_utility = -float('inf') 
        best_target_ure = None
        best_path_ure = None
        
        # --- 1. Evaluate Exploration Targets (Frontiers) ---
        frontiers = self.find_frontiers(known_map)
        # print(f"URE Debug: Found {len(frontiers)} frontiers.")
        if frontiers:
            for fr_pos in frontiers:
                path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
                if path:
                    path_cost = len(path) - 1
                    expl_utility = self._calculate_exploration_utility(path_cost, fr_pos, known_map)
                    weighted_utility = self.exploration_weight * expl_utility
                    # print(f"URE Debug: Frontier {fr_pos}, ExplUtil: {expl_utility:.2f}, Weighted: {weighted_utility:.2f}")
                    if weighted_utility > best_overall_utility:
                        best_overall_utility = weighted_utility
                        best_target_ure = fr_pos
                        best_path_ure = path

        # --- 2. Evaluate Consolidation Targets ---
        consolidation_candidates = self._find_consolidation_candidates(known_map)
        # print(f"URE Debug: Found {len(consolidation_candidates)} consolidation candidates.")
        if consolidation_candidates:
            for cand_info in consolidation_candidates:
                ct_pos, obs_count = cand_info["pos"], cand_info["obs_count"]
                path = self._is_reachable_and_get_path(robot_pos, ct_pos, known_map)
                if path:
                    path_cost = len(path) -1
                    cons_utility = self._calculate_consolidation_utility(path_cost, obs_count)
                    weighted_utility = self.consolidation_weight * cons_utility
                    # print(f"URE Debug: Consolidate {ct_pos} (obs:{obs_count}), ConsUtil: {cons_utility:.2f}, Weighted: {weighted_utility:.2f}")
                    if weighted_utility > best_overall_utility:
                        best_overall_utility = weighted_utility
                        best_target_ure = ct_pos
                        best_path_ure = path
        
        # --- Decision based on URE logic or Final Fallback ---
        if best_target_ure is not None and best_overall_utility >= self.min_utility_threshold:
            # URE's primary logic (exploration + consolidation with weights) found a good enough target
            # print(f"URE Selected (Primary Logic): {best_target_ure} with overall utility {best_overall_utility:.3f}")
            return best_target_ure, best_path_ure
        else: 
            # Primary URE logic failed or found a low utility target. Execute final fallback.
            # print(f"URE: Primary logic insufficient (best_utility: {best_overall_utility:.3f}). Executing _final_fallback_plan.")
            fallback_target, fallback_path = self._final_fallback_plan(robot_pos, known_map)
            # if fallback_target:
            #     print(f"URE Selected (Final Fallback): {fallback_target}")
            # else:
            #     print(f"URE: Final Fallback also failed. No target.")
            return fallback_target, fallback_path