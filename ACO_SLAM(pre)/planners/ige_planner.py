# planners/ige_planner.py
from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN

class IGEPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range

    def _calculate_information_gain(self, frontier_pos, known_map):
        ig = 0; r_f, c_f = frontier_pos
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr, nc = r_f + dr, c_f + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    ig += 1
        return ig

    def plan_next_action(self, robot_pos, known_map, **kwargs): # Added **kwargs
        frontiers = self.find_frontiers(known_map)
        # print(f"IGE RUN: Robot@ {robot_pos}. Found {len(frontiers)} initial frontiers.")
        if not frontiers:
            # print("IGE RUN: No frontiers found at all. Terminating plan.")
            return None, None

        best_utility = -float('inf')
        best_frontier_ige = None 
        best_path_ige = None
        
        # Check all reachable frontiers first, to have them for fallback if needed
        reachable_frontiers_for_fallback = []

        for fr_pos in frontiers:
            path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
            if path:
                path_cost = len(path) - 1
                if path_cost == 0: path_cost = 1e-5
                
                # Store for potential fallback
                reachable_frontiers_for_fallback.append({"pos": fr_pos, "path": path, "cost": path_cost})

                information_gain = self._calculate_information_gain(fr_pos, known_map)
                utility = -1.0 
                if information_gain <= 0 and path_cost > 1e-4 : utility = -1.0 
                elif information_gain <=0 and path_cost <=1e-4: utility = 0.1 
                else: utility = information_gain / path_cost
                
                # print(f"IGE RUN: Frontier {fr_pos}, IG: {information_gain}, Cost: {path_cost:.2f}, Utility: {utility:.2f}")
                if utility > best_utility:
                    best_utility = utility
                    best_frontier_ige = fr_pos
                    best_path_ige = path
        
        # --- Decision based on IGE logic or Final Fallback ---
        if best_frontier_ige is not None: # IGE logic found a target
            # print(f"IGE Selected (Primary Logic): {best_frontier_ige} with utility {best_utility:.2f}")
            return best_frontier_ige, best_path_ige
        else: # IGE logic failed to select a target, try final fallback
            # print(f"IGE: Primary logic failed (best_utility: {best_utility}). Executing _final_fallback_plan.")
            # The _final_fallback_plan will re-evaluate frontiers and reachability.
            # This is slightly redundant if reachable_frontiers_for_fallback is already populated,
            # but _final_fallback_plan is a self-contained "last resort".
            fallback_target, fallback_path = self._final_fallback_plan(robot_pos, known_map)
            # if fallback_target:
            #     print(f"IGE Selected (Final Fallback): {fallback_target}")
            # else:
            #     print(f"IGE: Final Fallback also failed. No target.")
            return fallback_target, fallback_path