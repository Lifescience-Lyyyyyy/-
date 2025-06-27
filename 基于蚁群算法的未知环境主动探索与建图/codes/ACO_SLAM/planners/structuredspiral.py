import numpy as np
import random
from .base_planner import BasePlanner
from environment import UNKNOWN, FREE, OBSTACLE
from .geometry_utils import check_line_of_sight 

class StructuredSpiralPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.height = self.environment.height
        self.width = self.environment.width

        self.PHASE_INITIAL_BOXING = 1
        self.PHASE_SPIRAL_INWARD = 2
        self.current_phase = self.PHASE_INITIAL_BOXING

        self._boxing_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)] 
        self._boxing_current_dir_idx = 0
        self._boxing_leg_targets = [
            self.height - 1 - self.robot_sensor_range, # Bottom edge
            self.width - 1 - self.robot_sensor_range,  # Right edge
            self.robot_sensor_range,                   # Top edge
            self.robot_sensor_range                    # Left edge
        ]
        self._boxing_legs_completed = 0 

        self._spiral_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U for inward spiral
        self._spiral_current_dir_idx = 0
        self._spiral_steps_taken_in_leg = 0
        self._spiral_current_leg_length = 1
        self._spiral_legs_done_at_current_length = 0 

        self._last_chosen_target = None 

    def _calculate_ig_simple(self, r_prospective, c_prospective, known_map):
        """Simple IG: count unknown cells in sensor range, no LoS for speed in planning."""
        ig = 0
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr, nc = r_prospective + dr, c_prospective + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    ig += 1
        return ig

    def _find_best_frontier_in_direction(self, robot_pos, known_map, target_direction_vector, frontiers_list):
        """
        From frontiers_list, find the one that is "most in" the target_direction_vector
        from robot_pos, has good IG, and is reachable.
        Returns: (best_frontier, path_to_frontier)
        """
        best_utility = -float('inf')
        chosen_frontier = None
        chosen_path = None


        for fr_pos in frontiers_list:
            if fr_pos == self._last_chosen_target: 
                continue

            vec_to_fr = (fr_pos[0] - robot_pos[0], fr_pos[1] - robot_pos[1])

            dot_product = vec_to_fr[0] * target_direction_vector[0] + vec_to_fr[1] * target_direction_vector[1]

            if dot_product < 0.1 : 
                continue

            path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
            if path:
                path_cost = len(path) - 1
                if path_cost <= 0: path_cost = 1e-5

                ig = self._calculate_ig_simple(fr_pos[0], fr_pos[1], known_map) 
   
                utility = (ig + 0.1) * (dot_product + 0.1) / path_cost 
                                
                if utility > best_utility:
                    best_utility = utility
                    chosen_frontier = fr_pos
                    chosen_path = path
        
        return chosen_frontier, chosen_path


    def plan_next_action(self, robot_pos, known_map, **kwargs):
        all_frontiers = self.find_frontiers(known_map)
        if not all_frontiers:
            return None, None

        target_frontier = None
        path_to_target = None

        if self.current_phase == self.PHASE_INITIAL_BOXING:
            current_target_direction = self._boxing_directions[self._boxing_current_dir_idx]
            leg_coord_target = self._boxing_leg_targets[self._boxing_current_dir_idx]

            leg_completed = False
            if self._boxing_current_dir_idx == 0: # Down
                if robot_pos[0] >= leg_coord_target: leg_completed = True
            elif self._boxing_current_dir_idx == 1: # Right
                if robot_pos[1] >= leg_coord_target: leg_completed = True
            elif self._boxing_current_dir_idx == 2: # Up
                if robot_pos[0] <= leg_coord_target: leg_completed = True
            elif self._boxing_current_dir_idx == 3: # Left
                if robot_pos[1] <= leg_coord_target: leg_completed = True
            
            if leg_completed:
                self._boxing_current_dir_idx = (self._boxing_current_dir_idx + 1) % 4
                self._boxing_legs_completed += 1
                current_target_direction = self._boxing_directions[self._boxing_current_dir_idx] 
                if self._boxing_legs_completed >= 4: 
                    self.current_phase = self.PHASE_SPIRAL_INWARD
                    self._spiral_current_dir_idx = 0 
                    self._spiral_steps_taken_in_leg = 0
                    self._spiral_current_leg_length = 1 
                    self._spiral_legs_done_at_current_length = 0

            if self.current_phase == self.PHASE_INITIAL_BOXING: 
                target_frontier, path_to_target = self._find_best_frontier_in_direction(
                    robot_pos, known_map, current_target_direction, all_frontiers
                )
                if target_frontier:
                    self._last_chosen_target = target_frontier
                    return target_frontier, path_to_target
                else: 
                    self._boxing_current_dir_idx = (self._boxing_current_dir_idx + 1) % 4
                    self._boxing_legs_completed += 1 
                    if self._boxing_legs_completed >= 8: 
                         self.current_phase = self.PHASE_SPIRAL_INWARD


        if self.current_phase == self.PHASE_SPIRAL_INWARD:
          
            current_spiral_direction_vec = self._spiral_directions[self._spiral_current_dir_idx]
            target_frontier, path_to_target = self._find_best_frontier_in_direction(
                robot_pos, known_map, current_spiral_direction_vec, all_frontiers
            )

            if target_frontier:
                self._last_chosen_target = target_frontier
                self._spiral_steps_taken_in_leg += 1 
                if self._spiral_steps_taken_in_leg >= self._spiral_current_leg_length:
                    self._spiral_current_dir_idx = (self._spiral_current_dir_idx + 1) % 4
                    self._spiral_steps_taken_in_leg = 0
                    self._spiral_legs_done_at_current_length += 1
                    if self._spiral_legs_done_at_current_length == 2:
                        self._spiral_current_leg_length += 1
                        self._spiral_legs_done_at_current_length = 0
                return target_frontier, path_to_target
            else:

                for _ in range(len(self._spiral_directions) -1): 
                    self._spiral_current_dir_idx = (self._spiral_current_dir_idx + 1) % 4
                    self._spiral_steps_taken_in_leg = 0 
                    self._spiral_legs_done_at_current_length = 0 

                    current_spiral_direction_vec = self._spiral_directions[self._spiral_current_dir_idx]
                    target_frontier_turned, path_to_target_turned = self._find_best_frontier_in_direction(
                        robot_pos, known_map, current_spiral_direction_vec, all_frontiers
                    )
                    if target_frontier_turned:
                        self._last_chosen_target = target_frontier_turned
                        self._spiral_steps_taken_in_leg += 1 
                        return target_frontier_turned, path_to_target_turned
        
        self._last_chosen_target = None 
        return self._final_fallback_plan(robot_pos, known_map)