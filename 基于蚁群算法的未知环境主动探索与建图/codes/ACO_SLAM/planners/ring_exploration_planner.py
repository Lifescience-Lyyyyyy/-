import numpy as np
from .base_planner import BasePlanner
from environment import UNKNOWN, FREE, OBSTACLE

class RingExplorationPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.height = self.environment.height
        self.width = self.environment.width

        self.current_ring_level = 0 
        
        self.ring_step_size = self.robot_sensor_range + 2 

        self._ring_nav_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self._ring_nav_current_dir_idx = 0
        self._ring_nav_legs_completed_this_ring = 0

        self._current_target_on_ring_edge = None 
        self._last_chosen_frontier = None 

        self._initialize_ring_boundaries()

    def _initialize_ring_boundaries(self):
        """Calculates the boundaries for the current ring based on self.current_ring_level."""

        offset = self.current_ring_level * self.ring_step_size + self.robot_sensor_range
        
        self.current_ring_min_r = min(offset, self.height // 2 -1 if self.height // 2 > offset else offset )
        self.current_ring_max_r = max(self.height - 1 - offset, self.height // 2 )
        self.current_ring_min_c = min(offset, self.width // 2 -1 if self.width // 2 > offset else offset)
        self.current_ring_max_c = max(self.width - 1 - offset, self.width // 2)

        if self.current_ring_min_r >= self.current_ring_max_r or \
           self.current_ring_min_c >= self.current_ring_max_c:
            self.current_phase_complete = True 
        else:
            self.current_phase_complete = False

        self._ring_nav_current_dir_idx = 0
        self._ring_nav_legs_completed_this_ring = 0
        self._current_target_on_ring_edge = None 


    def _get_target_for_current_ring_leg(self, robot_pos, known_map):
        """
        Determines a target point on the current "edge" of the ring being traversed.
        The robot tries to move towards this point, prioritizing frontiers in that direction.
        """
        corners = [
            (self.current_ring_min_r, self.current_ring_max_c), 
            (self.current_ring_max_r, self.current_ring_max_c), 
            (self.current_ring_max_r, self.current_ring_min_c), 
            (self.current_ring_min_r, self.current_ring_min_c)  
        ]

        direction_vec = self._ring_nav_directions[self._ring_nav_current_dir_idx]
        
        
        candidate_frontiers_on_leg = []
        all_frontiers = self.find_frontiers(known_map)
        if not all_frontiers: return None, None

        for fr_r, fr_c in all_frontiers:
            on_leg = False
            if self._ring_nav_current_dir_idx == 0: # Moving Right along top_r
                if abs(fr_r - self.current_ring_min_r) <= 2 and fr_c >= robot_pos[1]: on_leg = True
            elif self._ring_nav_current_dir_idx == 1: # Moving Down along right_c
                if abs(fr_c - self.current_ring_max_c) <= 2 and fr_r >= robot_pos[0]: on_leg = True
            elif self._ring_nav_current_dir_idx == 2: # Moving Left along bottom_r
                if abs(fr_r - self.current_ring_max_r) <= 2 and fr_c <= robot_pos[1]: on_leg = True
            elif self._ring_nav_current_dir_idx == 3: # Moving Up along left_c
                if abs(fr_c - self.current_ring_min_c) <= 2 and fr_r <= robot_pos[0]: on_leg = True
            
            if on_leg:
                candidate_frontiers_on_leg.append((fr_r, fr_c))

        if not candidate_frontiers_on_leg:

            return self._find_best_frontier_in_direction(robot_pos, known_map, direction_vec, all_frontiers)
        best_utility = -float('inf')
        chosen_frontier = None
        chosen_path = None

        for fr_pos_on_leg in candidate_frontiers_on_leg:
            if fr_pos_on_leg == self._last_chosen_frontier: continue

            path = self._is_reachable_and_get_path(robot_pos, fr_pos_on_leg, known_map)
            if path:
                path_cost = len(path) -1 
                if path_cost <= 0: path_cost = 1e-5
                
                ig = self._calculate_ig_simple(fr_pos_on_leg[0], fr_pos_on_leg[1], known_map)
                utility = ig / path_cost
                
                if utility > best_utility:
                    best_utility = utility
                    chosen_frontier = fr_pos_on_leg
                    chosen_path = path
        
        return chosen_frontier, chosen_path


    def _find_best_frontier_in_direction(self, robot_pos, known_map, target_direction_vector, frontiers_list):
        """Helper: Finds best frontier in a general direction (reused from StructuredSpiral)."""
        best_utility = -float('inf'); chosen_frontier = None; chosen_path = None
        for fr_pos in frontiers_list:
            if fr_pos == self._last_chosen_frontier: continue
            vec_to_fr = (fr_pos[0] - robot_pos[0], fr_pos[1] - robot_pos[1])
            dot_product = vec_to_fr[0] * target_direction_vector[0] + vec_to_fr[1] * target_direction_vector[1]
            if dot_product < 0.1 : continue
            path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
            if path:
                path_cost = len(path) - 1; 
                if path_cost <= 0: path_cost = 1e-5
                ig = self._calculate_ig_simple(fr_pos[0], fr_pos[1], known_map)
                utility = (ig + 0.1) * (dot_product + 0.1) / path_cost                 
                if utility > best_utility:
                    best_utility = utility; chosen_frontier = fr_pos; chosen_path = path
        return chosen_frontier, chosen_path

    def _calculate_ig_simple(self, r_prospective, c_prospective, known_map): 
        ig = 0
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr, nc = r_prospective + dr, c_prospective + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    ig += 1
        return ig


    def plan_next_action(self, robot_pos, known_map, **kwargs):
        self._last_chosen_frontier = None 

        if self.current_phase_complete: 
            self.current_ring_level += 1
            self._initialize_ring_boundaries()
            if self.current_phase_complete: 
                return self._final_fallback_plan(robot_pos, known_map)
        r, c = robot_pos
        leg_just_finished = False
        if self._ring_nav_current_dir_idx == 0: # Moving Right
            if c >= self.current_ring_max_c - self.robot_sensor_range // 2 : leg_just_finished = True
        elif self._ring_nav_current_dir_idx == 1: # Moving Down
            if r >= self.current_ring_max_r - self.robot_sensor_range // 2 : leg_just_finished = True
        elif self._ring_nav_current_dir_idx == 2: # Moving Left
            if c <= self.current_ring_min_c + self.robot_sensor_range // 2 : leg_just_finished = True
        elif self._ring_nav_current_dir_idx == 3: # Moving Up
            if r <= self.current_ring_min_r + self.robot_sensor_range // 2 : leg_just_finished = True

        if leg_just_finished:
            self._ring_nav_current_dir_idx = (self._ring_nav_current_dir_idx + 1) % 4
            self._ring_nav_legs_completed_this_ring += 1
            self._current_target_on_ring_edge = None 
            if self._ring_nav_legs_completed_this_ring >= 4:
                self.current_phase_complete = True 
                return self.plan_next_action(robot_pos, known_map, **kwargs) 


        target_frontier, path_to_target = self._get_target_for_current_ring_leg(robot_pos, known_map)

        if target_frontier:
            self._last_chosen_frontier = target_frontier
            return target_frontier, path_to_target
        else:
            self._ring_nav_current_dir_idx = (self._ring_nav_current_dir_idx + 1) % 4
            self._ring_nav_legs_completed_this_ring += 1
            self._current_target_on_ring_edge = None
            if self._ring_nav_legs_completed_this_ring >= 8: 
                self.current_phase_complete = True
                return self.plan_next_action(robot_pos, known_map, **kwargs)
            
            target_frontier, path_to_target = self._get_target_for_current_ring_leg(robot_pos, known_map)
            if target_frontier:
                self._last_chosen_frontier = target_frontier
                return target_frontier, path_to_target
            else:
                return self._final_fallback_plan(robot_pos, known_map)