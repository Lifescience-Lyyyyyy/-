import numpy as np
import random
from .base_planner import BasePlanner
from environment import UNKNOWN, FREE, OBSTACLE
from .geometry_utils import check_line_of_sight # If IG calc uses LoS

class StructuredSpiralPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.height = self.environment.height
        self.width = self.environment.width

        # --- Exploration Phases ---
        self.PHASE_INITIAL_BOXING = 1
        self.PHASE_SPIRAL_INWARD = 2
        self.current_phase = self.PHASE_INITIAL_BOXING

        # --- State for Initial Boxing Phase ---
        # Directions: 0:Down, 1:Right, 2:Up, 3:Left
        self._boxing_directions = [(1, 0), (0, 1), (-1, 0), (0, -1)] 
        self._boxing_current_dir_idx = 0
        # Target coordinates for the end of each leg of the boxing pattern
        # These are inner boundaries considering sensor range
        self._boxing_leg_targets = [
            self.height - 1 - self.robot_sensor_range, # Bottom edge
            self.width - 1 - self.robot_sensor_range,  # Right edge
            self.robot_sensor_range,                   # Top edge
            self.robot_sensor_range                    # Left edge
        ]
        self._boxing_legs_completed = 0 # How many of the 4 legs are done

        # --- State for Spiral Inward Phase ---
        self._spiral_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U for inward spiral
        self._spiral_current_dir_idx = 0
        self._spiral_steps_taken_in_leg = 0
        self._spiral_current_leg_length = 1
        self._spiral_legs_done_at_current_length = 0 # After 2 legs, increase length

        self._last_chosen_target = None # To avoid repeatedly choosing the same stuck target

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

        # Normalize target direction vector (not strictly necessary for dot product ranking)
        # norm_td = np.linalg.norm(target_direction_vector)
        # unit_td_vec = target_direction_vector / norm_td if norm_td > 0 else (0,0)

        for fr_pos in frontiers_list:
            if fr_pos == self._last_chosen_target: # Avoid re-picking same failing target immediately
                continue

            # Vector from robot to frontier
            vec_to_fr = (fr_pos[0] - robot_pos[0], fr_pos[1] - robot_pos[1])
            
            # Dot product to check alignment with target_direction_vector
            # We want frontiers that are "ahead" in the desired direction
            dot_product = vec_to_fr[0] * target_direction_vector[0] + vec_to_fr[1] * target_direction_vector[1]

            if dot_product < 0.1 : # Frontier is not significantly in the desired direction (or is behind)
                continue

            path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
            if path:
                path_cost = len(path) - 1
                if path_cost <= 0: path_cost = 1e-5

                ig = self._calculate_ig_simple(fr_pos[0], fr_pos[1], known_map) # Use simple IG for speed
                
                # Utility: IG combined with alignment and inversely with cost
                # Higher dot_product means better alignment
                utility = (ig + 0.1) * (dot_product + 0.1) / path_cost 
                                
                if utility > best_utility:
                    best_utility = utility
                    chosen_frontier = fr_pos
                    chosen_path = path
        
        return chosen_frontier, chosen_path


    def plan_next_action(self, robot_pos, known_map, **kwargs):
        all_frontiers = self.find_frontiers(known_map)
        if not all_frontiers:
            # print("StructuredSpiralPlanner: No frontiers found.")
            return None, None

        target_frontier = None
        path_to_target = None

        # --- Phase 1: Initial Boxing ---
        if self.current_phase == self.PHASE_INITIAL_BOXING:
            # print(f"BOXING phase, dir_idx: {self._boxing_current_dir_idx}, legs_done: {self._boxing_legs_completed}")
            current_target_direction = self._boxing_directions[self._boxing_current_dir_idx]
            leg_coord_target = self._boxing_leg_targets[self._boxing_current_dir_idx]

            # Determine if current leg is completed by checking robot's position
            # (More robust: check if a significant portion of the edge has been "seen" or if robot is near the target corner)
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
                # print(f"  Boxing leg {self._boxing_current_dir_idx} completed.")
                self._boxing_current_dir_idx = (self._boxing_current_dir_idx + 1) % 4
                self._boxing_legs_completed += 1
                current_target_direction = self._boxing_directions[self._boxing_current_dir_idx] # Update direction
                if self._boxing_legs_completed >= 4: # All 4 legs of boxing done
                    # print("  Boxing phase complete. Switching to SPIRAL_INWARD.")
                    self.current_phase = self.PHASE_SPIRAL_INWARD
                    # Initialize spiral state (start from a corner or center, here from current pos)
                    self._spiral_current_dir_idx = 0 
                    self._spiral_steps_taken_in_leg = 0
                    self._spiral_current_leg_length = 1 # Start with small spiral legs
                    self._spiral_legs_done_at_current_length = 0
                    # Fall through to SPIRAL_INWARD phase in the same planning cycle if needed

            if self.current_phase == self.PHASE_INITIAL_BOXING: # Still in boxing
                target_frontier, path_to_target = self._find_best_frontier_in_direction(
                    robot_pos, known_map, current_target_direction, all_frontiers
                )
                if target_frontier:
                    self._last_chosen_target = target_frontier
                    return target_frontier, path_to_target
                else: # Could not find a frontier in the boxing direction, try next boxing direction
                    # print(f"  No suitable frontier in boxing direction {current_target_direction}. Trying to turn boxing direction.")
                    self._boxing_current_dir_idx = (self._boxing_current_dir_idx + 1) % 4
                    self._boxing_legs_completed += 1 # Count as a (failed) leg completion
                    if self._boxing_legs_completed >= 8: # Emergency switch if stuck in boxing turns
                         self.current_phase = self.PHASE_SPIRAL_INWARD


        # --- Phase 2: Spiral Inward ---
        if self.current_phase == self.PHASE_SPIRAL_INWARD:
            # print(f"SPIRAL phase, dir_idx: {self._spiral_current_dir_idx}, leg_len: {self._spiral_current_leg_length}, steps_in_leg: {self._spiral_steps_taken_in_leg}")
            
            # Try to find a frontier in the current spiral direction
            current_spiral_direction_vec = self._spiral_directions[self._spiral_current_dir_idx]
            target_frontier, path_to_target = self._find_best_frontier_in_direction(
                robot_pos, known_map, current_spiral_direction_vec, all_frontiers
            )

            if target_frontier:
                self._last_chosen_target = target_frontier
                self._spiral_steps_taken_in_leg += 1 # Assume one step towards it counts
                if self._spiral_steps_taken_in_leg >= self._spiral_current_leg_length:
                    self._spiral_current_dir_idx = (self._spiral_current_dir_idx + 1) % 4
                    self._spiral_steps_taken_in_leg = 0
                    self._spiral_legs_done_at_current_length += 1
                    if self._spiral_legs_done_at_current_length == 2:
                        self._spiral_current_leg_length += 1
                        self._spiral_legs_done_at_current_length = 0
                return target_frontier, path_to_target
            else:
                # If no frontier in current spiral direction, try turning the spiral
                # print(f"  No suitable frontier in spiral direction {current_spiral_direction_vec}. Turning spiral.")
                # This is a simple turn, a more robust spiral would try to "hug" known obstacles
                # or find the closest frontier even if not perfectly in spiral direction.
                for _ in range(len(self._spiral_directions) -1): # Try other 3 directions
                    self._spiral_current_dir_idx = (self._spiral_current_dir_idx + 1) % 4
                    self._spiral_steps_taken_in_leg = 0 # Reset for new direction
                    # Reset leg progress if we are forced to turn before completing a leg pair
                    self._spiral_legs_done_at_current_length = 0 

                    current_spiral_direction_vec = self._spiral_directions[self._spiral_current_dir_idx]
                    target_frontier_turned, path_to_target_turned = self._find_best_frontier_in_direction(
                        robot_pos, known_map, current_spiral_direction_vec, all_frontiers
                    )
                    if target_frontier_turned:
                        self._last_chosen_target = target_frontier_turned
                        self._spiral_steps_taken_in_leg += 1 # Count this step
                        # (Leg completion logic for turns can be complex, keeping it simple here)
                        return target_frontier_turned, path_to_target_turned
        
        # Fallback if all phases/logic fail to find a target
        # print("StructuredSpiralPlanner: All strategies failed. Using final fallback.")
        self._last_chosen_target = None # Reset last chosen target
        return self._final_fallback_plan(robot_pos, known_map)