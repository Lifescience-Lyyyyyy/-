import numpy as np
from .base_planner import BasePlanner
from environment import UNKNOWN, FREE, OBSTACLE
# from .geometry_utils import check_line_of_sight # If IG calculation needs LoS

class RingExplorationPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.height = self.environment.height
        self.width = self.environment.width

        # --- Ring Exploration State ---
        self.current_ring_level = 0 # 0 is the outermost ring
        # Ring width can be based on sensor range, e.g., sensor_range or a bit more
        # For simplicity, let's define a ring by its min/max row/col based on current_ring_level
        # Ring thickness can be dynamic or fixed. Let's use a fixed conceptual thickness related to sensor range.
        # Here, we'll define the "active zone" of the ring.
        
        self.ring_step_size = self.robot_sensor_range + 2 # How much we shrink the boundaries for the next ring

        # State for navigating along the current ring's "edges"
        # Directions: 0:Right (along top_r), 1:Down (along right_c), 
        #             2:Left (along bottom_r), 3:Up (along left_c)
        self._ring_nav_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        self._ring_nav_current_dir_idx = 0
        self._ring_nav_legs_completed_this_ring = 0

        self._current_target_on_ring_edge = None # The specific (r,c) target on the current ring edge
        self._last_chosen_frontier = None # To avoid re-picking same stuck frontier

        self._initialize_ring_boundaries()

    def _initialize_ring_boundaries(self):
        """Calculates the boundaries for the current ring based on self.current_ring_level."""
        # Effective offset from the map border for the current ring
        # Outermost ring (level 0) is close to map edges minus sensor range
        # Inner rings shrink inwards
        offset = self.current_ring_level * self.ring_step_size + self.robot_sensor_range
        
        # Define the active "inner box" for the current ring's navigation path
        self.current_ring_min_r = min(offset, self.height // 2 -1 if self.height // 2 > offset else offset )
        self.current_ring_max_r = max(self.height - 1 - offset, self.height // 2 )
        self.current_ring_min_c = min(offset, self.width // 2 -1 if self.width // 2 > offset else offset)
        self.current_ring_max_c = max(self.width - 1 - offset, self.width // 2)

        # Ensure min is not greater than max if map is too small
        if self.current_ring_min_r >= self.current_ring_max_r or \
           self.current_ring_min_c >= self.current_ring_max_c:
            # print(f"Ring {self.current_ring_level}: Boundaries collapsed or invalid. Exploration might be complete.")
            self.current_phase_complete = True # Mark as complete if boundaries are bad
        else:
            self.current_phase_complete = False
        
        # Reset navigation state for the new ring
        self._ring_nav_current_dir_idx = 0
        self._ring_nav_legs_completed_this_ring = 0
        self._current_target_on_ring_edge = None # No specific target yet for this new ring leg
        # print(f"Ring Lvl {self.current_ring_level}: Bounds R:[{self.current_ring_min_r}-{self.current_ring_max_r}], C:[{self.current_ring_min_c}-{self.current_ring_max_c}]")


    def _get_target_for_current_ring_leg(self, robot_pos, known_map):
        """
        Determines a target point on the current "edge" of the ring being traversed.
        The robot tries to move towards this point, prioritizing frontiers in that direction.
        """
        # Define the 4 corner points of the current ring's navigation path
        # These are conceptual corners robot tries to move between along the ring's "inner boundary"
        corners = [
            (self.current_ring_min_r, self.current_ring_max_c), # Top-right of inner box
            (self.current_ring_max_r, self.current_ring_max_c), # Bottom-right
            (self.current_ring_max_r, self.current_ring_min_c), # Bottom-left
            (self.current_ring_min_r, self.current_ring_min_c)  # Top-left
        ]

        # The target is the *next* corner in the sequence based on current direction
        # However, this might be too far. Instead, let's aim for frontiers along the current leg.
        
        # Get current leg's direction and a point far along that leg
        direction_vec = self._ring_nav_directions[self._ring_nav_current_dir_idx]
        
        # Define a "lookahead" point along the current leg's direction
        # This point helps filter frontiers that are generally in the right direction.
        # The actual target will be a frontier.
        
        # Example: If moving Right (0,1) along top edge (current_ring_min_r)
        # from current robot_pos towards current_ring_max_c.
        # If moving Down (1,0) along right edge (current_ring_max_c)
        # from current robot_pos towards current_ring_max_r.

        # We need a simpler way to find a "good" frontier along the current sweeping direction.
        # Let's consider frontiers that are roughly on the current "edge" being swept.
        
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
            # No frontiers directly on the current leg, widen search or use _find_best_frontier_in_direction
            # For now, let's use _find_best_frontier_in_direction with the general sweep direction
            return self._find_best_frontier_in_direction(robot_pos, known_map, direction_vec, all_frontiers)

        # From candidate_frontiers_on_leg, pick the "best" one (e.g., highest IG/cost or just nearest)
        # This is where the "prioritize unvisited" comes in.
        # We can use a similar utility as IGE: IG/Cost.
        best_utility = -float('inf')
        chosen_frontier = None
        chosen_path = None

        for fr_pos_on_leg in candidate_frontiers_on_leg:
            if fr_pos_on_leg == self._last_chosen_frontier: continue

            path = self._is_reachable_and_get_path(robot_pos, fr_pos_on_leg, known_map)
            if path:
                path_cost = len(path) -1 
                if path_cost <= 0: path_cost = 1e-5
                
                # Using simple IG for now, can be _calculate_information_gain_with_los
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

    def _calculate_ig_simple(self, r_prospective, c_prospective, known_map): # Copied for now
        ig = 0
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr, nc = r_prospective + dr, c_prospective + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    ig += 1
        return ig


    def plan_next_action(self, robot_pos, known_map, **kwargs):
        self._last_chosen_frontier = None # Reset before new planning

        if self.current_phase_complete: # Current ring is done, or boundaries collapsed
            self.current_ring_level += 1
            # print(f"RingExploration: Moving to next ring level: {self.current_ring_level}")
            self._initialize_ring_boundaries()
            if self.current_phase_complete: # If new ring is also immediately complete (e.g., map fully explored/small)
                # print("RingExploration: All rings seem complete or map too small. Falling back.")
                return self._final_fallback_plan(robot_pos, known_map)

        # Determine if current leg of the ring traversal is "done"
        # This is tricky. A simple way: if robot is close to the "corner" of this leg.
        # Or, if no more good frontiers in current direction.
        
        # Let's define leg completion based on robot's position relative to ring "corners"
        # This is a simplified check. A more robust way would be to see if the "edge" is explored.
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
            # print(f"  Ring leg {self._ring_nav_current_dir_idx} considered finished.")
            self._ring_nav_current_dir_idx = (self._ring_nav_current_dir_idx + 1) % 4
            self._ring_nav_legs_completed_this_ring += 1
            self._current_target_on_ring_edge = None # Force re-evaluation of target for new leg
            if self._ring_nav_legs_completed_this_ring >= 4: # Completed a full circle around current ring
                self.current_phase_complete = True # Mark current ring exploration as complete
                # print(f"  Ring level {self.current_ring_level} completed. Will move to next ring.")
                # Recursive call to handle next ring or fallback immediately
                return self.plan_next_action(robot_pos, known_map, **kwargs) 


        # Get a target frontier for the current leg of the current ring
        target_frontier, path_to_target = self._get_target_for_current_ring_leg(robot_pos, known_map)

        if target_frontier:
            self._last_chosen_frontier = target_frontier
            return target_frontier, path_to_target
        else:
            # If no target found on current leg, try to advance the leg/ring state
            # This indicates this leg might be fully explored or blocked
            # print(f"  No target found on current ring leg {self._ring_nav_current_dir_idx}. Advancing leg.")
            self._ring_nav_current_dir_idx = (self._ring_nav_current_dir_idx + 1) % 4
            self._ring_nav_legs_completed_this_ring += 1
            self._current_target_on_ring_edge = None
            if self._ring_nav_legs_completed_this_ring >= 8: # Tried all directions twice, still nothing
                # print("  Stuck trying to complete ring legs. Moving to next ring or fallback.")
                self.current_phase_complete = True # Force advance to next ring
                # Recursive call to try next ring or ultimately fallback
                return self.plan_next_action(robot_pos, known_map, **kwargs)
            
            # Try to find a target in the *new* direction immediately
            # print(f"  Trying next leg direction: {self._ring_nav_current_dir_idx}")
            target_frontier, path_to_target = self._get_target_for_current_ring_leg(robot_pos, known_map)
            if target_frontier:
                self._last_chosen_frontier = target_frontier
                return target_frontier, path_to_target
            else: # Still no target after trying to advance, truly fallback
                # print("  Still no target after advancing leg. Using final fallback.")
                return self._final_fallback_plan(robot_pos, known_map)