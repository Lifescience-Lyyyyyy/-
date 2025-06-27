import numpy as np
import random

# Map cell states
UNKNOWN = 0
FREE = 1
OBSTACLE = 2 # Regular obstacles 
LANDMARK = 3 # Landmarks are the only obstacles

class Environment:
    def __init__(self, width, height, 
                 map_type="landmarks_only", # Default to new type
                 robot_start_pos_ref=None, 
                 env_seed=None, 
                 num_landmarks=10):
        
        self.width = width
        self.height = height
        self.grid_true = np.full((height, width), UNKNOWN) # Initially all UNKNOWN
        # Probabilistic map stores log-odds of occupation probability
        self.log_odds_map = np.zeros((height, width), dtype=np.float32)
        self.log_odds_free = -1.38 
        self.log_odds_occ = 1.9   
        
        self.robot_start_pos_ref = robot_start_pos_ref if robot_start_pos_ref else (height // 2, width // 2)

        if env_seed is not None:
            np.random.seed(env_seed)
            random.seed(env_seed)

        # --- Map Generation ---
        if map_type == "landmarks_only":
            # Start with a completely FREE map, then place landmarks
            self.grid_true[:, :] = FREE
            # Place landmarks as the only obstacles
            self._place_landmarks_on_free_map(num_landmarks)
        elif map_type == "random": # Keep this for compatibility
            self._generate_obstacles_random(0.15) # Example percentage
            self._place_landmarks_on_obstacles(num_landmarks)
        else:
            print(f"Warning: Unknown map_type '{map_type}'. Defaulting to 'landmarks_only'.")
            self.grid_true[:, :] = FREE
            self._place_landmarks_on_free_map(num_landmarks)
        
        # Ensure robot start position is FREE
        if self.robot_start_pos_ref:
             r_start, c_start = self.robot_start_pos_ref
             if self.is_within_bounds(r_start, c_start):
                self.grid_true[r_start, c_start] = FREE # Ensure it's not a landmark
                # Clear a small area around the start (less critical in sparse map)
                for dr_clear in range(-1, 2):
                    for dc_clear in range(-1, 2):
                        nr_clear, nc_clear = r_start + dr_clear, c_start + dc_clear
                        if self.is_within_bounds(nr_clear, nc_clear) and \
                           self.grid_true[nr_clear,nc_clear] == LANDMARK:
                             self.grid_true[nr_clear,nc_clear] = FREE

    def _place_landmarks_on_free_map(self, num_landmarks):
        """Places landmarks randomly on a free map, avoiding the robot's start area."""
        r_rob, c_rob = self.robot_start_pos_ref
        placed_count = 0
        attempts = 0
        while placed_count < num_landmarks and attempts < num_landmarks * 100:
            r = np.random.randint(1, self.height - 1)
            c = np.random.randint(1, self.width - 1)
            # Avoid placing too close to start position
            if abs(r - r_rob) <= 5 and abs(c - c_rob) <= 5:
                attempts += 1
                continue
            # Check if the cell is currently FREE to place a landmark
            if self.grid_true[r, c] == FREE:
                self.grid_true[r, c] = LANDMARK
                placed_count += 1
            attempts += 1
        
        if placed_count < num_landmarks:
            print(f"Warning: Could only place {placed_count}/{num_landmarks} landmarks.")

    # These methods are kept for compatibility with other map types if needed
    def _generate_obstacles_random(self, percentage):
        # ... (same as before) ...
        num_obstacles = int(self.width * self.height * percentage)
        r_rob, c_rob = self.robot_start_pos_ref
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 100: 
                r, c = np.random.randint(0, self.height), np.random.randint(0, self.width)
                if abs(r - r_rob) <= 2 and abs(c - c_rob) <= 2:
                    attempts += 1; continue
                if self.grid_true[r, c] == UNKNOWN: 
                    self.grid_true[r, c] = OBSTACLE; break
                attempts += 1

    def _place_landmarks_on_obstacles(self, num_landmarks):
        # ... (same as before) ...
        obstacle_indices = np.argwhere(self.grid_true == OBSTACLE)
        if len(obstacle_indices) < num_landmarks:
            num_landmarks = len(obstacle_indices)
        if num_landmarks > 0:
            landmark_indices_to_replace = np.random.choice(len(obstacle_indices), num_landmarks, replace=False)
            for chosen_idx in landmark_indices_to_replace:
                r, c = obstacle_indices[chosen_idx]
                self.grid_true[r, c] = LANDMARK

    # --- Other methods (get_true_map_state, update_probabilistic_map, etc. remain the same) ---
    def get_true_map_state(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid_true[r, c]
        return OBSTACLE # Treat out of bounds as obstacles

    def update_probabilistic_map(self, r, c, is_obstacle_or_landmark):
        if self.is_within_bounds(r, c):
            if is_obstacle_or_landmark: self.log_odds_map[r, c] += self.log_odds_occ
            else: self.log_odds_map[r, c] += self.log_odds_free
            self.log_odds_map[r,c] = np.clip(self.log_odds_map[r,c], -15, 15)

    def get_occupancy_probability_map(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds_map))

    def get_map_for_planner(self):
        prob_map = self.get_occupancy_probability_map()
        known_map_discrete = np.full((self.height, self.width), UNKNOWN)
        prob_free_thresh = 0.4; prob_occ_thresh = 0.6
        known_map_discrete[prob_map < prob_free_thresh] = FREE
        known_map_discrete[prob_map > prob_occ_thresh] = OBSTACLE
        return known_map_discrete

    def is_within_bounds(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width