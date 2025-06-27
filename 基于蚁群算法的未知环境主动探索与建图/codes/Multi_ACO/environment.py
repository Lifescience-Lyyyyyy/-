# environment.py
import numpy as np

# 地图单元格状态
UNKNOWN = 0
FREE = 1
OBSTACLE = 2

class Environment:
    def __init__(self, width, height, obstacle_percentage=0.2, map_type="random", robot_start_pos_ref=None):
        self.width = width
        self.height = height
        self.grid_true = np.full((height, width), UNKNOWN)
        self.grid_known = np.full((height, width), UNKNOWN)
        
        if robot_start_pos_ref:
            self.robot_start_positions = robot_start_pos_ref if isinstance(robot_start_pos_ref, list) else [robot_start_pos_ref]
        else:
            self.robot_start_positions = [(height // 2, width // 2)]

        if map_type == "random":
            self._generate_obstacles_random(obstacle_percentage)
        elif map_type == "deceptive_hallway":
            self._generate_deceptive_hallway_map()
        else:
            print(f"Warning: Unknown map_type '{map_type}'. Defaulting to random.")
            self._generate_obstacles_random(obstacle_percentage) 
        
        self.grid_true[self.grid_true == UNKNOWN] = FREE
        
        for start_pos in self.robot_start_positions:
            r_start, c_start = start_pos
            if self.is_within_bounds(r_start, c_start):
                self.grid_true[r_start, c_start] = FREE
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r_start + dr, c_start + dc
                        if self.is_within_bounds(nr, nc) and self.grid_true[nr,nc] == OBSTACLE and (abs(dr)+abs(dc)<=1):
                            self.grid_true[nr,nc] = FREE

    def _generate_obstacles_random(self, percentage):
        num_obstacles = int(self.width * self.height * percentage)
        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 100:
                r, c = np.random.randint(0, self.height), np.random.randint(0, self.width)
                is_near_start = False
                for start_pos in self.robot_start_positions:
                    if abs(r - start_pos[0]) <= 1 and abs(c - start_pos[1]) <= 1:
                        is_near_start = True
                        break
                if is_near_start:
                    attempts += 1
                    continue
                if self.grid_true[r, c] == UNKNOWN:
                    self.grid_true[r, c] = OBSTACLE
                    break
                attempts += 1
            if attempts >= 100:
                print("Warning: Could not place all random obstacles.")

    def _generate_deceptive_hallway_map(self):
        start_r, start_c = self.robot_start_positions[0]
        self.grid_true[:, :] = OBSTACLE
        room_half_size = 2
        min_r, max_r = max(0, start_r - room_half_size), min(self.height, start_r + room_half_size + 1)
        min_c, max_c = max(0, start_c - room_half_size), min(self.width, start_c + room_half_size + 1)
        self.grid_true[min_r:max_r, min_c:max_c] = UNKNOWN
        dead_end_len = self.width // 6
        if start_c + dead_end_len + 1 < self.width:
            self.grid_true[start_r, start_c + room_half_size + 1 : start_c + room_half_size + 1 + dead_end_len] = UNKNOWN
            self.grid_true[start_r-1 : start_r+2, start_c + room_half_size + dead_end_len : start_c + room_half_size + dead_end_len + 2] = UNKNOWN
            if start_c + room_half_size + dead_end_len + 2 < self.width:
                 self.grid_true[:, start_c + room_half_size + dead_end_len + 2] = OBSTACLE
        hall_len_down = self.height // 3
        path_c1 = start_c
        if start_r + hall_len_down < self.height:
            self.grid_true[start_r + room_half_size + 1 : start_r + room_half_size + 1 + hall_len_down, path_c1] = UNKNOWN
        turn_r = start_r + room_half_size + hall_len_down
        if turn_r < self.height:
            hall_len_right = self.width // 2 + self.width // 4
            max_c_hall_right = min(self.width, path_c1 + hall_len_right + 1)
            self.grid_true[turn_r, path_c1 : max_c_hall_right] = UNKNOWN
            open_area_r_start = max(1, turn_r - self.height // 5)
            open_area_r_end = min(self.height -1, turn_r + self.height // 5 + 1)
            open_area_c_start = max(1, max_c_hall_right - 1)
            open_area_c_end = self.width - 1
            if open_area_r_start < open_area_r_end and open_area_c_start < open_area_c_end:
                self.grid_true[open_area_r_start:open_area_r_end, open_area_c_start:open_area_c_end] = UNKNOWN
        self.grid_true[0, :], self.grid_true[-1, :] = OBSTACLE, OBSTACLE
        self.grid_true[:, 0], self.grid_true[:, -1] = OBSTACLE, OBSTACLE
        self.grid_true[start_r, start_c] = UNKNOWN

    def get_true_map_state(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid_true[r, c]
        return OBSTACLE

    def update_known_map(self, r, c, value):
        if 0 <= r < self.height and 0 <= c < self.width:
            self.grid_known[r, c] = value

    def is_within_bounds(self, r, c):
        return 0 <= r < self.height and 0 <= c < self.width

    def get_total_explorable_area(self):
        """返回真实地图中所有 FREE 单元格的总数。这个方法是正确的。"""
        return np.sum(self.grid_true == FREE)

    def get_explored_area(self):
        """(已修复) 返回机器人已知地图中所有 FREE 单元格的总数。"""
        return np.sum(self.grid_known == FREE)
    
    def get_known_map_for_planner(self):
        return np.copy(self.grid_known)