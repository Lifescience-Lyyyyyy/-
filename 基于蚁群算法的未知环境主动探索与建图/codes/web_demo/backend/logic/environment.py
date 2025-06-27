import numpy as np

FREE, OBSTACLE = 1, 2

class Environment:
    def __init__(self, width, height, start_pos, goal_pos, obstacle_percentage=0.2, map_type="random"):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.true_grid = np.full((height, width), FREE)

        if map_type == "random":
            self._generate_obstacles_random(obstacle_percentage)
        elif map_type == "deceptive_hallway":
            self._generate_deceptive_hallway_map()
        
        self.true_grid[self.start_pos] = FREE
        if self.goal_pos:
            self.true_grid[self.goal_pos] = FREE

    def _generate_obstacles_random(self, percentage):
        num_obstacles = int(self.width * self.height * percentage)
        for _ in range(num_obstacles):
            r, c = np.random.randint(0, self.height), np.random.randint(0, self.width)
            if (r, c) != self.start_pos and (r, c) != self.goal_pos:
                self.true_grid[r, c] = OBSTACLE

    def _generate_deceptive_hallway_map(self):
        self.true_grid[:, :] = OBSTACLE
        start_r, start_c = self.start_pos
        room_half_size = 2
        min_r, max_r = max(0, start_r - room_half_size), min(self.height, start_r + room_half_size + 1)
        min_c, max_c = max(0, start_c - room_half_size), min(self.width, start_c + room_half_size + 1)
        self.true_grid[min_r:max_r, min_c:max_c] = FREE
        dead_end_len = self.width // 6
        if start_c + dead_end_len + 2 < self.width:
            self.true_grid[start_r, start_c + room_half_size + 1: start_c + room_half_size + 1 + dead_end_len] = FREE
            self.true_grid[start_r-1 : start_r+2, start_c + room_half_size + dead_end_len : start_c + room_half_size + dead_end_len + 2] = FREE
            self.true_grid[:, start_c + room_half_size + dead_end_len + 2] = OBSTACLE
        hall_len_down = self.height // 3
        path_c1 = start_c
        if start_r + hall_len_down < self.height:
            self.true_grid[start_r + room_half_size + 1: start_r + hall_len_down, path_c1] = FREE
        turn_r = start_r + hall_len_down
        if turn_r < self.height:
            hall_len_right = self.width // 2 + self.width // 4
            max_c_hall_right = min(self.width, path_c1 + hall_len_right + 1)
            self.true_grid[turn_r, path_c1: max_c_hall_right] = FREE
            open_area_r_start = max(1, turn_r - self.height // 5)
            open_area_r_end = min(self.height - 2, turn_r + self.height // 5)
            open_area_c_start = max(1, max_c_hall_right - 1)
            open_area_c_end = self.width - 2
            if open_area_r_start < open_area_r_end and open_area_c_start < open_area_c_end:
                self.true_grid[open_area_r_start:open_area_r_end, open_area_c_start:open_area_c_end] = FREE
                self.goal_pos = (np.random.randint(open_area_r_start, open_area_r_end), np.random.randint(open_area_c_start, open_area_c_end))
        self.true_grid[0, :] = OBSTACLE; self.true_grid[-1, :] = OBSTACLE
        self.true_grid[:, 0] = OBSTACLE; self.true_grid[:, -1] = OBSTACLE