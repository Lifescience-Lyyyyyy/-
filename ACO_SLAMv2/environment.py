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
        self.grid_true = np.full((height, width), UNKNOWN) # 真实地图，初始全未知
        self.grid_known = np.full((height, width), UNKNOWN) # 机器人已知的地图
        self.robot_start_pos_ref = robot_start_pos_ref if robot_start_pos_ref else (height // 2, width // 2)


        if map_type == "random":
            self._generate_obstacles_random(obstacle_percentage)
        elif map_type == "deceptive_hallway":
            self._generate_deceptive_hallway_map()
        else:
            print(f"Warning: Unknown map_type '{map_type}'. Defaulting to random.")
            self._generate_obstacles_random(obstacle_percentage) 
        
        # 将非障碍物区域设为FREE (在生成障碍物之后)
        self.grid_true[self.grid_true == UNKNOWN] = FREE
        
        # 确保机器人起始点在真实地图中是FREE的 (如果之前是OBSTACLE会被覆盖)
        # 这一步很重要，特别是对于手动设计的地图
        if self.robot_start_pos_ref:
             r_start, c_start = self.robot_start_pos_ref
             if self.is_within_bounds(r_start, c_start):
                self.grid_true[r_start, c_start] = FREE
                # 如果周围也是障碍物，尝试清理一小块区域
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        nr, nc = r_start + dr, c_start + dc
                        if self.is_within_bounds(nr, nc) and self.grid_true[nr,nc] == OBSTACLE and (abs(dr)+abs(dc)<=1) : # Only adjacent
                             self.grid_true[nr,nc] = FREE # Make adjacent free if it was obstacle


    def _generate_obstacles_random(self, percentage):
        # 简单生成一些随机障碍物
        num_obstacles = int(self.width * self.height * percentage)
        # 获取机器人起始位置，避免在此处生成障碍物
        r_rob, c_rob = self.robot_start_pos_ref

        for _ in range(num_obstacles):
            attempts = 0
            while attempts < 100: # Max attempts to find a valid spot
                r, c = np.random.randint(0, self.height), np.random.randint(0, self.width)
                # 避免在机器人起始点及其紧邻位置生成障碍物
                if abs(r - r_rob) <= 1 and abs(c - c_rob) <= 1:
                    attempts += 1
                    continue
                if self.grid_true[r, c] == UNKNOWN: # Check against UNKNOWN before it's set to FREE later
                    self.grid_true[r, c] = OBSTACLE
                    break
                attempts += 1
            if attempts >=100:
                print("Warning: Could not place all random obstacles due to constraints.")


    def _generate_deceptive_hallway_map(self):
        # 设计一个具有欺骗性的地图：一个短的、明显的死胡同靠近起点，
        # 以及一个更长、可能不那么明显但通向大片区域的路径。
        self.grid_true[:, :] = OBSTACLE  # Start with all obstacles

        # 机器人起始区域 (确保与main_simulation.py中的ROBOT_START_POS一致或传入)
        start_r, start_c = self.robot_start_pos_ref
        
        # 创建一个小的起始房间
        room_half_size = 2
        min_r, max_r = max(0, start_r - room_half_size), min(self.height, start_r + room_half_size + 1)
        min_c, max_c = max(0, start_c - room_half_size), min(self.width, start_c + room_half_size + 1)
        self.grid_true[min_r:max_r, min_c:max_c] = UNKNOWN


        # 1. 短的死胡同 (例如，向右)
        dead_end_len = self.width // 6
        if start_c + dead_end_len + 1 < self.width :
            # Hallway
            self.grid_true[start_r, start_c + room_half_size + 1 : start_c + room_half_size + 1 + dead_end_len] = UNKNOWN
            # Small room at the end, but blocked further
            self.grid_true[start_r-1 : start_r+2, start_c + room_half_size + dead_end_len : start_c + room_half_size + dead_end_len + 2] = UNKNOWN
            # Block it
            if start_c + room_half_size + dead_end_len + 2 < self.width:
                 self.grid_true[:, start_c + room_half_size + dead_end_len + 2] = OBSTACLE


        # 2. 长的、通往大区域的路径 (例如，向下，然后向右)
        # Hallway downwards
        hall_len_down = self.height // 3
        path_c1 = start_c # Maintain column
        if start_r + hall_len_down < self.height:
            self.grid_true[start_r + room_half_size + 1 : start_r + room_half_size + 1 + hall_len_down, path_c1] = UNKNOWN
        
        # Turn right
        turn_r = start_r + room_half_size + hall_len_down
        if turn_r < self.height: # Ensure turn_r is within bounds
            hall_len_right = self.width // 2 + self.width // 4 # Make it long
            max_c_hall_right = min(self.width, path_c1 + hall_len_right +1)
            self.grid_true[turn_r, path_c1 : max_c_hall_right] = UNKNOWN

            # Large open area at the end
            open_area_r_start = max(1, turn_r - self.height // 5)
            open_area_r_end = min(self.height -1, turn_r + self.height // 5 +1)
            open_area_c_start = max(1, max_c_hall_right -1) # Start from end of hallway
            open_area_c_end = self.width -1 # To the edge

            if open_area_r_start < open_area_r_end and open_area_c_start < open_area_c_end:
                self.grid_true[open_area_r_start:open_area_r_end, open_area_c_start:open_area_c_end] = UNKNOWN


        # Make sure map boundaries are obstacles
        self.grid_true[0, :] = OBSTACLE
        self.grid_true[-1, :] = OBSTACLE
        self.grid_true[:, 0] = OBSTACLE
        self.grid_true[:, -1] = OBSTACLE
        
        # Crucially, ensure the robot's specific start cell is UNKNOWN (will become FREE)
        # This overrides any obstacle that might have been placed there by broad strokes
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
        return np.sum(self.grid_true == FREE)

    def get_explored_area(self):
        return np.sum(self.grid_known == FREE)

    def get_known_map_for_planner(self):
        return np.copy(self.grid_known)

if __name__ == '__main__':
    # Test random map
    env_random = Environment(30, 20, 0.25, map_type="random", robot_start_pos_ref=(10,15))
    print("Random True Grid:")
    print(env_random.grid_true)
    print(f"Random Total explorable: {env_random.get_total_explorable_area()}")

    # Test deceptive map
    env_deceptive = Environment(50, 40, map_type="deceptive_hallway", robot_start_pos_ref=(20,5)) # Example start
    print("\nDeceptive True Grid:")
    # For better visualization of larger maps in console, you might need to adjust print options or save to file
    # np.set_printoptions(linewidth=np.inf) # Might help for wider console output
    for row in env_deceptive.grid_true:
        print("".join(map(str, row)).replace('0','_').replace('1','.').replace('2','#')) # Simple char viz
    print(f"Deceptive Total explorable: {env_deceptive.get_total_explorable_area()}")