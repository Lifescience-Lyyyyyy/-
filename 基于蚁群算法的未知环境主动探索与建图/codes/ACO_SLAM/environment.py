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

    def _generate_four_rooms_map(self):
        """Generates a map with a simple four-room layout connected by a central area or corridors."""
        self.grid_true[:, :] = OBSTACLE  # Start with all obstacles

        h, w = self.height, self.width
        mid_r, mid_c = h // 2, w // 2
        door_width = max(1, min(3, h // 10, w // 10)) # Width of doorways

        # Define room approximate boundaries (can be adjusted for more/less symmetry)
        # Room 1: Top-Left
        self.grid_true[1 : mid_r-1, 1 : mid_c-1] = UNKNOWN
        # Room 2: Top-Right
        self.grid_true[1 : mid_r-1, mid_c+2 : w-1] = UNKNOWN
        # Room 3: Bottom-Left
        self.grid_true[mid_r+2 : h-1, 1 : mid_c-1] = UNKNOWN
        # Room 4: Bottom-Right
        self.grid_true[mid_r+2 : h-1, mid_c+2 : w-1] = UNKNOWN

        # Create openings (doors) between rooms or to a central corridor
        # Door between Room 1 and a central vertical corridor
        if mid_r-1 > 1 + door_width // 2: # Ensure space for door
            self.grid_true[max(1, mid_r - 1 - door_width // 2) : mid_r-1, mid_c-1 : mid_c+1] = UNKNOWN 
        # Door between Room 2 and a central vertical corridor
        if mid_r-1 > 1 + door_width // 2:
            self.grid_true[max(1, mid_r - 1 - door_width // 2) : mid_r-1, mid_c : mid_c+2] = UNKNOWN
        # Door between Room 3 and a central vertical corridor
        if h-1 > mid_r+2 + door_width //2:
            self.grid_true[mid_r+2 : min(h-1, mid_r+2+door_width//2+1), mid_c-1 : mid_c+1] = UNKNOWN
        # Door between Room 4 and a central vertical corridor
        if h-1 > mid_r+2 + door_width //2:
            self.grid_true[mid_r+2 : min(h-1, mid_r+2+door_width//2+1), mid_c : mid_c+2] = UNKNOWN

        # Central Corridors (optional, can also just have rooms open to a central square)
        # Vertical central corridor
        self.grid_true[1:h-1, mid_c-1:mid_c+2] = UNKNOWN 
        # Horizontal central corridor (can make a '+' shape)
        self.grid_true[mid_r-1:mid_r+2, 1:w-1] = UNKNOWN

        # Ensure map boundaries are obstacles
        self.grid_true[0, :] = OBSTACLE; self.grid_true[h-1, :] = OBSTACLE
        self.grid_true[:, 0] = OBSTACLE; self.grid_true[:, w-1] = OBSTACLE
        
        # Ensure robot start position is valid (make it FREE)
        # If no specific start, place it in one of the rooms, e.g., top-left
        if self.robot_start_pos_ref:
            start_r, start_c = self.robot_start_pos_ref
        else: # Default start if not provided
            start_r, start_c = h // 4, w // 4 
            self.robot_start_pos_ref = (start_r, start_c)

        if self.is_within_bounds(start_r, start_c):
             self.grid_true[start_r, start_c] = UNKNOWN # Will become FREE
        else: # Fallback if provided start is out of bounds for some reason
            self.grid_true[h // 4, w // 4] = UNKNOWN
            self.robot_start_pos_ref = (h // 4, w // 4)


    def _generate_office_layout_map(self):
        """Generates a more complex map resembling an office with rooms and corridors."""
        self.grid_true[:, :] = OBSTACLE # Start with all obstacles
        h, w = self.height, self.width

        # --- Helper to carve out rectangular rooms/corridors ---
        def carve_rectangle(r_start, c_start, r_end, c_end, fill_val=UNKNOWN):
            r_min, r_max = min(r_start, r_end), max(r_start, r_end)
            c_min, c_max = min(c_start, c_end), max(c_start, c_end)
            self.grid_true[max(0,r_min):min(h,r_max+1), max(0,c_min):min(w,c_max+1)] = fill_val
        
        def create_door(r, c, is_horizontal_door=True, door_size=1):
            if is_horizontal_door: # Door in a vertical wall
                carve_rectangle(r, c - door_size // 2, r, c + door_size // 2, UNKNOWN)
            else: # Door in a horizontal wall
                carve_rectangle(r - door_size // 2, c, r + door_size // 2, c, UNKNOWN)

        # --- Main Corridor ---
        corridor_width = max(2, min(4, h // 10))
        carve_rectangle(h // 2 - corridor_width // 2, 1, 
                        h // 2 + corridor_width // 2, w - 2, UNKNOWN)

        # --- Rooms on one side of the corridor ---
        num_rooms_side1 = 3
        room_height1 = h // 2 - corridor_width // 2 - 2 # Available height for rooms
        room_w1 = (w - 2 - (num_rooms_side1 -1)*1) // num_rooms_side1 # *1 for wall between rooms
        
        for i in range(num_rooms_side1):
            rs1 = 1
            re1 = rs1 + room_height1 -1 
            cs1 = 1 + i * (room_w1 + 1)
            ce1 = cs1 + room_w1 -1
            if cs1 < ce1 and rs1 < re1 : # Ensure valid room dimensions
                carve_rectangle(rs1, cs1, re1, ce1, UNKNOWN)
                # Door to corridor
                if h // 2 - corridor_width // 2 -1 >= rs1: # Ensure wall exists
                    create_door(h // 2 - corridor_width // 2 -1, cs1 + room_w1 // 2, is_horizontal_door=False)
        
        # --- Rooms on the other side ---
        num_rooms_side2 = 2
        room_height2 = h - 1 - (h // 2 + corridor_width // 2 + 1) -1
        room_w2 = (w - 2 - (num_rooms_side2 -1)*1) // num_rooms_side2
        
        for i in range(num_rooms_side2):
            rs2 = h // 2 + corridor_width // 2 + 2
            re2 = rs2 + room_height2 -1
            cs2 = 1 + i * (room_w2 + 1)
            ce2 = cs2 + room_w2 -1
            if cs2 < ce2 and rs2 < re2 :
                carve_rectangle(rs2, cs2, re2, ce2, UNKNOWN)
                if rs2-1 <= h-2: # Ensure wall exists
                    create_door(rs2-1, cs2 + room_w2 // 2, is_horizontal_door=False)
        
        # --- Optional: Add a larger common area or cross-corridor ---
        if w > 20 and h > 20:
            cross_cor_r_start = h//4
            cross_cor_r_end = h*3//4
            cross_cor_c = w*3//4
            carve_rectangle(cross_cor_r_start, cross_cor_c - corridor_width//2,
                            cross_cor_r_end, cross_cor_c + corridor_width//2, UNKNOWN)
            # Connect it to main corridor
            if h // 2 >= cross_cor_r_start and h // 2 <= cross_cor_r_end:
                 carve_rectangle(h // 2 - corridor_width // 2, cross_cor_c - corridor_width//2,
                                h // 2 + corridor_width // 2, cross_cor_c + corridor_width//2, UNKNOWN)


        # --- Add some random obstacles within rooms/corridors to make it more complex ---
        # (Careful not to block all paths or the start position)
        if self.robot_start_pos_ref:
            r_rob, c_rob = self.robot_start_pos_ref
        else: # Fallback if no ref, less ideal for this map type
            r_rob,c_rob = h // 2, w // 10 
            self.robot_start_pos_ref = (r_rob, c_rob)

        num_internal_obstacles = int(w * h * 0.03) # 3% internal obstacles
        for _ in range(num_internal_obstacles):
            attempts = 0
            while attempts < 50:
                r_obs, c_obs = np.random.randint(1,h-1), np.random.randint(1,w-1)
                # Try to place only in existing UNKNOWN (soon to be FREE) areas
                # And avoid robot start and its immediate vicinity
                if self.grid_true[r_obs,c_obs] == UNKNOWN and not (abs(r_obs - r_rob) <=2 and abs(c_obs - c_rob) <=2):
                    self.grid_true[r_obs,c_obs] = OBSTACLE
                    break
                attempts +=1

        # Ensure map boundaries are obstacles
        self.grid_true[0, :] = OBSTACLE; self.grid_true[h-1, :] = OBSTACLE
        self.grid_true[:, 0] = OBSTACLE; self.grid_true[:, w-1] = OBSTACLE
        
        # Set robot start (e.g., in the main corridor or a specific room)
        # This overrides any obstacle that might have been placed by the general room carving
        # A good start for this layout might be near one end of the main corridor
        if self.robot_start_pos_ref:
            start_r, start_c = self.robot_start_pos_ref
            if not self.is_within_bounds(start_r, start_c) or self.grid_true[start_r,start_c] == OBSTACLE:
                # If pre-defined start is bad, find a new one.
                start_r, start_c = h // 2, max(1, corridor_width) # In main corridor
                self.robot_start_pos_ref = (start_r, start_c)
        else: # Should have been set earlier
            start_r, start_c = h // 2, max(1, corridor_width)
            self.robot_start_pos_ref = (start_r, start_c)

        self.grid_true[start_r, start_c] = UNKNOWN # Ensure start is clear (will become FREE)


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

    def print_map(grid_map):
        for row in grid_map:
            print("".join(map(str, row)).replace(str(UNKNOWN),'_').replace(str(FREE),'.').replace(str(OBSTACLE),'#'))

    print("--- Random Map Test ---")
    env_random = Environment(30, 20, 0.25, map_type="random", robot_start_pos_ref=(10,15))
    print_map(env_random.grid_true)
    print(f"Random Total explorable: {env_random.get_total_explorable_area()}")

    print("\n--- Deceptive Hallway Map Test ---")
    env_deceptive = Environment(40, 30, map_type="deceptive_hallway", robot_start_pos_ref=(15,5))
    print_map(env_deceptive.grid_true)
    print(f"Deceptive Total explorable: {env_deceptive.get_total_explorable_area()}")

    print("\n--- Four Rooms Map Test ---")
    env_four_rooms = Environment(30, 30, map_type="four_rooms", robot_start_pos_ref=(5,5))
    print_map(env_four_rooms.grid_true)
    print(f"Four Rooms Total explorable: {env_four_rooms.get_total_explorable_area()}")

    print("\n--- Office Layout Map Test ---")
    env_office = Environment(50, 40, map_type="office_layout", robot_start_pos_ref=(20,3)) # Start in main corridor
    print_map(env_office.grid_true)
    print(f"Office Layout Total explorable: {env_office.get_total_explorable_area()}")