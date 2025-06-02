# robot.py
import numpy as np
from environment import FREE, OBSTACLE, UNKNOWN

class Robot:
    def __init__(self, start_pos, sensor_range=5):
        self.pos = np.array(start_pos) # [row, col]
        self.sensor_range = sensor_range # 方形视野的半边长
        self.path_taken = [list(start_pos)] # 记录机器人走过的路径

    def sense(self, environment):
        """机器人感知周围环境并更新已知地图"""
        r_rob, c_rob = self.pos
        newly_discovered_frontiers = []
        
        # 模拟方形视野
        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                # 检查是否在圆形视野内 (可选，当前为方形)
                # if dr**2 + dc**2 > self.sensor_range**2:
                #     continue

                r, c = r_rob + dr, c_rob + dc
                
                if environment.is_within_bounds(r, c):
                    current_known_state = environment.grid_known[r, c]
                    true_state = environment.get_true_map_state(r, c)
                    environment.update_known_map(r, c, true_state)
                    
                    # 如果一个未知单元格被发现是FREE，并且它之前是边界，现在可能不再是
                    # 但我们主要关心新产生的边界
                    # (边界点检测在planner中进行)


    def move_to(self, target_pos):
        """移动到目标位置 (简化：直接移动)"""
        # 在实际系统中，这里会有路径规划和运动控制
        self.pos = np.array(target_pos)
        self.path_taken.append(list(self.pos))

    def get_position(self):
        return tuple(self.pos)

if __name__ == '__main__':
    from environment import Environment
    env = Environment(30,20)
    robot_start_pos = (env.height // 2, env.width // 2)
    robot = Robot(robot_start_pos, sensor_range=3)
    
    print(f"Robot initial pos: {robot.get_position()}")
    robot.sense(env)
    print("\nKnown Grid (After 1st sense):")
    print(env.grid_known)
    print(f"Explored area: {env.get_explored_area()}/{env.get_total_explorable_area()}")