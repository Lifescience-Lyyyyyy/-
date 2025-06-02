# environment.py
import numpy as np

# 地图单元格状态
UNKNOWN = 0
FREE = 1
OBSTACLE = 2

class Environment:
    def __init__(self, width, height, obstacle_percentage=0.2):
        self.width = width
        self.height = height
        self.grid_true = np.full((height, width), UNKNOWN) # 真实地图，初始全未知
        self.grid_known = np.full((height, width), UNKNOWN) # 机器人已知的地图
        self._generate_obstacles(obstacle_percentage)
        
        # 初始化时，将机器人起始位置周围设为已知FREE
        # (假设机器人在(height//2, width//2)开始，并能看到一个小的初始区域)
        # 这个初始感知可以在robot.sense()中完成


    def _generate_obstacles(self, percentage):
        # 简单生成一些随机障碍物 (不包括边缘，以确保可探索性)
        num_obstacles = int(self.width * self.height * percentage)
        for _ in range(num_obstacles):
            while True:
                r, c = np.random.randint(1, self.height - 1), np.random.randint(1, self.width - 1)
                if self.grid_true[r, c] == UNKNOWN:
                    self.grid_true[r, c] = OBSTACLE
                    break
        # 将非障碍物区域设为FREE
        self.grid_true[self.grid_true == UNKNOWN] = FREE


    def get_true_map_state(self, r, c):
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.grid_true[r, c]
        return OBSTACLE # 超出边界视为障碍物

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
        # 返回一个副本，防止规划器意外修改
        return np.copy(self.grid_known)

if __name__ == '__main__':
    env = Environment(30, 20)
    print("True Grid:")
    print(env.grid_true)
    print("\nKnown Grid (Initial):")
    print(env.grid_known)
    print(f"Total explorable (FREE in true map): {env.get_total_explorable_area()}")