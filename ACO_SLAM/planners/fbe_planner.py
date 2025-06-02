# planners/fbe_planner.py
from .base_planner import BasePlanner
import numpy as np

class FBEPlanner(BasePlanner):
    def __init__(self, environment):
        super().__init__(environment)

    def plan_next_action(self, robot_pos, known_map):
        frontiers = self.find_frontiers(known_map)
        
        if not frontiers:
            return None # No more frontiers to explore

        # 选择最近的边界点
        robot_r, robot_c = robot_pos
        min_dist = float('inf')
        best_frontier = None

        for fr, fc in frontiers:
            # 使用曼哈顿距离或欧氏距离
            dist = abs(fr - robot_r) + abs(fc - robot_c) # Manhattan distance
            # dist = np.sqrt((fr - robot_r)**2 + (fc - robot_c)**2) # Euclidean
            if dist < min_dist:
                 # 确保目标点本身是可达的 (虽然边界点定义为FREE)
                if self._is_reachable(robot_pos, (fr, fc), known_map): # Simplified reachability
                    min_dist = dist
                    best_frontier = (fr, fc)
        
        return best_frontier

if __name__ == '__main__':
    from environment import Environment, UNKNOWN, FREE, OBSTACLE
    # Test FBE
    env = Environment(10,10, obstacle_percentage=0.1)
    env.grid_known = np.copy(env.grid_true) # Simulate fully known for testing frontiers
    env.grid_known[3:6, 3:6] = UNKNOWN # Create an unknown region
    env.grid_known[4,4] = FREE # Ensure a free cell inside to start finding frontiers around it
    
    planner = FBEPlanner(env)
    robot_pos = (4,4) 
    
    # Manually set some known cells around robot_pos to test frontier finding
    for r_off in range(-1, 2):
        for c_off in range(-1, 2):
            if env.is_within_bounds(robot_pos[0]+r_off, robot_pos[1]+c_off):
                if env.grid_true[robot_pos[0]+r_off, robot_pos[1]+c_off] == FREE:
                     env.grid_known[robot_pos[0]+r_off, robot_pos[1]+c_off] = FREE
                elif env.grid_true[robot_pos[0]+r_off, robot_pos[1]+c_off] == OBSTACLE:
                     env.grid_known[robot_pos[0]+r_off, robot_pos[1]+c_off] = OBSTACLE


    print("Known map for FBE test:")
    print(env.grid_known)
    frontiers = planner.find_frontiers(env.grid_known)
    print("Found frontiers:", frontiers)
    
    next_target = planner.plan_next_action(robot_pos, env.grid_known)
    print("FBE Next Target:", next_target)