# planners/base_planner.py
from abc import ABC, abstractmethod
import numpy as np
from environment import UNKNOWN, FREE

class BasePlanner(ABC):
    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def plan_next_action(self, robot_pos, known_map):
        """
        根据当前机器人位置和已知地图，规划下一个目标点。
        返回目标点 (r, c) 或 None (如果无法规划)。
        """
        pass

    def find_frontiers(self, known_map, robot_pos_for_reachability=None):
        """
        找到已知区域和未知区域之间的边界点。
        边界点是已知为FREE且至少有一个邻居是UNKNOWN的单元格。
        """
        frontiers = []
        height, width = known_map.shape
        for r in range(height):
            for c in range(width):
                if known_map[r, c] == FREE: # 必须是已知的空闲区域
                    is_frontier = False
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: # 四邻域
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if known_map[nr, nc] == UNKNOWN:
                                is_frontier = True
                                break
                        # 如果机器人位置已知，我们也可以考虑地图边界外的未知区域
                        # elif robot_pos_for_reachability is not None and known_map[nr,nc] != OBSTACLE: 
                        # (简化：只考虑地图内的未知)
                    if is_frontier:
                        frontiers.append((r, c))
        return frontiers

    def _is_reachable(self, start_pos, end_pos, known_map):
        """
        (简化) 检查两点之间是否直线可达 (没有障碍物)。
        在实际应用中，这里会用A*等算法。
        """
        # Bresenham's line algorithm or simple check for obstacles
        # For simplicity, assume direct line of sight if no obstacles on the line
        # This is a very naive reachability check.
        r0, c0 = start_pos
        r1, c1 = end_pos
        
        # Check direct path (very simplified)
        # A proper implementation would use A* on the known_map
        # For now, we assume if the target is a frontier, it's "reachable" conceptually
        # and the robot.move_to will handle it.
        # Let's just ensure the end_pos itself is not an obstacle
        if known_map[end_pos[0], end_pos[1]] == FREE: # OBSTACLE is already filtered by frontier def
             return True # Simplified
        return False