from abc import ABC, abstractmethod
import numpy as np
from environment import UNKNOWN, FREE, OBSTACLE
from .pathfinding import a_star_search

class BasePlanner(ABC):
    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def plan_next_action(self, robot_pos, known_map, **kwargs): 
        """
        根据当前机器人位置和已知地图，规划下一个目标点。
        返回目标点 (r, c) 和到该目标点的路径列表，或 (None, None) (如果无法规划)。
        """
        pass

    def find_frontiers(self, known_map): 
        frontiers = []
        height, width = known_map.shape
        for r in range(height):
            for c in range(width):
                if known_map[r, c] == FREE:
                    is_frontier = False
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if known_map[nr, nc] == UNKNOWN:
                                is_frontier = True
                                break
                    if is_frontier:
                        frontiers.append((r, c))
        return frontiers

    def _is_reachable_and_get_path(self, start_pos, end_pos, known_map):
        if start_pos == end_pos:
            return [start_pos]
        path = a_star_search(known_map, start_pos, end_pos)
        return path

    def _final_fallback_plan(self, robot_pos, known_map):
        """
        A universal fallback: find the closest reachable frontier.
        This is similar to FBE's core logic.
        Returns (target_pos, path_to_target) or (None, None).
        """        
        frontiers = self.find_frontiers(known_map)
        if not frontiers:
            return None, None

        min_path_len = float('inf')
        best_frontier_fallback = None
        best_path_fallback = None

        for fr_pos in frontiers:
            path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
            if path:
                path_len = len(path) - 1
                if path_len < min_path_len:
                    min_path_len = path_len
                    best_frontier_fallback = fr_pos
                    best_path_fallback = path
           
        return best_frontier_fallback, best_path_fallback