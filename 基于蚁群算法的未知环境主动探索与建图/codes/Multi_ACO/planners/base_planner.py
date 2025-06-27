from abc import ABC, abstractmethod
import numpy as np
from environment import UNKNOWN, FREE, OBSTACLE
from .pathfinding import a_star_search

class BasePlanner(ABC):
    def __init__(self, environment):
        self.environment = environment

    @abstractmethod
    def plan_next_action(self, robot_pos, known_map, **kwargs):
        pass

    def find_frontiers(self, known_map):
        """寻找与未知区域邻接的自由单元格 (标准边界点)。"""
        frontiers = []
        height, width = known_map.shape
        free_rows, free_cols = np.where(known_map == FREE)
        
        for r, c in zip(free_rows, free_cols):
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and known_map[nr, nc] == UNKNOWN:
                    frontiers.append((r, c))
                    break
        return list(set(frontiers))

    def find_internal_frontiers(self, known_map):
        """
        寻找被自由空间包围的未知单元格的邻居 (内部边界点)。
        这用于探索后期，当标准边界点消失时，清理地图内部的未知“洞穴”。
        """
        internal_frontiers = []
        height, width = known_map.shape
        # 找到所有未知单元格
        unknown_rows, unknown_cols = np.where(known_map == UNKNOWN)

        for r, c in zip(unknown_rows, unknown_cols):
            # 检查这个未知单元格是否邻接一个自由单元格
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and known_map[nr, nc] == FREE:
                    # 如果是，那么这个自由单元格(nr, nc)就是一个内部边界点
                    internal_frontiers.append((nr, nc))
        
        return list(set(internal_frontiers))


    def _is_reachable_and_get_path(self, start_pos, end_pos, known_map):
        if start_pos == end_pos:
            return [start_pos]
        path = a_star_search(known_map, start_pos, end_pos)
        return path

    def _final_fallback_plan(self, robot_pos, known_map, reserved_targets={}):
        """
        回退策略：首先寻找标准边界点，如果找不到，再寻找内部边界点。
        """
        # 1. 优先寻找标准边界点
        frontiers = self.find_frontiers(known_map)
        available_frontiers = [fr for fr in frontiers if fr not in reserved_targets.values()]
        
        if available_frontiers:
            # 在可用边界点中寻找最近的
            min_path_len = float('inf')
            best_frontier, best_path = None, None
            for fr_pos in available_frontiers:
                path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
                if path and (len(path) - 1) < min_path_len:
                    min_path_len = len(path) - 1
                    best_frontier, best_path = fr_pos, path
            if best_frontier:
                return best_frontier, best_path
        
        # 2. 如果没有标准边界点，则进入“清扫”模式，寻找内部边界点
        internal_frontiers = self.find_internal_frontiers(known_map)
        available_internal = [fr for fr in internal_frontiers if fr not in reserved_targets.values()]
        
        if available_internal:
            min_path_len = float('inf')
            best_frontier, best_path = None, None
            for fr_pos in available_internal:
                path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
                if path and (len(path) - 1) < min_path_len:
                    min_path_len = len(path) - 1
                    best_frontier, best_path = fr_pos, path
            if best_frontier:
                return best_frontier, best_path

        # 如果两种边界点都找不到，则确实没有可探索的地方了
        return None, None