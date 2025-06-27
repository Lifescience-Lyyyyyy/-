from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN, FREE, OBSTACLE
from robot import bresenham_line 

class UREPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range
        self.observation_counts = np.zeros_like(environment.grid_known, dtype=int)
        
        # --- 新参数 ---
        # 当探索效用低于此阈值时，才考虑巩固
        self.exploration_utility_threshold = 0.5 
        # 巩固目标的最低观察次数阈值
        self.low_obs_threshold = 3

    def update_observation_counts(self, robot_pos, known_map_after_sense):
        """
        这个方法在 `sense()` 之后被调用，`sense()` 已经处理了LoS。
        因此，这里我们只增加已知为 FREE 的单元格的计数。
        """
        r_rob, c_rob = robot_pos
        # 我们可以简化这个逻辑，因为 sense() 已经完成了 LoS 检查
        # 我们只关心传感器范围内的、现在已知是 FREE 的格子
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr, nc = r_rob + dr, c_rob + dc
                if self.environment.is_within_bounds(nr, nc) and known_map_after_sense[nr, nc] == FREE:
                    # 只要它在范围内且是FREE，就增加计数。
                    # LoS的限制已经体现在哪些格子能变成FREE上。
                    self.observation_counts[nr, nc] += 1

    def _is_visible(self, start_pos, end_pos, known_map):
        line = bresenham_line(start_pos, end_pos)
        for r, c in line[1:-1]:
            if not self.environment.is_within_bounds(r, c): return False
            if known_map[r, c] == OBSTACLE: return False
        return True

    def _calculate_exploration_utility(self, path_cost, frontier_pos, known_map):
        ig = 0
        r_f, c_f = frontier_pos
        for dr_s in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc_s in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                nr_s, nc_s = r_f + dr_s, c_f + dc_s
                if self.environment.is_within_bounds(nr_s, nc_s) and known_map[nr_s, nc_s] == UNKNOWN:
                    if self._is_visible((r_f, c_f), (nr_s, nc_s), known_map):
                        ig += 1
        if ig <= 0: return 0.0
        # 使用 path_cost + 1 来避免除以非常小的数，增加稳定性
        return ig / (path_cost + 1.0)

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        """
        新版决策逻辑：
        1. 找到最佳的探索目标。
        2. 如果最佳探索目标的效用足够高，就去探索。
        3. 如果探索效用太低，则寻找最佳的巩固目标。
        4. 如果有可行的巩固目标，就去巩固。
        5. 如果两者都不可行，执行最终回退（找最近的前沿点）。
        """
        
        # --- 1. 评估所有探索目标 ---
        frontiers = self.find_frontiers(known_map)
        best_expl_utility = -1.0
        best_expl_target = None
        best_expl_path = None

        if frontiers:
            for fr_pos in frontiers:
                path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map)
                if path:
                    path_cost = len(path) - 1
                    utility = self._calculate_exploration_utility(path_cost, fr_pos, known_map)
                    if utility > best_expl_utility:
                        best_expl_utility = utility
                        best_expl_target = fr_pos
                        best_expl_path = path

        # --- 2. 决策：探索还是巩固？ ---

        # 如果找到了一个不错的探索目标，就执行它
        if best_expl_target is not None and best_expl_utility > self.exploration_utility_threshold:
            # print(f"URE Decision: EXPLORE. Target: {best_expl_target}, Utility: {best_expl_utility:.2f}")
            return best_expl_target, best_expl_path

        # --- 3. 如果探索不吸引人，则评估巩固目标 ---
        # print(f"URE Info: Exploration utility ({best_expl_utility:.2f}) is low. Considering consolidation.")
        
        # 寻找所有可达的、观察次数少的自由单元格
        rows, cols = np.where((known_map == FREE) & (self.observation_counts < self.low_obs_threshold))
        
        best_cons_score = -1.0
        best_cons_target = None
        best_cons_path = None

        for r, c in zip(rows, cols):
            cand_pos = (r, c)
            path = self._is_reachable_and_get_path(robot_pos, cand_pos, known_map)
            if path:
                path_cost = len(path) - 1
                obs_count = self.observation_counts[r, c]
                
                # 巩固评分：优先选择观察次数少、且路径成本低的
                # (low_obs_threshold - obs_count) 确保观察次数越少，得分越高
                score = (self.low_obs_threshold - obs_count) / (path_cost + 1.0)

                if score > best_cons_score:
                    best_cons_score = score
                    best_cons_target = cand_pos
                    best_cons_path = path

        # 如果找到了一个可行的巩固目标，就执行它
        if best_cons_target is not None:
            # print(f"URE Decision: CONSOLIDATE. Target: {best_cons_target}, Score: {best_cons_score:.2f}")
            return best_cons_target, best_cons_path
        
        # --- 4. 最终回退 ---
        
        # 如果连巩固目标也找不到（比如地图已充分观察），
        # 但之前找到了一个探索目标（即使效用低），也去执行它
        if best_expl_target is not None:
            # print(f"URE Decision: FALLBACK to best available exploration target.")
            return best_expl_target, best_expl_path
            
        # 如果连一个探索目标都找不到（例如被困住了），则调用最终回退方案
        # 这种情况很少见，但作为保险是必要的
        # print(f"URE Decision: CRITICAL FALLBACK. No exploration or consolidation targets found.")
        return self._final_fallback_plan(robot_pos, known_map)