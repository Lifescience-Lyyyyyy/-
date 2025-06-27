from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN, FREE, OBSTACLE # OBSTACLE 可能在 geometry_utils 中被间接使用
# 从新的 geometry_utils 模块导入 LoS 检查函数
try:
    from .geometry_utils import check_line_of_sight
except ImportError: # Fallback for different project structures or direct execution
    try:
        from geometry_utils import check_line_of_sight
    except ImportError:
        def check_line_of_sight(sr, sc, tr, tc, m): # Dummy
            print("Warning (ure_planner.py): Using dummy check_line_of_sight. Ensure geometry_utils.py is accessible.")
            return True


class UREPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range # 机器人实际的传感器范围
        # 观测计数矩阵：记录已知地图上每个FREE单元格被机器人传感器观测到的次数
        self.observation_counts = np.zeros_like(environment.grid_known, dtype=int)
        self.exploration_weight = 0.2  # 探索效用的权重
        self.consolidation_weight = 0.8 # 巩固效用的权重
        self.min_utility_threshold = 1e-4 # 选择目标的最小效用阈值

    def update_observation_counts(self, robot_pos, known_map_after_sense):
        """
        当机器人在新位置感知后，更新其传感器范围内的FREE单元格的观测次数。
        这个方法由 main_simulation.py 在机器人 sense() 之后调用。
        """
        r_rob, c_rob = robot_pos
        for dr in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
            for dc in range(-self.robot_sensor_range, self.robot_sensor_range + 1):
                # Optional: circular sensor shape
                # if dr**2 + dc**2 > self.robot_sensor_range**2:
                #     continue
                nr, nc = r_rob + dr, c_rob + dc
                # 如果在地图内且是已知的自由空间
                if self.environment.is_within_bounds(nr, nc) and \
                   known_map_after_sense[nr, nc] == FREE:
                    self.observation_counts[nr, nc] += 1 # 增加观测计数

    def _calculate_ig_with_los_for_ure(self, 
                                       prospective_robot_pos, 
                                       current_known_map,
                                       sensor_range_for_ig):
        """
        为URE的探索部分计算信息增益，考虑从 prospective_robot_pos 出发的LoS。
        LoS 是基于 current_known_map 判断的。
        """
        ig = 0
        r_prospective, c_prospective = prospective_robot_pos
        for dr in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
            for dc in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
                # Optional: circular sensor shape
                # if dr**2 + dc**2 > sensor_range_for_ig**2:
                #     continue

                r_target_cell, c_target_cell = r_prospective + dr, c_prospective + dc

                if self.environment.is_within_bounds(r_target_cell, c_target_cell):
                    # 步骤 1: 检查这个单元格在当前已知地图中是否是 UNKNOWN
                    if current_known_map[r_target_cell, c_target_cell] == UNKNOWN:
                        # 步骤 2: 检查从机器人假设的新位置到这个目标未知单元格是否有LoS
                        if check_line_of_sight(r_prospective, c_prospective, 
                                               r_target_cell, c_target_cell, 
                                               current_known_map):
                            ig += 1 # 如果是未知且视线可达，则计入信息增益
        return ig

    def _calculate_exploration_utility_with_los(self, path_cost, frontier_pos, known_map_snapshot):
        """
        计算探索边界点的效用，信息增益的计算考虑LoS。
        使用机器人配置的实际传感器范围 (self.robot_sensor_range)。
        """
        # 使用带有LoS的IG计算
        ig = self._calculate_ig_with_los_for_ure(
            frontier_pos,        # 机器人假设会到达这个边界点
            known_map_snapshot,  # 基于当前的已知情况(快照)进行LoS判断
            self.robot_sensor_range # 使用UREPlanner配置的实际传感器范围
        )
        
        # 效用计算，处理特殊情况
        if ig <= 0 and path_cost > 1e-4: # 没有信息增益且不是当前位置
            return -1.0 
        if ig <=0 and path_cost <= 1e-4 : # 没有信息增益但在当前位置（成本极小）
            return 0.1 # 给一个小的正效用，避免完全忽略，以防这是唯一选择
        
        # 正常情况：IG / Cost (加一个小数避免除零)
        return ig / (path_cost + 1e-5) 

    def _find_consolidation_candidates(self, known_map):
        """
        寻找巩固目标：已探索但观测次数较少的自由单元格。
        """
        candidates = []
        low_obs_threshold = 2 # 观测次数低于此阈值的被认为是候选
        # 找到所有是FREE且观测次数低于或等于阈值的单元格
        rows, cols = np.where((known_map == FREE) & (self.observation_counts <= low_obs_threshold))
        for r, c in zip(rows, cols): 
            candidates.append({"pos": (r,c), "obs_count": self.observation_counts[r,c]})
        
        # 按观测次数升序排序（优先选择观测次数更少的）
        candidates.sort(key=lambda x: x["obs_count"])
        return candidates[:30] # 返回前N个候选，避免过多计算

    def _calculate_consolidation_utility(self, path_cost, obs_count):
        """
        计算巩固目标的效用。观测次数越少，成本越低，效用越高。
        """
        # (1 / (观测次数 + epsilon)) 使得观测次数少的项更大
        # 再除以 (路径成本 + epsilon)
        return (1.0 / (obs_count + 0.1)) / (path_cost + 0.1)

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        best_overall_utility = -float('inf') 
        best_target_ure = None
        best_path_ure = None
        
        # 在规划开始时获取一份已知地图的快照
        # 本轮规划中的所有路径查找、LoS检查、IG计算都将基于此快照
        known_map_snapshot = np.copy(known_map)

        # --- 1. 评估探索目标 (边界点) ---
        frontiers = self.find_frontiers(known_map_snapshot) # 基于快照找边界
        # print(f"URE Debug: Found {len(frontiers)} frontiers.")
        if frontiers:
            for fr_pos in frontiers:
                # 路径规划也基于快照
                path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map_snapshot)
                if path:
                    path_cost = len(path) - 1
                    if path_cost < 0: path_cost = 0 # 如果起点即边界点
                    
                    # 使用更新后的探索效用计算（包含LoS）
                    expl_utility = self._calculate_exploration_utility_with_los(
                        path_cost, fr_pos, known_map_snapshot
                    )
                    weighted_utility = self.exploration_weight * expl_utility
                    # print(f"URE Debug: Frontier {fr_pos}, ExplUtil (w/ LoS): {expl_utility:.2f}, Weighted: {weighted_utility:.2f}")
                    if weighted_utility > best_overall_utility:
                        best_overall_utility = weighted_utility
                        best_target_ure = fr_pos
                        best_path_ure = path

        # --- 2. 评估巩固目标 ---
        # 巩固目标的选择基于当前的观测计数和快照地图
        consolidation_candidates = self._find_consolidation_candidates(known_map_snapshot)
        # print(f"URE Debug: Found {len(consolidation_candidates)} consolidation candidates.")
        if consolidation_candidates:
            for cand_info in consolidation_candidates:
                ct_pos, obs_count = cand_info["pos"], cand_info["obs_count"]
                # 路径规划基于快照
                path = self._is_reachable_and_get_path(robot_pos, ct_pos, known_map_snapshot)
                if path:
                    path_cost = len(path) -1
                    if path_cost < 0: path_cost = 0

                    cons_utility = self._calculate_consolidation_utility(path_cost, obs_count)
                    weighted_utility = self.consolidation_weight * cons_utility
                    # print(f"URE Debug: Consolidate {ct_pos} (obs:{obs_count}), ConsUtil: {cons_utility:.2f}, Weighted: {weighted_utility:.2f}")
                    if weighted_utility > best_overall_utility:
                        best_overall_utility = weighted_utility
                        best_target_ure = ct_pos
                        best_path_ure = path
        
        # --- 决策 ---
        # 如果找到了一个效用高于预设最小阈值的目标
        if best_target_ure is not None and best_overall_utility >= self.min_utility_threshold:
            # print(f"URE Selected (Primary Logic): {best_target_ure} with overall utility {best_overall_utility:.3f}")
            return best_target_ure, best_path_ure
        else: 
            # 如果主要逻辑未能找到足够好的目标，则执行最终的回退计划
            # (寻找最近的可达边界点，不考虑复杂效用)
            # print(f"URE: Primary logic insufficient (best_utility: {best_overall_utility:.3f}). Executing _final_fallback_plan.")
            # 回退计划应该使用最新的（非快照的）known_map，因为它是在当前时刻做出的最后努力
            fallback_target, fallback_path = self._final_fallback_plan(robot_pos, known_map)
            # if fallback_target:
            #     print(f"URE Selected (Final Fallback): {fallback_target}")
            # else:
            #     print(f"URE: Final Fallback also failed. No target.")
            return fallback_target, fallback_path