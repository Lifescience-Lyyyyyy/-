# planners/ige_planner.py
from .base_planner import BasePlanner
import numpy as np
from environment import UNKNOWN # 地图状态常量
# 从新的 geometry_utils 模块导入 LoS 检查函数
from .geometry_utils import check_line_of_sight 

class IGEPlanner(BasePlanner):
    def __init__(self, environment, robot_sensor_range):
        super().__init__(environment)
        self.robot_sensor_range = robot_sensor_range # 规划器用于计算IG的传感器范围

    def _calculate_information_gain_with_los(self, 
                                             prospective_robot_pos, # 机器人假设将要到达的位置 (即边界点)
                                             current_known_map,     # 当前的已知地图，用于LoS和判断UNKNOWN
                                             sensor_range_for_ig):  # 计算IG时使用的传感器范围
        """
        计算如果机器人位于 prospective_robot_pos 并进行感知，能够获得的信息增益，
        同时考虑从 prospective_robot_pos 出发的视线遮挡。
        LoS 是基于 current_known_map 判断的。
        """
        ig = 0
        r_prospective, c_prospective = prospective_robot_pos

        for dr in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
            for dc in range(-sensor_range_for_ig, sensor_range_for_ig + 1):
                # Optional: circular sensor shape
                # if dr**2 + dc**2 > sensor_range_for_ig**2:
                #     continue

                # 传感器范围内的目标单元格的绝对坐标
                r_target_cell, c_target_cell = r_prospective + dr, c_prospective + dc

                if self.environment.is_within_bounds(r_target_cell, c_target_cell):
                    # 步骤 1: 检查这个单元格在当前已知地图中是否是 UNKNOWN
                    if current_known_map[r_target_cell, c_target_cell] == UNKNOWN:
                        # 步骤 2: 检查从机器人假设的新位置(r_prospective, c_prospective)
                        #         到这个目标未知单元格(r_target_cell, c_target_cell)之间是否有视线。
                        #         LoS检查基于当前的已知地图 (current_known_map)。
                        if check_line_of_sight(r_prospective, c_prospective, 
                                               r_target_cell, c_target_cell, 
                                               current_known_map):
                            ig += 1 # 如果是未知且视线可达，则计入信息增益
        return ig

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        frontiers = self.find_frontiers(known_map)
        if not frontiers:
            # print(f"IGE: No frontiers found from {robot_pos}.")
            return None, None

        best_utility = -float('inf')
        best_frontier_ige = None 
        best_path_ige = None
        
        # 在规划开始时获取一份已知地图的快照，用于本轮所有IG和路径规划计算
        # 这样可以保证在评估不同边界点时，所基于的地图信息是一致的
        known_map_snapshot = np.copy(known_map)

        for fr_pos in frontiers: # 遍历所有找到的边界点
            # 路径规划基于快照
            path = self._is_reachable_and_get_path(robot_pos, fr_pos, known_map_snapshot)
            if path: # 如果边界点可达
                path_cost = len(path) - 1
                if path_cost <= 0: path_cost = 1e-5 # 避免除零或负成本

                # 使用带有LoS检查的新方法计算信息增益
                # IG计算也基于快照，并且传感器范围使用规划器自己的设置
                information_gain = self._calculate_information_gain_with_los(
                    fr_pos, # 机器人假设会到达这个边界点
                    known_map_snapshot, # 基于当前的已知情况进行LoS判断
                    self.robot_sensor_range # 使用IGEPlanner配置的传感器范围
                )
                
                utility = -1.0 
                if information_gain <= 0 and path_cost > 1e-4 : 
                    utility = -1.0 
                elif information_gain <=0 and path_cost <=1e-4: # 例如，当前位置就是一个IG为0的边界
                    utility = 0.1 # 给一个小的正效用，避免完全忽略
                else: 
                    utility = information_gain / path_cost # 标准效用计算
                
                if utility > best_utility:
                    best_utility = utility
                    best_frontier_ige = fr_pos
                    best_path_ige = path
        
        if best_frontier_ige is not None:
            # print(f"IGE Selected: {best_frontier_ige} with utility {best_utility:.2f}")
            return best_frontier_ige, best_path_ige
        else:
            # print(f"IGE: No suitable frontier found with positive utility. Falling back.")
            # 如果IGE的主要逻辑未能选出目标，则调用回退策略
            # 回退策略应该使用原始的、最新的 known_map 进行规划，而不是快照
            return self._final_fallback_plan(robot_pos, known_map)