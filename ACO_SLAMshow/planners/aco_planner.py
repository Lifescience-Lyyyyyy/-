# planners/aco_planner.py
from .base_planner import BasePlanner
import numpy as np
import random
from environment import UNKNOWN, FREE, OBSTACLE # 地图状态常量
from .geometry_utils import check_line_of_sight # 视线检查 (用于启发式函数中的IG计算)
from .pathfinding import heuristic as manhattan_heuristic # 曼哈顿距离启发式

class ACOPlanner(BasePlanner):
    def __init__(self, environment, 
                 # --- 蚂蚁模拟更新路径信息素的参数 ---
                 n_ants_update=10,          # 用于更新信息素地图的蚂蚁数量
                 n_iterations_update=10,    # 更新信息素地图时的迭代次数
                 alpha_step_choice=1.0,     # 蚂蚁选择下一步时，信息素的影响因子 (τ^α)
                 beta_step_heuristic=2.0,   # 蚂蚁选择下一步时，启发式信息的影响因子 (η^β)
                 evaporation_rate_map=0.1,  # 全局信息素地图的蒸发率 (较低的值使信息素保留更久)
                 pheromone_min_map=0.01,    # 信息素地图上单元格的最小信息素值
                 pheromone_max_map=10.0,    # 信息素地图上单元格的最大信息素值
                 q_path_deposit_factor=1.0, # 蚂蚁在路径上沉积信息素的基础因子
                 ant_max_steps = 50,        # 单只蚂蚁在更新信息素时走的最大步数

                 # --- 机器人信息素导航参数 ---
                 max_pheromone_nav_steps=75, # 机器人沿信息素导航时的最大步数

                 # --- 蚂蚁选择下一步的启发式参数 ---
                 eta_weight_to_unknown=2.5,     # 启发式中朝向未知单元格的权重
                 eta_weight_to_frontiers_centroid=1.5, # 启发式中朝向边界点质心的权重
                 
                 visualize_ants_callback=None, # 可视化回调函数
                 robot_actual_sensor_range=5     # 机器人实际的传感器范围，用于IG启发式
                 ):
        super().__init__(environment)
        
        # 蚂蚁模拟参数
        self.n_ants_update = n_ants_update
        self.n_iterations_update = n_iterations_update
        self.alpha_step_choice = alpha_step_choice
        self.beta_step_heuristic = beta_step_heuristic
        self.evaporation_rate_map = evaporation_rate_map
        self.pheromone_min_map = pheromone_min_map
        self.pheromone_max_map = pheromone_max_map
        self.q_path_deposit_factor = q_path_deposit_factor
        self.ant_max_steps = ant_max_steps

        # 信息素导航参数
        self.max_pheromone_nav_steps = max_pheromone_nav_steps

        # 启发式参数
        self.eta_weight_to_unknown = eta_weight_to_unknown
        self.eta_weight_to_frontiers_centroid = eta_weight_to_frontiers_centroid
        
        self.visualize_ants_callback = visualize_ants_callback
        self.robot_actual_sensor_range = robot_actual_sensor_range

        # 初始化全局路径信息素地图
        # 初始值可以设为最小值或一个较小的基础值
        initial_pheromone_on_map = (self.pheromone_min_map + self.pheromone_max_map) / 10.0 
        self.pheromone_map = np.full((self.environment.height, self.environment.width), 
                                     initial_pheromone_on_map, dtype=float)
        # 注意：如果环境有初始已知障碍物，应该在第一次sense之后调用
        # _update_pheromone_map_for_obstacles 来设置障碍物处信息素为最小值


    def _update_pheromone_map_for_obstacles(self, known_map):
        """确保已知障碍物在信息素地图上具有最低信息素值。"""
        obstacle_mask = (known_map == OBSTACLE)
        self.pheromone_map[obstacle_mask] = self.pheromone_min_map


    def _get_local_step_heuristic(self, from_r, from_c, to_r, to_c, known_map, current_frontiers_list):
        """
        计算蚂蚁从 (from_r, from_c) 移动到相邻单元格 (to_r, to_c) 这一步的启发式值 (eta)。
        这个启发式用于指导蚂蚁在更新信息素地图时的路径构建。
        """
        eta = 1.0  # 基础启发值

        # 1. 鼓励走向未知区域
        if self.environment.is_within_bounds(to_r, to_c) and \
           known_map[to_r, to_c] == UNKNOWN:
            eta += self.eta_weight_to_unknown

        # 2. 鼓励走向整体边界区域的质心方向
        #    （这是一个全局性的指引，让蚂蚁大致有个探索方向）
        if current_frontiers_list and self.eta_weight_to_frontiers_centroid > 0:
            # 为避免每次都计算所有边界点的质心（可能较慢），可以采样或使用缓存的质心
            # 这里简化为使用所有当前边界点的质心
            avg_fr, avg_fc = np.mean(current_frontiers_list, axis=0)
            
            dist_from_to_centroid = manhattan_heuristic((from_r, from_c), (avg_fr, avg_fc))
            dist_to_to_centroid = manhattan_heuristic((to_r, to_c), (avg_fr, avg_fc))

            if dist_to_to_centroid < dist_from_to_centroid:  # 如果下一步更接近质心
                eta += self.eta_weight_to_frontiers_centroid
            elif dist_to_to_centroid > dist_from_to_centroid: # 如果下一步远离质心
                eta -= self.eta_weight_to_frontiers_centroid * 0.5 # 轻微惩罚

        # 3. （可选）考虑从 (to_r, to_c) 出发能看到的（带LoS的）信息增益
        #    这会使启发式计算更复杂，但可能更精确
        #    ig_at_next_step = self._calculate_ig_with_los(to_r, to_c, known_map, self.robot_actual_sensor_range)
        #    eta += ig_at_next_step * some_weight 
        
        return max(0.1, eta) # 确保启发式值至少为一个很小正数


    def _calculate_ig_with_los(self, prospective_r, prospective_c, known_map, sensor_range):
        """辅助函数：计算在(prospective_r, prospective_c)处的信息增益，考虑LoS。"""
        ig = 0
        for dr_ig in range(-sensor_range, sensor_range + 1):
            for dc_ig in range(-sensor_range, sensor_range + 1):
                # if dr_ig**2 + dc_ig**2 > sensor_range**2: continue # Optional circular
                
                r_target, c_target = prospective_r + dr_ig, prospective_c + dc_ig
                if self.environment.is_within_bounds(r_target, c_target):
                    if known_map[r_target, c_target] == UNKNOWN:
                        if check_line_of_sight(prospective_r, prospective_c, 
                                               r_target, c_target, 
                                               known_map): # LoS基于当前known_map
                            ig += 1
        return ig


    def _run_ant_simulation_to_update_pheromones(self, robot_current_pos, current_known_map):
        """
        核心的蚂蚁模拟过程，用于更新全局路径信息素地图 self.pheromone_map。
        蚂蚁从机器人当前位置出发，构建路径，并在路径上沉积信息素。
        """
        # print(f"ACO DEBUG: Updating pheromone map from robot_pos={robot_current_pos}...")
        all_ant_paths_for_visualization = [] # 用于可视化回调
        
        # 获取当前的边界点列表，供蚂蚁的启发式函数使用
        live_frontiers = self.find_frontiers(current_known_map) 

        for iteration_num in range(self.n_iterations_update):
            # print(f"  ACO Pheromone Update Iteration: {iteration_num + 1}/{self.n_iterations_update}")
            paths_this_iter_for_viz = []
            ant_paths_data_this_iter = [] # 存储 [(path, quality), ...]

            for ant_idx in range(self.n_ants_update):
                ant_r, ant_c = robot_current_pos[0], robot_current_pos[1]
                current_ant_path = [(ant_r, ant_c)] # 路径以机器人当前位置开始

                for step_num in range(self.ant_max_steps):
                    # --- 蚂蚁选择下一步 ---
                    possible_next_moves = [] # {'pos': (r,c), 'score': X}
                    # 蚂蚁可以向8个方向移动
                    for dr_ant, dc_ant in [(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0)]:
                        next_r, next_c = ant_r + dr_ant, ant_c + dc_ant

                        if not self.environment.is_within_bounds(next_r, next_c): continue
                        if current_known_map[next_r, next_c] == OBSTACLE: continue
                        # 简单地避免立即返回上一步或上上一步，防止抖动
                        if (next_r, next_c) in current_ant_path[-2:]: continue 

                        # 获取信息素 τ (tau)
                        pheromone_val = self.pheromone_map[next_r, next_c]
                        tau = pheromone_val ** self.alpha_step_choice
                        
                        # 获取启发式信息 η (eta)
                        heuristic_val = self._get_local_step_heuristic(ant_r, ant_c, next_r, next_c, 
                                                                     current_known_map, live_frontiers)
                        eta = heuristic_val ** self.beta_step_heuristic
                        
                        possible_next_moves.append({'pos': (next_r, next_c), 'score': tau * eta})
                    
                    if not possible_next_moves: break # 蚂蚁卡住了

                    # --- 根据分数进行轮盘赌选择 ---
                    total_score_sum = sum(move['score'] for move in possible_next_moves)
                    chosen_move = None
                    if total_score_sum <= 1e-9: # 如果所有分数都接近0
                        chosen_move = random.choice(possible_next_moves) # 随机选一个
                    else:
                        rand_select_val = random.random() * total_score_sum
                        current_sum_for_select = 0
                        for move in possible_next_moves:
                            current_sum_for_select += move['score']
                            if current_sum_for_select >= rand_select_val:
                                chosen_move = move
                                break
                        if chosen_move is None: # 以防万一（浮点数问题）
                            chosen_move = random.choice(possible_next_moves)
                    
                    ant_r, ant_c = chosen_move['pos']
                    current_ant_path.append((ant_r, ant_c))
                
                # 蚂蚁路径构建完成，评估其质量
                # 简单质量度量：路径中独立单元格的数量
                path_quality_metric = len(set(current_ant_path)) 
                ant_paths_data_this_iter.append({'path': current_ant_path, 'quality': path_quality_metric})
                
                if self.visualize_ants_callback:
                    paths_this_iter_for_viz.append(current_ant_path)
            
            if self.visualize_ants_callback and paths_this_iter_for_viz:
                 all_ant_paths_for_visualization.append(paths_this_iter_for_viz)

            # --- 信息素更新 ---
            # 1. 全局蒸发
            self.pheromone_map *= (1.0 - self.evaporation_rate_map)
            
            # 2. 蚂蚁路径沉积
            for ant_data in ant_paths_data_this_iter:
                path = ant_data['path']
                quality = ant_data['quality']
                if not path or len(path) <= 1: continue # 路径太短无意义

                # 信息素沉积量可以与路径质量成正比，与路径长度成反比
                # delta_tau = (self.q_path_deposit_factor * quality) / len(path)
                # 或者每一步都沉积少量，累积起来
                # 简化：每条路径平均分配固定总量的信息素，质量好的路径分配更多
                base_deposit_for_path = self.q_path_deposit_factor * (quality / (self.ant_max_steps + 1e-5)) # 归一化质量

                for r_cell, c_cell in path:
                    self.pheromone_map[r_cell, c_cell] += base_deposit_for_path
            
            # 3. 确保信息素在边界内，并重置障碍物信息素
            self.pheromone_map = np.clip(self.pheromone_map, self.pheromone_min_map, self.pheromone_max_map)
            self._update_pheromone_map_for_obstacles(current_known_map) # 重新确保障碍物信息素最低

        # 可视化所有蚂蚁迭代（如果启用了）
        if self.visualize_ants_callback and all_ant_paths_for_visualization:
            self.visualize_ants_callback(robot_current_pos, current_known_map, 
                                         all_ant_paths_for_visualization, self.environment)


    def _navigate_by_pheromones_to_frontier(self, robot_start_pos, current_known_map):
        """
        从机器人起点开始，贪婪地沿着信息素最高的路径行走，直到到达一个边界点。
        返回: 找到的边界点坐标 (r,c)，如果失败则返回 None。
        """
        current_r, current_c = robot_start_pos
        # 记录本次导航走过的路径，避免在同一次导航中陷入小循环
        navigation_path_taken_log = {(current_r, current_c)} 

        for nav_step_count in range(self.max_pheromone_nav_steps):
            # 检查当前位置是否是边界点
            if current_known_map[current_r, current_c] == FREE:
                is_frontier_now = False
                for dr_f, dc_f in [(0,1),(0,-1),(1,0),(-1,0)]: # 检查4个邻居
                    nr_f, nc_f = current_r + dr_f, current_c + dc_f
                    if self.environment.is_within_bounds(nr_f,nc_f) and current_known_map[nr_f,nc_f] == UNKNOWN:
                        is_frontier_now = True
                        break
                if is_frontier_now:
                    # print(f"  Pheromone Nav: Reached frontier { (current_r, current_c) } in {nav_step_count} steps.")
                    return (current_r, current_c) # 成功找到边界点

            # 选择信息素最高的、可走的下一步
            best_next_candidate_pos = None
            max_pher_val_for_next = -1.0
            
            # 机器人导航时通常只考虑4个方向
            for dr_nav, dc_nav in [(0,1),(0,-1),(1,0),(-1,0)]:
                next_r_nav, next_c_nav = current_r + dr_nav, current_c + dc_nav

                if not self.environment.is_within_bounds(next_r_nav, next_c_nav): continue
                if current_known_map[next_r_nav, next_c_nav] == OBSTACLE: continue
                if (next_r_nav, next_c_nav) in navigation_path_taken_log: continue # 避免立即回头

                current_cell_pheromone = self.pheromone_map[next_r_nav, next_c_nav]
                if current_cell_pheromone > max_pher_val_for_next:
                    max_pher_val_for_next = current_cell_pheromone
                    best_next_candidate_pos = (next_r_nav, next_c_nav)
            
            if best_next_candidate_pos is None: # 没有可走的、信息素更高的邻居了
                # print(f"  Pheromone Nav: Stuck at {(current_r, current_c)} after {nav_step_count} steps. No valid next move.")
                return None # 卡住了
            
            current_r, current_c = best_next_candidate_pos
            navigation_path_taken_log.add((current_r, current_c)) # 记录已走过的点

        # print(f"  Pheromone Nav: Max steps ({self.max_pheromone_nav_steps}) reached without finding frontier.")
        return None # 达到最大导航步数


    def plan_next_action(self, robot_pos, known_map, **kwargs):
        """
        规划机器人的下一步行动。
        1. 更新全局路径信息素地图。
        2. 使用信息素导航找到一个有潜力的边界点。
        3. 使用A*规划到该边界点的路径。
        4. 如果失败，则使用回退策略。
        """
        # print(f"ACO Planner called at robot_pos={robot_pos}")
        
        # 步骤 1: 更新全局路径信息素地图
        # (按照您的要求，每次规划都更新。在实际应用中，这个频率可能需要调整)
        # print("  ACO: Running ant simulation to update pheromone map...")
        # 使用一份地图快照进行蚂蚁模拟，以保证在模拟期间地图认知不变
        known_map_for_ant_sim = np.copy(known_map)
        self._update_pheromone_map_for_obstacles(known_map_for_ant_sim) # 确保障碍物信息素最低
        self._run_ant_simulation_to_update_pheromones(robot_pos, known_map_for_ant_sim)
        # print("  ACO: Pheromone map updated.")

        # 步骤 2: 使用当前更新后的信息素地图进行导航，找到目标边界点
        # print("  ACO: Navigating by pheromones to find a frontier...")
        # 导航时也使用一份快照，以防在导航过程中 known_map 被其他线程修改（不太可能在此架构中）
        known_map_for_navigation = np.copy(known_map)
        targeted_frontier_by_pheromone = self._navigate_by_pheromones_to_frontier(
            robot_pos, 
            known_map_for_navigation
        )

        if targeted_frontier_by_pheromone:
            # print(f"  ACO: Pheromone navigation targeted frontier: {targeted_frontier_by_pheromone}")
            # 步骤 3: 使用A*规划到这个由信息素引导选出的边界点的路径
            # A*规划应该在最新的已知地图上进行
            path_to_target = self._is_reachable_and_get_path(
                robot_pos, 
                targeted_frontier_by_pheromone, 
                known_map # 使用最新的known_map
            )
            if path_to_target:
                # print(f"  ACO: A* path found to {targeted_frontier_by_pheromone}.")
                return targeted_frontier_by_pheromone, path_to_target
            else:
                # print(f"  ACO: Pheromone navigation found {targeted_frontier_by_pheromone}, but A* failed. Falling back.")
                return self._final_fallback_plan(robot_pos, known_map)
        else:
            # print("  ACO: Pheromone navigation failed to find any frontier. Falling back.")
            return self._final_fallback_plan(robot_pos, known_map)