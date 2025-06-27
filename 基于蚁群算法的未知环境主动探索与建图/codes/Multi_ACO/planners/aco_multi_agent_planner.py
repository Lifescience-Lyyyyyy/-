from .base_planner import BasePlanner
import numpy as np
import random
from environment import UNKNOWN, FREE, OBSTACLE
from .pathfinding import heuristic as manhattan_heuristic

class ACOMultiAgentPlanner(BasePlanner):
    def __init__(self, environment, 
                 robot_actual_sensor_range,
                 n_ants_update, n_iterations_update,
                 alpha_step_choice, beta_step_heuristic,
                 evaporation_rate_map, pheromone_min_map, pheromone_max_map,
                 q_path_deposit_factor, ant_max_steps,
                 max_pheromone_nav_steps,
                 eta_weight_to_unknown, eta_weight_to_frontiers_centroid,
                 visualize_ants_callback=None):
        
        super().__init__(environment)
        self.config = {
            'robot_actual_sensor_range': robot_actual_sensor_range,
            'n_ants_update': n_ants_update,
            'n_iterations_update': n_iterations_update,
            'alpha_step_choice': alpha_step_choice,
            'beta_step_heuristic': beta_step_heuristic,
            'evaporation_rate_map': evaporation_rate_map,
            'pheromone_min_map': pheromone_min_map,
            'pheromone_max_map': pheromone_max_map,
            'q_path_deposit_factor': q_path_deposit_factor,
            'ant_max_steps': ant_max_steps,
            'max_pheromone_nav_steps': max_pheromone_nav_steps,
            'eta_weight_to_unknown': eta_weight_to_unknown,
            'eta_weight_to_frontiers_centroid': eta_weight_to_frontiers_centroid,
        }
        self.visualize_ants_callback = visualize_ants_callback

    def _update_pheromone_map_for_obstacles(self, pheromone_map, known_map):
        obstacle_mask = (known_map == OBSTACLE)
        pheromone_map[obstacle_mask] = self.config['pheromone_min_map']

    def _get_local_step_heuristic(self, from_r, from_c, to_r, to_c, known_map, frontiers_list):
        eta = 1.0
        if self.environment.is_within_bounds(to_r, to_c) and known_map[to_r, to_c] == UNKNOWN:
            eta += self.config['eta_weight_to_unknown']
        if frontiers_list and self.config['eta_weight_to_frontiers_centroid'] > 0:
            avg_fr, avg_fc = np.mean(frontiers_list, axis=0)
            dist_from = manhattan_heuristic((from_r, from_c), (avg_fr, avg_fc))
            dist_to = manhattan_heuristic((to_r, to_c), (avg_fr, avg_fc))
            if dist_to < dist_from:
                eta += self.config['eta_weight_to_frontiers_centroid']
        return max(0.1, eta)

    def update_shared_pheromones(self, all_robot_positions, known_map, shared_pheromone_map):
        """
        运行蚂蚁模拟来更新共享信息素地图。
        蚂蚁从所有机器人的当前位置开始探索。
        """
        frontiers = self.find_frontiers(known_map)
        if not frontiers: return

        # 全局蒸发一次
        shared_pheromone_map *= (1.0 - self.config['evaporation_rate_map'])
        
        # 蚂蚁从所有机器人位置出发，增加探索广度
        combined_ant_paths_data = []
        for robot_pos in all_robot_positions:
            for _ in range(self.config['n_ants_update']):
                ant_r, ant_c = robot_pos
                path = [(ant_r, ant_c)]
                for _ in range(self.config['ant_max_steps']):
                    possible_moves = []
                    for dr, dc in [(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0)]:
                        nr, nc = ant_r + dr, ant_c + dc
                        if not self.environment.is_within_bounds(nr, nc) or \
                           known_map[nr, nc] == OBSTACLE or (nr, nc) in path[-2:]:
                            continue
                        tau = shared_pheromone_map[nr, nc] ** self.config['alpha_step_choice']
                        eta = self._get_local_step_heuristic(ant_r, ant_c, nr, nc, known_map, frontiers) ** self.config['beta_step_heuristic']
                        possible_moves.append({'pos': (nr, nc), 'score': tau * eta})
                    if not possible_moves: break
                    total_score = sum(m['score'] for m in possible_moves)
                    if total_score <= 1e-9: chosen_pos = random.choice(possible_moves)['pos']
                    else:
                        rand_val = random.random() * total_score
                        current_sum = 0
                        for move in possible_moves:
                            current_sum += move['score']
                            if current_sum >= rand_val: chosen_pos = move['pos']; break
                    ant_r, ant_c = chosen_pos
                    path.append((ant_r, ant_c))
                
                quality = len(set(path))
                combined_ant_paths_data.append({'path': path, 'quality': quality})

        # 所有蚂蚁路径完成后，统一进行信息素沉积
        for ant_data in combined_ant_paths_data:
            path, quality = ant_data['path'], ant_data['quality']
            if len(path) > 1:
                deposit = self.config['q_path_deposit_factor'] * (quality / len(path))
                for r, c in path:
                    shared_pheromone_map[r, c] += deposit
        
        # 应用最大/最小值约束
        shared_pheromone_map.clip(self.config['pheromone_min_map'], self.config['pheromone_max_map'], out=shared_pheromone_map)
        self._update_pheromone_map_for_obstacles(shared_pheromone_map, known_map)

    def _navigate_by_pheromones(self, robot_pos, known_map, pheromone_map, reserved_targets):
        r, c = robot_pos
        nav_path = {(r, c)}
        for _ in range(self.config['max_pheromone_nav_steps']):
            is_frontier = False
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if self.environment.is_within_bounds(nr, nc) and known_map[nr, nc] == UNKNOWN:
                    is_frontier = True
                    break
            if is_frontier and (r, c) not in reserved_targets.values():
                return (r, c)
            best_next, max_pher = None, -1.0
            for dr, dc in [(0,1), (0,-1), (1,0), (-1,0)]:
                nr, nc = r + dr, c + dc
                if self.environment.is_within_bounds(nr, nc) and \
                   known_map[nr, nc] != OBSTACLE and (nr, nc) not in nav_path:
                    if pheromone_map[nr, nc] > max_pher:
                        max_pher, best_next = pheromone_map[nr, nc], (nr, nc)
            if best_next is None: return None
            r, c = best_next
            nav_path.add((r, c))
        return None

    def plan_next_action(self, robot_id, robot_pos, known_map, shared_pheromone_map, reserved_targets, **kwargs):
        """
        轻量级规划：仅使用现有信息素地图进行导航，不运行蚂蚁模拟。
        """
        # 1. 尝试根据当前信息素导航
        target = self._navigate_by_pheromones(robot_pos, known_map, shared_pheromone_map, reserved_targets)
        
        if target:
            path = self._is_reachable_and_get_path(robot_pos, target, known_map)
            if path:
                reserved_targets[robot_id] = target
                return target, path
        
        # 2. 如果导航失败，使用回退计划（寻找最近的可用边界点）
        fallback_target, fallback_path = self._final_fallback_plan(robot_pos, known_map, reserved_targets)
        
        if fallback_target:
            reserved_targets[robot_id] = fallback_target

        return fallback_target, fallback_path