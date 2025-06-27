# main_multi_agent_sim.py
import pygame
import numpy as np
import random
from collections import deque

from environment import Environment, UNKNOWN, FREE, OBSTACLE
from robot import Robot
from visualizer import Visualizer
from planners.aco_multi_agent_planner import ACOMultiAgentPlanner

# --- Configuration ---
MAP_WIDTH = 50; MAP_HEIGHT = 50
OBSTACLE_PERCENTAGE = 0.25
MAP_TYPE = "random"
CELL_SIZE = 15
MAX_SIMULATION_STEPS = 3000 # 增加最大步数以确保能完全探索
RANDOM_SEED = 42

NUM_ROBOTS = 3
ROBOT_SENSOR_RANGE = 7
PHEROMONE_UPDATE_INTERVAL = 15

ACO_SHARED_PHEROMONE_CONFIG = {
    'n_ants_update': 8, 'n_iterations_update': 3, 'ant_max_steps': 50,
    'alpha_step_choice': 1.0, 'beta_step_heuristic': 2.5,
    'evaporation_rate_map': 0.05, 'pheromone_min_map': 0.01,
    'pheromone_max_map': 10.0, 'q_path_deposit_factor': 1.5,
    'max_pheromone_nav_steps': 100, 'eta_weight_to_unknown': 3.5,
    'eta_weight_to_frontiers_centroid': 2.0,
}

VISUALIZE = True
SIM_DELAY_MS = 20

def main():
    if RANDOM_SEED is not None:
        np.random.seed(RANDOM_SEED); random.seed(RANDOM_SEED)

    start_positions = []
    while len(start_positions) < NUM_ROBOTS:
        pos = (random.randint(1, MAP_HEIGHT-2), random.randint(1, MAP_WIDTH-2))
        if pos not in start_positions: start_positions.append(pos)
    
    env = Environment(MAP_WIDTH, MAP_HEIGHT, OBSTACLE_PERCENTAGE, MAP_TYPE, robot_start_pos_ref=start_positions)
    visualizer = Visualizer(MAP_WIDTH, MAP_HEIGHT, CELL_SIZE) if VISUALIZE else None
    
    shared_pheromone_map = np.full((MAP_HEIGHT, MAP_WIDTH), 5.0, dtype=float)
    shared_reserved_targets = {}

    robots = []
    robot_colors = [ (255,0,0), (0,0,255), (0,255,0), (255,165,0), (128,0,128) ]
    shared_planner = ACOMultiAgentPlanner(
        environment=env, robot_actual_sensor_range=ROBOT_SENSOR_RANGE, **ACO_SHARED_PHEROMONE_CONFIG
    )
    for i in range(NUM_ROBOTS):
        robot = Robot(start_pos=start_positions[i], sensor_range=ROBOT_SENSOR_RANGE, 
                      robot_id=i, color=robot_colors[i % len(robot_colors)])
        robots.append(robot)

    robot_paths_to_target = [deque() for _ in range(NUM_ROBOTS)]
    
    # (新) 探索停滞检测逻辑
    idle_steps_counter = 0
    MAX_IDLE_STEPS_BEFORE_FORCING_UPDATE = 30 # 如果所有机器人都空闲这么多步，就强制更新

    print("--- Multi-Agent ACO Collaborative Exploration (Full Coverage) ---")
    print("Simulation ends when no reachable frontiers (standard or internal) exist.")

    for step in range(MAX_SIMULATION_STEPS):
        if VISUALIZE and not visualizer.handle_events_simple(): break

        knowledge_updated_this_step = False
        
        # --- 信息素更新 (定时或在停滞时强制) ---
        force_pheromone_update = False
        if idle_steps_counter >= MAX_IDLE_STEPS_BEFORE_FORCING_UPDATE:
            print(f"Step {step}: Robots idle for {idle_steps_counter} steps. Forcing pheromone update.")
            force_pheromone_update = True
            idle_steps_counter = 0

        if (step % PHEROMONE_UPDATE_INTERVAL == 0) or force_pheromone_update:
            all_pos = [r.get_position() for r in robots]
            shared_planner.update_shared_pheromones(all_pos, env.get_known_map_for_planner(), shared_pheromone_map)
            if VISUALIZE: visualizer.is_pheromones_dirty = True

        # --- 规划 & 执行 ---
        shared_reserved_targets.clear()
        robots_are_all_idle = True

        for i, robot in enumerate(robots):
            # 只有在机器人没有任务时才重新规划
            if not robot_paths_to_target[i]:
                target, path = shared_planner.plan_next_action(
                    robot_id=robot.robot_id, robot_pos=robot.get_position(),
                    known_map=env.get_known_map_for_planner(),
                    shared_pheromone_map=shared_pheromone_map,
                    reserved_targets=shared_reserved_targets
                )
                if target and path:
                    robot_paths_to_target[i] = deque(path[1:])
                    shared_reserved_targets[i] = target
                    robots_are_all_idle = False # 至少有一个机器人找到了新任务
            else:
                 robots_are_all_idle = False # 机器人正在执行任务，不算空闲

            # 首次感知
            if step == 0:
                robot.sense(env); knowledge_updated_this_step = True
            
            # 移动
            if robot_paths_to_target[i]:
                next_pos = robot_paths_to_target[i].popleft()
                if robot.attempt_move_one_step(next_pos, env):
                    robot.sense(env); knowledge_updated_this_step = True
                else:
                    robot_paths_to_target[i].clear()
                    robot.sense(env); knowledge_updated_this_step = True
        
        # --- 停滞检测 & 完成度判断 ---
        if robots_are_all_idle:
            idle_steps_counter += 1
            # (新) 最终完成度检查：如果所有机器人都空闲，并且地图上确实没有任何可达的目标了
            # 我们需要检查整个地图是否还有任何边界点。
            # 这里的检查是为了确认探索是否真的完成了。
            all_frontiers = shared_planner.find_frontiers(env.grid_known)
            all_internal_frontiers = shared_planner.find_internal_frontiers(env.grid_known)
            
            if not all_frontiers and not all_internal_frontiers:
                print(f"\n--- Exploration Complete ---")
                print(f"No more standard or internal frontiers found at step {step+1}.")
                break
        else:
            idle_steps_counter = 0 # 只要有机器人移动或规划，就重置空闲计数器

        # --- 可视化 ---
        if VISUALIZE:
            if knowledge_updated_this_step:
                visualizer.is_background_dirty = True

            total_explorable = env.get_total_explorable_area()
            explored_area = env.get_explored_area()
            explored_percentage = explored_area / total_explorable if total_explorable > 0 else 1.0

            texts_to_draw = [
                {'text': f"Step: {step+1}/{MAX_SIMULATION_STEPS}", 'pos': (10, 10)},
                {'text': f"Explored: {explored_percentage:.2%}", 'pos': (10, 30)},
                {'text': f"Idle Counter: {idle_steps_counter}", 'pos': (10, 50)},
            ]

            visualizer.draw_simulation_state(
                known_map=env.grid_known, pheromone_map=shared_pheromone_map,
                robots=robots, paths=robot_paths_to_target,
                targets=shared_reserved_targets, texts=texts_to_draw
            )
            pygame.time.wait(SIM_DELAY_MS)

    # --- 模拟结束 ---
    if step == MAX_SIMULATION_STEPS - 1: print(f"\nMax steps reached.")
    
    total_explorable = env.get_total_explorable_area()
    final_explored_area = env.get_explored_area()
    final_percentage = final_explored_area / total_explorable if total_explorable > 0 else 1.0
    print(f"Final exploration of FREE space: {final_percentage:.2%}")
    
    if VISUALIZE:
        while visualizer.handle_events_simple(): pygame.time.wait(100)
        visualizer.quit()

if __name__ == '__main__':
    # 确保文件名和类名与您的项目匹配
    try:
        from visualizer import Visualizer
        from planners.aco_multi_agent_planner import ACOMultiAgentPlanner
        main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure your file names match the imports in 'main_multi_agent_sim.py'")