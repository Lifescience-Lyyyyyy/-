# batch_runner_multi_agent.py
import subprocess
import os
import itertools
from datetime import datetime
import shlex

# --- Configuration ---
PYTHON_EXECUTABLE = "python3"
MAIN_SIM_SCRIPT = "main_multi_agent_sim.py"

# --- Parameters to Iterate Over ---
MAP_SIZES = ["50x50", "70x70"]
OBSTACLE_PERCENTAGES = ["0.20", "0.35"]
MAP_TYPES = ["random", "deceptive_hallway"]
NUM_ROBOTS_LIST = [1, 3, 5] # 对比不同数量的机器人

# --- Fixed Parameters for All Runs ---
BASE_OUTPUT_DIR = "simulation_batch_multi_agent_output"
MAX_SIM_STEPS = 2000
ROBOT_SENSOR_RANGE = 7
EXPLORATION_GOAL_PERCENTAGE = 0.95
MASTER_SEED_START = 1000

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_run_dir = os.path.join(BASE_OUTPUT_DIR, f"run_{timestamp}")
    os.makedirs(batch_run_dir, exist_ok=True)
    
    print(f"Starting multi-agent batch run. Results will be saved in: {batch_run_dir}")

    # 生成所有参数组合
    combinations = list(itertools.product(
        MAP_SIZES,
        OBSTACLE_PERCENTAGES,
        MAP_TYPES,
        NUM_ROBOTS_LIST
    ))

    current_seed = MASTER_SEED_START
    total_configs = len(combinations)

    for i, (map_size, obs_perc, map_type, num_robots) in enumerate(combinations):
        width, height = map(int, map_size.split('x'))
        
        # 为每个配置创建唯一的输出目录
        config_name = f"size_{map_size}_obs_{obs_perc}_type_{map_type}_robots_{num_robots}"
        output_dir = os.path.join(batch_run_dir, config_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n--- [{i+1}/{total_configs}] Running Config: {config_name} ---")

        # 构建命令行参数 (注意：在批处理中通常禁用可视化)
        command = [
            PYTHON_EXECUTABLE,
            MAIN_SIM_SCRIPT,
            # 这里需要修改 main_multi_agent_sim.py 以接受命令行参数
            # 为简化，当前版本直接在脚本中配置。
            # 若要实现，main_multi_agent_sim.py需添加 argparse 库来解析这些参数。
            # 例如： f"--map_width={width}", f"--num_robots={num_robots}", etc.
            # 假设 main 脚本被修改为接受参数，下面是示例：
            # "--map_width", str(width),
            # "--map_height", str(height),
            # "--obstacle_percentage", str(obs_perc),
            # "--map_type", map_type,
            # "--num_robots", str(num_robots),
            # "--max_simulation_steps", str(MAX_SIM_STEPS),
            # "--robot_sensor_range", str(ROBOT_SENSOR_RANGE),
            # "--exploration_goal_percentage", str(EXPLORATION_GOAL_PERCENTAGE),
            # "--random_seed", str(current_seed),
            # "--no-visualize" # 添加一个标志来禁用pygame
        ]
        
        # 由于当前 main 脚本是硬编码配置，我们暂时只打印将要运行的命令
        # 在实际使用中，你需要修改 main_multi_agent_sim.py 以支持 argparse
        print("Command (Example - requires argparse in main):")
        print(" ".join(command))
        
        # 实际运行示例 (需要修改 main 脚本)
        # log_file_path = os.path.join(output_dir, "simulation.log")
        # with open(log_file_path, 'w') as log_file:
        #     process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.PIPE)
        #     stderr_output = process.communicate()[1]
        #     if process.returncode != 0:
        #         print(f"  ERROR running config. Check log and stderr below.")
        #         print(f"  Stderr: {stderr_output.decode()}")
        #     else:
        #         print(f"  SUCCESS. Log saved to {log_file_path}")

        current_seed += 1

    print("\n--- Batch run finished. ---")

if __name__ == "__main__":
    print("NOTE: This batch runner is a template.")
    print("The 'main_multi_agent_sim.py' script must be modified to accept command-line arguments (using argparse) for this to run automatically.")
    # main() # 取消注释以运行（在修改主脚本后）