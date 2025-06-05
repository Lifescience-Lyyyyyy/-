import subprocess
import time
import os
import itertools
from datetime import datetime
import shlex

# --- Configuration ---
PYTHON_EXECUTABLE = "python3"
MAIN_SIM_SCRIPT = "main_simulation.py"

# Define parameter ranges to iterate over
MAP_SIZES_TO_RUN_STR = ["50x50", "100x100"] # Add "200x200" for larger maps
OBSTACLE_PERCENTAGES_TO_RUN_STR = ["0.15", "0.30"] # Add "0.45"
MAP_TYPES_TO_RUN_STR = ["random", "deceptive_hallway"]
# This list defines which planners will be *compared on a single plot* by each main_simulation.py call
PLANNERS_TO_COMPARE_IN_ONE_CALL_STR = "FBE,ACO,IGE,URE" 

NUM_RUNS_PER_PLANNER_PER_CONFIG = 1 # How many times each planner runs for a given map config
MASTER_SEED_START = 420
BASE_OUTPUT_DIR = "simulation_runs_final_batch"
SCREENSHOT_INTERVAL = 0  # Set to e.g., 300 for screenshots. Relies on display.
VIZ_ANTS_FOR_BATCH = False # ACO ant visualization

TMUX_SESSION_NAME_PREFIX = "sim_final_batch"

def check_tmux_session(session_name):
    try:
        subprocess.check_output(["tmux", "has-session", "-t", session_name], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("Error: tmux command not found. Please ensure tmux is installed and in PATH.")
        exit(1)

def create_tmux_session(session_name):
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name], check=True)
        print(f"Tmux session '{session_name}' created.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create tmux session '{session_name}': {e}")
        exit(1)
    except FileNotFoundError:
        print("Error: tmux command not found.")
        exit(1)

def send_command_to_tmux(session_name, command_str, window_index=0, pane_index=0):
    target = f"{session_name}:{window_index}.{pane_index}"
    try:
        subprocess.run(["tmux", "send-keys", "-t", target, command_str, "C-m"], check=True)
        time.sleep(0.2) 
    except subprocess.CalledProcessError as e:
        print(f"Failed to send command '{command_str[:70]}...' to tmux session '{target}': {e}")
    except FileNotFoundError:
        print("Error: tmux command not found.")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmux_session_name = f"{TMUX_SESSION_NAME_PREFIX}_{timestamp}"

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    log_dir = os.path.join(BASE_OUTPUT_DIR, "_tmux_logs_py")
    os.makedirs(log_dir, exist_ok=True)
    
    if not check_tmux_session(tmux_session_name):
        create_tmux_session(tmux_session_name)
    else:
        print(f"Warning: Tmux session '{tmux_session_name}' already exists. Will attempt to use it.")

    print(f"Simulations will run in tmux session '{tmux_session_name}'.")
    print(f"Results will be saved in '{BASE_OUTPUT_DIR}'.")
    print(f"Attach to session: tmux attach -t {tmux_session_name}")

    # The outer loop now iterates through map configurations.
    # main_simulation.py will be called once per map configuration,
    # and it will internally run all planners specified in PLANNERS_TO_COMPARE_IN_ONE_CALL_STR.
    map_config_combinations = list(itertools.product(
        MAP_SIZES_TO_RUN_STR,
        OBSTACLE_PERCENTAGES_TO_RUN_STR,
        MAP_TYPES_TO_RUN_STR
    ))

    master_seed_current = MASTER_SEED_START
    total_map_configs = len(map_config_combinations)
    map_config_count = 0
    
    initial_message = f"Batch processing for comparison plots started at: $(date)"
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(initial_message)}")

    session_log_file = os.path.join(log_dir, f"tmux_session_log_{timestamp}.txt")
    redirect_command = f"exec > >(tee -a {shlex.quote(session_log_file)}) 2>&1"
    send_command_to_tmux(tmux_session_name, redirect_command)
    log_message_to_tmux = f"All tmux output will be logged to: {session_log_file}"
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(log_message_to_tmux)}")

    for map_size_str, obs_perc_str, map_type_str in map_config_combinations:
        map_config_count += 1
        
        config_info_message_tmux = (f"\nPreparing map configuration {map_config_count}/{total_map_configs}: "
                               f"Size={map_size_str}, Obs%={obs_perc_str}, MapT={map_type_str}")
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(config_info_message_tmux)}")

        size_folder = map_size_str.replace('x','by')
        obs_folder = f"Obs{obs_perc_str.replace('.', '_')}"
        # Output directory for this specific map configuration.
        # main_simulation.py will place its summary CSV and comparison plot here.
        # Screenshots (if any) will go into a 'screenshots' subdirectory created by main_simulation.py
        # within this directory.
        map_config_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, size_folder, obs_folder, map_type_str)

        python_command_parts = [
            PYTHON_EXECUTABLE,
            MAIN_SIM_SCRIPT,
            "--map_sizes", map_size_str,
            "--obs_percentages", obs_perc_str,
            "--map_types", map_type_str,
            "--planners", PLANNERS_TO_COMPARE_IN_ONE_CALL_STR, # Pass all planners to compare
            "--num_runs", str(NUM_RUNS_PER_PLANNER_PER_CONFIG),
            "--master_seed", str(master_seed_current),
            "--base_output_dir", map_config_specific_output_dir, # main_sim output goes here
            "--screenshot_interval", str(SCREENSHOT_INTERVAL)
            # Add other args for main_simulation.py if needed (e.g., --max_steps, --sensor_range)
            # These will use defaults in main_simulation.py if not specified here
        ]

        if SCREENSHOT_INTERVAL <= 0:
            python_command_parts.append("--no_viz")
        
        # Ant visualization control
        # It's active if VIZ_ANTS_FOR_BATCH is True, "ACO" is among planners, and visualization isn't fully disabled
        visualization_will_be_attempted_by_main_sim = SCREENSHOT_INTERVAL > 0 or "--no_viz" not in python_command_parts
        if VIZ_ANTS_FOR_BATCH and "ACO" in PLANNERS_TO_COMPARE_IN_ONE_CALL_STR and visualization_will_be_attempted_by_main_sim:
            python_command_parts.append("--viz_ants")
        
        command_to_run_on_shell = " ".join(shlex.quote(part) for part in python_command_parts)
        
        run_info_message_tmux = (f"--- [{map_config_count}/{total_map_configs}] "
                            f"Executing for Map Config: Size={map_size_str}, Obs%={obs_perc_str}, MapT={map_type_str} "
                            f"(Planners: {PLANNERS_TO_COMPARE_IN_ONE_CALL_STR}) "
                            f"SeedStart={master_seed_current} (Output to: {map_config_specific_output_dir}) ---")
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(run_info_message_tmux)}")
        
        send_command_to_tmux(tmux_session_name, command_to_run_on_shell)
        
        started_message_tmux = f"main_simulation.py for map configuration {map_config_count} has been launched..."
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(started_message_tmux)}")
        
        # Increment master seed for the *next map configuration*.
        # main_simulation.py itself will handle seed increments for its internal num_runs loop for each planner.
        # To ensure truly distinct seeds for each (map_config, planner, run_index) triplet,
        # a more robust seed generation might be needed if main_simulation.py doesn't sufficiently vary it.
        # Current main_simulation.py: current_env_seed_iter = env_master_seed_final + i_run
        # So, for the next map config, we need to advance beyond all seeds used by the current main.py call.
        num_planners_in_call = len(PLANNERS_TO_COMPARE_IN_ONE_CALL_STR.split(','))
        master_seed_current += NUM_RUNS_PER_PLANNER_PER_CONFIG * num_planners_in_call
        
        time.sleep(0.5)

    final_message1_tmux = f"\n--- All {total_map_configs} map configuration commands have been sent ---"
    final_message2_tmux = "Simulations are running in the background. The tmux session will remain active."
    final_message3_tmux = f"All tmux output is being logged to: {session_log_file}"
    final_message4_tmux = (f"Once complete, check logs and output directories. Close session with: tmux kill-session -t {tmux_session_name}")

    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message1_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message2_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message3_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message4_tmux)}")

    print(f"\nAll map configurations scheduled in tmux session '{tmux_session_name}'.")
    print(f"Each call to main_simulation.py will compare planners: {PLANNERS_TO_COMPARE_IN_ONE_CALL_STR}")
    print(f"Attach using: tmux attach -t {tmux_session_name}")
    print(f"Tmux session output logged to: {session_log_file}")

if __name__ == "__main__":
    main()