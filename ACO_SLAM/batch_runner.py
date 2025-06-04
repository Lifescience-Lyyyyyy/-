import subprocess
import time
import os
import itertools
from datetime import datetime
import shlex

# --- Configuration ---
PYTHON_EXECUTABLE = "python3"  # Or your Python interpreter path
MAIN_SIM_SCRIPT = "main_simulation.py" # Your main simulation script name

# Define parameter ranges to iterate over
MAP_SIZES_TO_RUN_STR = ["50x50","100x100","200x200"]
OBSTACLE_PERCENTAGES_TO_RUN_STR = ["0.15","0.30","0.45"]
MAP_TYPES_TO_RUN_STR = ["deceptive_hallway"] #, "random"
PLANNERS_TO_RUN_STR = ["FBE", "ACO", "IGE", "URE"] # Select planners to test
NUM_RUNS_PER_CONFIG = 1 # Number of runs per configuration
MASTER_SEED_START = 420 # Initial master seed, will be incremented for each config set
BASE_OUTPUT_DIR = "simulation_runs_batch_output_en" # Output directory for this batch runner
SCREENSHOT_INTERVAL = 300  # If > 0, main_simulation.py will attempt screenshots.
                           # This depends on an available display service in the execution environment.
                           # Set to 0 for no screenshots.
VIZ_ANTS_FOR_BATCH = False # Usually False for batch processing, unless specifically viewing in a window

TMUX_SESSION_NAME_PREFIX = "sim_batch_run_en" # tmux session name prefix

def check_tmux_session(session_name):
    """Checks if the specified tmux session exists."""
    try:
        subprocess.check_output(["tmux", "has-session", "-t", session_name], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False
    except FileNotFoundError:
        print("Error: tmux command not found. Please ensure tmux is installed and in PATH.")
        exit(1)


def create_tmux_session(session_name):
    """Creates a new detached tmux session."""
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name], check=True)
        print(f"Tmux session '{session_name}' created.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to create tmux session '{session_name}': {e}")
        exit(1)
    except FileNotFoundError:
        print("Error: tmux command not found. Please ensure tmux is installed and in PATH.")
        exit(1)


def send_command_to_tmux(session_name, command_str, window_index=0, pane_index=0):
    """Sends a command string to the specified tmux session, window, and pane."""
    target = f"{session_name}:{window_index}.{pane_index}"
    try:
        subprocess.run(["tmux", "send-keys", "-t", target, command_str, "C-m"], check=True)
        time.sleep(0.2) 
    except subprocess.CalledProcessError as e:
        print(f"Failed to send command '{command_str[:70]}...' to tmux session '{target}': {e}")
    except FileNotFoundError:
        print("Error: tmux command not found. Cannot send command.")


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

    param_combinations = list(itertools.product(
        MAP_SIZES_TO_RUN_STR,
        OBSTACLE_PERCENTAGES_TO_RUN_STR,
        MAP_TYPES_TO_RUN_STR,
        PLANNERS_TO_RUN_STR
    ))

    master_seed_current = MASTER_SEED_START
    total_configs = len(param_combinations)
    config_count = 0
    
    initial_message = f"Batch processing started at: $(date)"
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(initial_message)}")

    session_log_file = os.path.join(log_dir, f"tmux_session_log_{timestamp}.txt")
    redirect_command = f"exec > >(tee -a {shlex.quote(session_log_file)}) 2>&1"
    send_command_to_tmux(tmux_session_name, redirect_command)
    log_message_to_tmux = f"All tmux output will be logged to: {session_log_file}"
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(log_message_to_tmux)}")


    for map_size_str, obs_perc_str, map_type_str, planner_str in param_combinations:
        config_count += 1
        
        config_info_message_tmux = (f"\nPreparing configuration {config_count}/{total_configs}: "
                               f"Size={map_size_str}, Obs%={obs_perc_str}, MapT={map_type_str}, Planner={planner_str}")
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(config_info_message_tmux)}")

        size_folder = map_size_str.replace('x','by')
        obs_folder = f"Obs{obs_perc_str.replace('.', '_')}"
        config_specific_output_dir = os.path.join(BASE_OUTPUT_DIR, size_folder, obs_folder, map_type_str, planner_str)

        python_command_parts = [
            PYTHON_EXECUTABLE,
            MAIN_SIM_SCRIPT,
            "--map_sizes", map_size_str,
            "--obs_percentages", obs_perc_str,
            "--map_types", map_type_str,
            "--planners", planner_str,
            "--num_runs", str(NUM_RUNS_PER_CONFIG),
            "--master_seed", str(master_seed_current),
            "--base_output_dir", config_specific_output_dir,
            "--screenshot_interval", str(SCREENSHOT_INTERVAL)
        ]

        if SCREENSHOT_INTERVAL <= 0:
            python_command_parts.append("--no_viz")
        
        visualization_is_active_for_main_sim = SCREENSHOT_INTERVAL > 0 or "--no_viz" not in python_command_parts
        if VIZ_ANTS_FOR_BATCH and planner_str == "ACO" and visualization_is_active_for_main_sim:
            python_command_parts.append("--viz_ants")
        
        command_to_run_on_shell = " ".join(shlex.quote(part) for part in python_command_parts)
        
        run_info_message_tmux = (f"--- [{config_count}/{total_configs}] "
                            f"Executing: Planner={planner_str}, Size={map_size_str}, Obs%={obs_perc_str}, MapT={map_type_str} "
                            f"SeedStart={master_seed_current} (Output: {config_specific_output_dir}) ---")
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(run_info_message_tmux)}")
        
        send_command_to_tmux(tmux_session_name, command_to_run_on_shell)
        
        started_message_tmux = f"main_simulation.py for configuration {config_count} has been launched..."
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(started_message_tmux)}")
        
        # master_seed_current += NUM_RUNS_PER_CONFIG
        
        time.sleep(0.2)

    final_message1_tmux = f"\n--- All {total_configs} configuration commands have been sent ---"
    final_message2_tmux = "Simulations are running in the background. The tmux session will remain active."
    final_message3_tmux = f"All tmux output is being logged to: {session_log_file}"
    final_message4_tmux = (f"Once complete, you can check the log file and then manually close this tmux window or session "
                        f"(e.g., tmux kill-session -t {tmux_session_name}).")

    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message1_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message2_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message3_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message4_tmux)}")

    print(f"\nAll configurations scheduled in tmux session '{tmux_session_name}'.")
    print(f"Attach using: tmux attach -t {tmux_session_name}")
    print(f"All output within the tmux session (including stdout/stderr from Python scripts) will be logged to: {session_log_file}")

if __name__ == "__main__":
    main()