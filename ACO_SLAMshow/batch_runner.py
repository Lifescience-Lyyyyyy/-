import subprocess
import time
import os
import itertools
from datetime import datetime
import shlex

# --- Configuration ---
PYTHON_EXECUTABLE = "python3"
MAIN_SIM_SCRIPT = "main_simulation.py"

MAP_SIZES_TO_RUN_STR = ["50x50", "100x100"] #, "200x200"]
OBSTACLE_PERCENTAGES_TO_RUN_STR = ["0.15", "0.30"] #, "0.45"]
MAP_TYPES_TO_RUN_STR = ["random", "deceptive_hallway"]
# This string is passed to main_simulation.py, which will iterate through these planners
PLANNERS_TO_COMPARE_IN_ONE_MAIN_CALL = "FBE,ACO,IGE,URE" 

NUM_RUNS_PER_PLANNER_IN_MAIN_CALL = 1 # main_simulation.py will run each planner this many times
MASTER_SEED_START_BATCH = 420 # Master seed for the entire batch
BASE_OUTPUT_DIR_BATCH = "simulation_batch_final_output" # Base for all batch runs
SCREENSHOT_INTERVAL_BATCH = 0  # e.g., 300. If >0, visualization must be attempted.
VIZ_ANTS_FOR_BATCH = False

TMUX_SESSION_NAME_PREFIX_BATCH = "sim_batch_compare"

def check_tmux_session(session_name):
    try:
        subprocess.check_output(["tmux", "has-session", "-t", session_name], stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError: return False
    except FileNotFoundError: print("Error: tmux not found."); exit(1)

def create_tmux_session(session_name):
    try:
        subprocess.run(["tmux", "new-session", "-d", "-s", session_name], check=True)
        print(f"Tmux session '{session_name}' created.")
    except Exception as e: print(f"Failed to create tmux session '{session_name}': {e}"); exit(1)

def send_command_to_tmux(session_name, command_str, window_index=0, pane_index=0):
    target = f"{session_name}:{window_index}.{pane_index}"
    try:
        subprocess.run(["tmux", "send-keys", "-t", target, command_str, "C-m"], check=True)
        time.sleep(0.2) 
    except Exception as e: print(f"Failed to send command to tmux '{target}': {e}")

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmux_session_name = f"{TMUX_SESSION_NAME_PREFIX_BATCH}_{timestamp}"

    os.makedirs(BASE_OUTPUT_DIR_BATCH, exist_ok=True)
    log_dir = os.path.join(BASE_OUTPUT_DIR_BATCH, "_tmux_logs_py")
    os.makedirs(log_dir, exist_ok=True)
    
    if not check_tmux_session(tmux_session_name):
        create_tmux_session(tmux_session_name)
    else:
        print(f"Warning: Tmux session '{tmux_session_name}' already exists.")

    print(f"Simulations will run in tmux session '{tmux_session_name}'.")
    print(f"Results will be saved under '{BASE_OUTPUT_DIR_BATCH}'.")
    print(f"Attach to session: tmux attach -t {tmux_session_name}")

    map_config_combinations = list(itertools.product(
        MAP_SIZES_TO_RUN_STR,
        OBSTACLE_PERCENTAGES_TO_RUN_STR,
        MAP_TYPES_TO_RUN_STR
    ))

    master_seed_for_this_main_py_call = MASTER_SEED_START_BATCH
    total_map_configs = len(map_config_combinations)
    map_config_count = 0
    
    initial_message = f"Batch processing started at: $(date)"
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
        # This is the directory for one map configuration, main_simulation.py will output its summary and plot here.
        current_config_output_dir_for_main_py = os.path.join(BASE_OUTPUT_DIR_BATCH, size_folder, obs_folder, map_type_str)
        # os.makedirs(current_config_output_dir_for_main_py, exist_ok=True) # main.py will make it

        python_command_parts = [
            PYTHON_EXECUTABLE,
            MAIN_SIM_SCRIPT,
            "--map_sizes", map_size_str,             # main.py will use this single size
            "--obs_percentages", obs_perc_str,       # main.py will use this single obs rate
            "--map_types", map_type_str,             # main.py will use this single map type
            "--planners", PLANNERS_TO_COMPARE_IN_ONE_MAIN_CALL, # main.py will loop through these
            "--num_runs", str(NUM_RUNS_PER_PLANNER_IN_MAIN_CALL),
            "--master_seed", str(master_seed_for_this_main_py_call),
            "--base_output_dir", current_config_output_dir_for_main_py,
            "--screenshot_interval", str(SCREENSHOT_INTERVAL_BATCH)
            # Add other general args for main_simulation.py here if they are constant for all batch runs
            # e.g., --robot_sensor_range, --max_steps_override, --cell_size etc.
            # Or, define them at the top of batch_runner and pass them.
            # Example:
            # "--robot_sensor_range", str(ROBOT_SENSOR_RANGE_DEFAULT), # from main.py
            # "--max_steps_override", str(MAX_PHYSICAL_STEPS_DEFAULT_FALLBACK) # from main.py
        ]

        if SCREENSHOT_INTERVAL_BATCH <= 0:
            python_command_parts.append("--no_viz")
        
        visualization_will_be_attempted = SCREENSHOT_INTERVAL_BATCH > 0 or "--no_viz" not in python_command_parts
        if VIZ_ANTS_FOR_BATCH and "ACO" in PLANNERS_TO_COMPARE_IN_ONE_MAIN_CALL and visualization_will_be_attempted:
            python_command_parts.append("--viz_ants")
        
        command_to_run_on_shell = " ".join(shlex.quote(part) for part in python_command_parts)
        
        run_info_message_tmux = (f"--- [{map_config_count}/{total_map_configs}] "
                            f"Executing for Map Config: Size={map_size_str}, Obs%={obs_perc_str}, MapT={map_type_str} "
                            f"(Planners: {PLANNERS_TO_COMPARE_IN_ONE_MAIN_CALL}) "
                            f"Seed for this main.py call: {master_seed_for_this_main_py_call} "
                            f"(Output to: {current_config_output_dir_for_main_py}) ---")
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(run_info_message_tmux)}")
        send_command_to_tmux(tmux_session_name, command_to_run_on_shell)
        
        started_message_tmux = f"main_simulation.py for map configuration {map_config_count} launched..."
        send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(started_message_tmux)}")
        
        # Increment master seed for the *next call to main_simulation.py*.
        # main_simulation.py will handle its internal loop of num_runs with this as start.
        num_planners_in_this_call = len(PLANNERS_TO_COMPARE_IN_ONE_MAIN_CALL.split(','))
        master_seed_for_this_main_py_call += NUM_RUNS_PER_PLANNER_IN_MAIN_CALL * num_planners_in_this_call
        
        time.sleep(0.5)

    # ... (final messages: final_message1_tmux, etc. from previous version) ...
    final_message1_tmux = f"\n--- All {total_map_configs} map configuration commands have been sent ---"
    final_message2_tmux = "Simulations are running in the background. The tmux session will remain active."
    final_message3_tmux = f"All tmux output is being logged to: {session_log_file}"
    final_message4_tmux = (f"Once complete, check logs and output directories. Close session with: tmux kill-session -t {tmux_session_name}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message1_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message2_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message3_tmux)}")
    send_command_to_tmux(tmux_session_name, f"echo {shlex.quote(final_message4_tmux)}")

    print(f"\nAll map configurations scheduled in tmux session '{tmux_session_name}'.")
    print(f"Each call to main_simulation.py will compare planners: {PLANNERS_TO_COMPARE_IN_ONE_MAIN_CALL}")
    print(f"Attach using: tmux attach -t {tmux_session_name}")
    print(f"Tmux session output logged to: {session_log_file}")

if __name__ == "__main__":
    main()