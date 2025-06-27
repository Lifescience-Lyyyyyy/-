import uvicorn
import numpy as np
import traceback
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pathlib import Path
from logic.environment import Environment
from logic.aco import PathPlanningACO, ExplorationEngine
from logic.a_star import a_star_search

FRONTEND_PATH = Path(__file__).resolve().parent.parent / "frontend"
ALT_FRONTEND_PATH = Path(__file__).resolve().parent.parent
app = FastAPI()

def get_start_goal_positions(height, width):
    """A robust function to get valid start and goal positions."""
    print(f"DEBUG: Calculating positions for map size: height={height}, width={width}")
    start_r = height // 2
    goal_r = height // 2
    start_c = 2
    if start_c >= width - 1:
        start_c = 0
    goal_c = width - 3
    if goal_c <= start_c:
        goal_c = width - 1
    return (start_r, start_c), (goal_r, goal_c)

@app.get("/")
async def read_root():
    index_path = FRONTEND_PATH / "index.html"
    if not index_path.is_file():
        index_path = ALT_FRONTEND_PATH / "index.html"
        if not index_path.is_file(): return {"error": "index.html not found"}
    return FileResponse(index_path)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("INFO:     Connection accepted.")
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            params = data.get("params", {}) 
            print(f"INFO:     Received action: '{action}'")

            safe_params = params.copy()
            for key, value in safe_params.items():
                if value is None or value == '': continue
                try:
                    if key in ["map_width", "map_height", "n_ants_pp", "n_iterations_pp", "sensor_range", "n_ants_exp", "n_iterations_exp"]: safe_params[key] = int(value)
                    elif key in ["obstacle_perc", "alpha", "beta", "evaporation"]: safe_params[key] = float(value)
                except (ValueError, TypeError): pass

            if action == "generate_map":
                mode = safe_params.get('mode_switch', 'path_planning')
                width, height = safe_params.get("map_width", 40), safe_params.get("map_height", 30)
                start_pos, goal_pos_guess = get_start_goal_positions(height, width)
                final_goal_pos = goal_pos_guess if mode == 'path_planning' else None
                env = Environment(width, height, start_pos, final_goal_pos, safe_params.get("obstacle_perc", 0.25), safe_params.get("map_type", "random"))
                await websocket.send_json({"type": "map_generated", "grid": env.true_grid.tolist(), "start_pos": env.start_pos, "goal_pos": env.goal_pos, "width": env.width, "height": env.height})
            
            elif action == "from_drawing":
                grid_data = safe_params.get("grid")
                mode = safe_params.get('mode', 'path_planning')
                
                if not isinstance(grid_data, list) or not grid_data:
                    print(f"ERROR: 'from_drawing' received invalid grid data.")
                    continue

                final_grid = np.array(grid_data)
                height, width = final_grid.shape

                start_pos, goal_pos = get_start_goal_positions(height, width)
                final_goal_pos = goal_pos if mode == 'path_planning' else None

                final_grid[start_pos] = 1 # 1 is FREE
                if final_goal_pos:
                    final_grid[final_goal_pos] = 1
                
                await websocket.send_json({"type": "map_generated", "grid": final_grid.tolist(), "start_pos": start_pos, "goal_pos": final_goal_pos, "width": width, "height": height})

            elif action == "run_path_planning" or action == "run_exploration":
                grid_data = safe_params.get("grid"); start_pos_data = safe_params.get("start_pos")
                if not (isinstance(grid_data, list) and grid_data and isinstance(start_pos_data, list)):
                    print(f"ERROR: '{action}' received invalid grid or start_pos. Skipping."); continue
                grid = np.array(grid_data); start_pos = tuple(start_pos_data)

                if action == "run_path_planning":
                    goal_pos_data = safe_params.get("goal_pos")
                    if not isinstance(goal_pos_data, list): print(f"ERROR: 'run_path_planning' received invalid goal_pos. Skipping."); continue
                    goal_pos = tuple(goal_pos_data)
                    planner = PathPlanningACO(grid, start_pos, goal_pos, safe_params, websocket)
                    aco_path = await planner.find_path()
                    final_path, algorithm = (aco_path, "aco") if aco_path else (a_star_search(grid, start_pos, goal_pos), "astar")
                    await websocket.send_json({"type": "planning_end", "path": final_path, "algorithm": algorithm, "pheromone_map": planner.pheromone_map.tolist()})
                else:
                    engine = ExplorationEngine(grid, start_pos, safe_params, websocket); await engine.explore()

    except WebSocketDisconnect: print("INFO:     Client disconnected.")
    except Exception: traceback.print_exc(); await websocket.close(code=1011)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)