from .base_planner import BasePlanner
import numpy as np

class FBEPlanner(BasePlanner):
    def __init__(self, environment):
        super().__init__(environment)
        # FBE doesn't need backtrack_history like the more complex version we tried
        # It will inherently use the _final_fallback_plan logic

    def plan_next_action(self, robot_pos, known_map, **kwargs): # Added **kwargs
        # FBE's core logic is essentially the final fallback.
        # No need for its own specific fallback beyond what _final_fallback_plan provides
        # if it were to fail (which it shouldn't if find_frontiers or A* works).
        
        # For consistency and to ensure it uses the shared method if we refine it later:
        chosen_frontier, path_to_chosen_frontier = self._final_fallback_plan(robot_pos, known_map)
        
        # if chosen_frontier:
        #     print(f"FBE Selected: {chosen_frontier}")
        # else:
        #     print(f"FBE: No target selected (likely no reachable frontiers).")
            
        return chosen_frontier, path_to_chosen_frontier