from .base_planner import BasePlanner
import numpy as np

class FBEPlanner(BasePlanner):
    def __init__(self, environment):
        super().__init__(environment)

    def plan_next_action(self, robot_pos, known_map, **kwargs): 
        chosen_frontier, path_to_chosen_frontier = self._final_fallback_plan(robot_pos, known_map)
            
        return chosen_frontier, path_to_chosen_frontier