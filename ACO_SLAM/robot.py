# robot.py
import numpy as np
from environment import FREE, OBSTACLE, UNKNOWN
# from planners.pathfinding import a_star_search # No longer directly called by robot for full path

class Robot:
    def __init__(self, start_pos, sensor_range=5):
        self.pos = np.array(start_pos)
        self.sensor_range = sensor_range
        # self.path_taken_visual = [] # This will be populated by main_simulation now
        self.actual_pos_history = [list(start_pos)] # For drawing the actual robot trail

    def sense(self, environment):
        r_rob, c_rob = self.pos
        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                # if dr**2 + dc**2 > self.sensor_range**2: continue # Optional circular sensor
                r, c = r_rob + dr, c_rob + dc
                if environment.is_within_bounds(r, c):
                    true_state = environment.get_true_map_state(r, c)
                    environment.update_known_map(r, c, true_state)

    def attempt_move_one_step(self, next_step_pos, environment):
        """
        Attempts to move one step to next_step_pos.
        Assumes next_step_pos is adjacent.
        Robot senses *before* deciding if the move is valid based on *current* knowledge.
        Then, if it moves, it will be into what it *thought* was free/unknown.
        The environment then dictates the true outcome if it hits an unknown obstacle.

        This function should be called AFTER sensing at current position.
        It checks if the *known_map* allows movement to next_step_pos.
        """
        nr, nc = next_step_pos
        if not environment.is_within_bounds(nr, nc):
            # print(f"Robot: Attempted move to {next_step_pos} is out of bounds.")
            return False # Cannot move out of bounds

        # Check K NOWN map before moving
        if environment.grid_known[nr, nc] == OBSTACLE:
            # print(f"Robot: Attempted move to known obstacle at {next_step_pos}. Staying put.")
            return False # Cannot move into a known obstacle

        # If it's UNKNOWN or FREE in known_map, attempt the move
        # The "real" collision happens based on grid_true if it was UNKNOWN
        
        # Simulate move
        prev_pos = list(self.pos)
        self.pos = np.array(next_step_pos)
        
        # Check if the new position is actually an obstacle in the true map
        # This simulates hitting an unexpected obstacle
        if environment.grid_true[self.pos[0], self.pos[1]] == OBSTACLE:
            # print(f"Robot: Moved to {self.pos} and hit an undiscovered obstacle!")
            self.pos = np.array(prev_pos) # Move back, robot stays at prev_pos
            # Sense at prev_pos to reveal the obstacle at next_step_pos
            # This requires careful handling of sense timing.
            # For now, assume planner will re-evaluate after this failed attempt.
            # The environment.update_known_map should have been called by sense()
            # at the *new* (obstacle) location if the robot could "poke" it.
            # Let's refine: if it *would* move into an obstacle, it discovers it but doesn't move.
            environment.update_known_map(nr, nc, OBSTACLE) # Reveal the obstacle
            self.pos = np.array(prev_pos) # Stay put
            return False # Move failed due to hitting new obstacle

        # If move is successful (not a known or newly discovered obstacle)
        self.actual_pos_history.append(list(self.pos))
        return True


    def get_position(self):
        return tuple(self.pos)