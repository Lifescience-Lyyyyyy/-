import numpy as np
from environment import FREE, OBSTACLE, UNKNOWN

try:
    from planners.geometry_utils import check_line_of_sight
except ImportError:

    def check_line_of_sight(sr, sc, tr, tc, m):
        print("Warning (robot.py): Using dummy check_line_of_sight. Install/configure geometry_utils correctly.")
        return True 

class Robot:
    def __init__(self, start_pos, sensor_range=5):
        self.pos = np.array(start_pos)
        self.sensor_range = sensor_range
        self.actual_pos_history = [list(start_pos)]

    def sense(self, environment):
        """
        Robot senses its surroundings.
        Updates the known_map based on true_map_state for cells within sensor_range
        AND with a clear line of sight (LoS) from the robot's current position.
        LoS is checked against a snapshot of the known_map taken at the beginning of this sense operation.
        """
        r_rob, c_rob = self.pos
        
        known_map_snapshot_for_los = np.copy(environment.grid_known)

        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):

                r_abs_target, c_abs_target = r_rob + dr, c_rob + dc

                if environment.is_within_bounds(r_abs_target, c_abs_target):
                    if check_line_of_sight(r_rob, c_rob, r_abs_target, c_abs_target, known_map_snapshot_for_los):
                        # If LoS is clear, get the true state of the cell from the environment
                        true_state = environment.get_true_map_state(r_abs_target, c_abs_target)
                        # Update the robot's known map with this true state
                        environment.update_known_map(r_abs_target, c_abs_target, true_state)
                        pass


    def attempt_move_one_step(self, next_step_pos, environment):
        """
        Attempts to move one step to next_step_pos.
        Assumes next_step_pos is adjacent.
        Checks against the known_map before attempting the move.
        If the move is into an UNKNOWN or FREE cell (in known_map),
        the true outcome is determined by grid_true (collision with new obstacle).
        """
        nr, nc = next_step_pos
        if not environment.is_within_bounds(nr, nc):
            return False # Cannot move out of bounds

        # Check KNOWN map before moving
        if environment.grid_known[nr, nc] == OBSTACLE:
            return False # Cannot move into a known obstacle

        # If it's UNKNOWN or FREE in known_map, attempt the move
        prev_pos = list(self.pos)
        self.pos = np.array(next_step_pos)
        
        # Check for collision with an undiscovered obstacle in the true map
        if environment.grid_true[self.pos[0], self.pos[1]] == OBSTACLE:
            # Collision! Robot doesn't actually move.
            # Update known_map to reveal the obstacle at the attempted new cell.
            environment.update_known_map(nr, nc, OBSTACLE) 
            self.pos = np.array(prev_pos) # Robot stays at its previous position
            return False # Move failed

        # Move is successful
        self.actual_pos_history.append(list(self.pos))
        return True


    def get_position(self):
        return tuple(self.pos)