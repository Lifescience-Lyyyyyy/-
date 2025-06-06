# robot.py
import numpy as np
from environment import FREE, OBSTACLE, UNKNOWN

# ---------------------------------------------------------------------------
#  HELPER FUNCTION: BRESENHAM'S LINE ALGORITHM
#  This function calculates all the grid cells that a straight line
#  between two points passes through. It's essential for line-of-sight.
# ---------------------------------------------------------------------------
def bresenham_line(start, end):
    """
    Generate the integer coordinates for a line between two points.
    Args:
        start (tuple): The (row, col) of the start point.
        end (tuple): The (row, col) of the end point.
    Yields:
        (tuple): The (row, col) of each cell along the line.
    """
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    r, c = r0, c0
    r_inc = 1 if r1 > r0 else -1
    c_inc = 1 if c1 > c0 else -1
    
    # The error term determines when to step in the minor direction
    error = dr - dc
    
    points = []
    
    while True:
        points.append((r, c))
        if r == r1 and c == c1:
            break
        
        e2 = 2 * error
        if e2 > -dc:
            error -= dc
            r += r_inc
        if e2 < dr:
            error += dr
            c += c_inc
            
    return points


class Robot:
    def __init__(self, start_pos, sensor_range=5):
        self.pos = np.array(start_pos)
        self.sensor_range = sensor_range
        self.actual_pos_history = [list(start_pos)]

    # ---------------------------------------------------------------------------
    #  MODIFIED SENSE METHOD with Line-of-Sight
    # ---------------------------------------------------------------------------
    def sense(self, environment):
        """
        Senses the environment using a ray-casting model.
        The robot casts rays from its position to the edge of its sensor range.
        The map is revealed along each ray until an obstacle is hit.
        This simulates line-of-sight.
        """
        r_rob, c_rob = self.pos
        
        # The robot always knows its own location is FREE
        environment.update_known_map(r_rob, c_rob, FREE)

        # Iterate through all cells on the perimeter of the sensor's square range
        # and cast a ray to each one.
        for i in range(-self.sensor_range, self.sensor_range + 1):
            # Top and bottom edges of the square
            self._cast_ray_and_update_map((r_rob, c_rob), (r_rob - self.sensor_range, c_rob + i), environment)
            self._cast_ray_and_update_map((r_rob, c_rob), (r_rob + self.sensor_range, c_rob + i), environment)
            # Left and right edges (excluding corners already covered)
            if i > -self.sensor_range and i < self.sensor_range:
                self._cast_ray_and_update_map((r_rob, c_rob), (r_rob + i, c_rob - self.sensor_range), environment)
                self._cast_ray_and_update_map((r_rob, c_rob), (r_rob + i, c_rob + self.sensor_range), environment)

    def _cast_ray_and_update_map(self, start_pos, end_pos, environment):
        """
        Helper function to cast a single ray and update the known map.
        """
        # Get all cells along the line of sight using Bresenham's algorithm
        line_of_sight = bresenham_line(start_pos, end_pos)
        
        for r, c in line_of_sight:
            # Check if the cell is within the circular sensor range for more realism
            # (optional, but makes the explored area look circular)
            if (r - start_pos[0])**2 + (c - start_pos[1])**2 > self.sensor_range**2:
                continue

            # Check if the cell is within the map bounds
            if not environment.is_within_bounds(r, c):
                continue

            # Get the true state of the cell from the environment
            true_state = environment.get_true_map_state(r, c)
            
            # Update the robot's known map with this new information
            environment.update_known_map(r, c, true_state)

            # IMPORTANT: If we hit an obstacle, the ray is blocked.
            # We cannot see anything beyond it, so we stop processing this ray.
            if true_state == OBSTACLE:
                break
                
    def attempt_move_one_step(self, next_step_pos, environment):
        """
        (This method remains unchanged, its logic is still correct)
        """
        nr, nc = next_step_pos
        if not environment.is_within_bounds(nr, nc):
            return False

        # Cannot move into a known obstacle
        if environment.grid_known[nr, nc] == OBSTACLE:
            return False

        # Attempt the move
        prev_pos = list(self.pos)
        self.pos = np.array(next_step_pos)
        
        # Check for collision with an unknown obstacle in the true map
        if environment.grid_true[self.pos[0], self.pos[1]] == OBSTACLE:
            # Move failed, revert position
            self.pos = np.array(prev_pos)
            # The robot just "bumped" into the obstacle, so it now knows about it.
            # This reveal happens on the next sense(), but we can also do it here for immediate feedback.
            environment.update_known_map(nr, nc, OBSTACLE)
            return False

        # Move was successful
        self.actual_pos_history.append(list(self.pos))
        return True

    def get_position(self):
        return tuple(self.pos)