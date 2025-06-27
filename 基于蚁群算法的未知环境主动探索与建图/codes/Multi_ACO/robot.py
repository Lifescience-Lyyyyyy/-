# robot.py
import numpy as np
from environment import FREE, OBSTACLE, UNKNOWN
try:
    from planners.geometry_utils import check_line_of_sight
except ImportError:
    def check_line_of_sight(sr, sc, tr, tc, m):
        print("Warning (robot.py): Using dummy check_line_of_sight.")
        return True

class Robot:
    def __init__(self, start_pos, sensor_range=5, robot_id=0, color=(255, 0, 0)):
        self.pos = np.array(start_pos)
        self.sensor_range = sensor_range
        self.actual_pos_history = [list(start_pos)]
        self.robot_id = robot_id # 新增：机器人ID
        self.color = color       # 新增：用于可视化的颜色

    def sense(self, environment):
        r_rob, c_rob = self.pos
        known_map_snapshot_for_los = np.copy(environment.grid_known)
        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                r_abs_target, c_abs_target = r_rob + dr, c_rob + dc
                if environment.is_within_bounds(r_abs_target, c_abs_target):
                    if check_line_of_sight(r_rob, c_rob, r_abs_target, c_abs_target, known_map_snapshot_for_los):
                        true_state = environment.get_true_map_state(r_abs_target, c_abs_target)
                        environment.update_known_map(r_abs_target, c_abs_target, true_state)

    def attempt_move_one_step(self, next_step_pos, environment):
        nr, nc = next_step_pos
        if not environment.is_within_bounds(nr, nc):
            return False
        if environment.grid_known[nr, nc] == OBSTACLE:
            return False
        prev_pos = list(self.pos)
        self.pos = np.array(next_step_pos)
        if environment.grid_true[self.pos[0], self.pos[1]] == OBSTACLE:
            environment.update_known_map(nr, nc, OBSTACLE) 
            self.pos = np.array(prev_pos)
            return False
        self.actual_pos_history.append(list(self.pos))
        return True

    def get_position(self):
        return tuple(self.pos)