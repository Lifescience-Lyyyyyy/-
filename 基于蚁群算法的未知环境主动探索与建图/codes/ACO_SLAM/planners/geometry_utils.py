import numpy as np
try:
    from environment import OBSTACLE
except ImportError:
    OBSTACLE = 2 
    print("Warning (geometry_utils.py): Could not import OBSTACLE from environment. Using default value 2.")


def bresenham_line_algo(r0, c0, r1, c1):
    """Bresenham's line algorithm generates integer coordinates for a line between two points."""
    points = []
    dr_abs = abs(r1 - r0)
    dc_abs = abs(c1 - c0)
    sr_sign = 1 if r0 < r1 else -1
    sc_sign = 1 if c0 < c1 else -1
    
    err = dr_abs - dc_abs
    
    curr_r, curr_c = r0, c0

    max_iterations = dr_abs + dc_abs + 2 
    count = 0

    while count < max_iterations:
        points.append((curr_r, curr_c))
        if curr_r == r1 and curr_c == c1:
            break
        
        e2 = 2 * err
        if e2 > -dc_abs:
            err -= dc_abs
            curr_r += sr_sign
        if e2 < dr_abs:
            err += dc_abs
            curr_c += sc_sign
        count += 1
    
    if points[-1] != (r1,c1) and count >= max_iterations:
        pass


    return points

def check_line_of_sight(start_r, start_c, target_r, target_c, map_for_los_check, obstacle_value=2):
    if start_r == target_r and start_c == target_c:
        return True

    line_points = bresenham_line_algo(start_r, start_c, target_r, target_c)
    map_height, map_width = map_for_los_check.shape

    if not line_points: return False 

    for i in range(1, len(line_points)): 
        pr, pc = line_points[i]
        
        if not (0 <= pr < map_height and 0 <= pc < map_width):
            return False 
            
        if map_for_los_check[pr, pc] == obstacle_value:
            return False 

        if i > 0: 
            prev_r, prev_c = line_points[i-1]
            dr = pr - prev_r
            dc = pc - prev_c
            
            if dr != 0 and dc != 0:
                corner_cell1_r, corner_cell1_c = prev_r + dr, prev_r 
                corner_cell2_r, corner_cell2_c = prev_r, prev_c + dc 

                is_corner1_obstacle = False
                if (0 <= corner_cell1_r < map_height and 0 <= corner_cell1_c < map_width):
                    if map_for_los_check[corner_cell1_r, corner_cell1_c] == obstacle_value:
                        is_corner1_obstacle = True

                is_corner2_obstacle = False
                if (0 <= corner_cell2_r < map_height and 0 <= corner_cell2_c < map_width):
                    if map_for_los_check[corner_cell2_r, corner_cell2_c] == obstacle_value:
                        is_corner2_obstacle = True

                if is_corner1_obstacle and is_corner2_obstacle:
                    return False
    return True