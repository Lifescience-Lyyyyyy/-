import numpy as np
try:
    from environment import OBSTACLE
except ImportError:
    OBSTACLE = 2
    print("Warning (geometry_utils.py): Could not import OBSTACLE. Using default value 2.")


def bresenham_line_algo(r0, c0, r1, c1):
    """Bresenham's line algorithm generates integer coordinates for a line."""
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    
    curr_r, curr_c = r0, c0

    max_iter = dr + dc + 2
    count = 0
    while count < max_iter:
        points.append((curr_r, curr_c))
        if curr_r == r1 and curr_c == c1:
            break
        
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            curr_r += sr
        if e2 < dr:
            err += dr
            curr_c += sc
        count += 1
        
    return points

def check_line_of_sight(start_r, start_c, target_r, target_c, map_for_los_check):
    """
    检查在给定的地图上，从起点到目标点是否有清晰的视线。
    """
    if (start_r, start_c) == (target_r, target_c):
        return True

    line_points = bresenham_line_algo(start_r, start_c, target_r, target_c)
    map_height, map_width = map_for_los_check.shape

    # 检查路径上的中间点（不包括起点和终点）是否有障碍物
    for i in range(1, len(line_points) - 1): 
        pr, pc = line_points[i]
        
        if not (0 <= pr < map_height and 0 <= pc < map_width):
            return False # 视线路径超出了地图边界
            
        if map_for_los_check[pr, pc] == OBSTACLE:
            return False # 视线被已知障碍物阻挡
            
    return True