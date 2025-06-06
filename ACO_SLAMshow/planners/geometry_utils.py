# planners/geometry_utils.py
import numpy as np
# 从 environment 导入 OBSTACLE。假设 environment.py 在上一级目录或PYTHONPATH中
# 如果 planners 是顶级包，而 environment 是另一个顶级模块，导入方式可能需要调整
# 例如：from ..environment import OBSTACLE (如果 planners 是 environment 的兄弟目录下的包)
# 或者，如果它们都在同一个根目录下运行，可以尝试直接导入
try:
    from environment import OBSTACLE
except ImportError:
    # 作为后备，如果直接从planners目录运行或环境设置不同
    # 这假设 environment.py 中的 OBSTACLE 值是 2
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

    # Limit iterations to prevent infinite loops in edge cases,
    # though Bresenham should terminate. Max path length on grid.
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
    
    # Ensure the last point is always the target if the loop terminated early due to max_iterations
    # (though for Bresenham, it should naturally reach if within reasonable map bounds)
    if points[-1] != (r1,c1) and count >= max_iterations:
         # This indicates an issue or very large distance; Bresenham should typically terminate by condition.
         # For safety, one might add the target, but it's better to understand why it didn't reach.
         # print(f"Warning (bresenham_line_algo): Max iterations reached before target. Line from ({r0},{c0}) to ({r1},{c1})")
         pass


    return points

def check_line_of_sight(start_r, start_c, target_r, target_c, map_for_los_check):
    """
    Checks if there is a line of sight from (start_r, start_c) to (target_r, target_c)
    based on the provided map_for_los_check (usually the robot's known_map).
    """
    if start_r == target_r and start_c == target_c:
        return True

    line_points = bresenham_line_algo(start_r, start_c, target_r, target_c)
    
    map_height, map_width = map_for_los_check.shape

    # Check intermediate points on the line (excluding start and end points themselves for obstacle check)
    # The line_points list includes both start and end.
    # We check obstacles on the path *between* start and target.
    for i in range(1, len(line_points) - 1): 
        pr, pc = line_points[i]
        
        # Bounds check for the point on the line (should be within map if target is)
        if not (0 <= pr < map_height and 0 <= pc < map_width):
            # This means part of the line goes out of bounds before reaching the target.
            # This can happen if the target is near the edge and the line "cuts a corner"
            # out of bounds according to Bresenham.
            # print(f"LoS: Point ({pr},{pc}) on line to ({target_r},{target_c}) is out of map bounds.")
            return False 
            
        if map_for_los_check[pr, pc] == OBSTACLE:
            return False # Line of sight is blocked by a known obstacle
    return True