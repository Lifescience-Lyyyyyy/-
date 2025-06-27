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

def check_line_of_sight(start_r, start_c, target_r, target_c, map_for_los_check, obstacle_value=2):
    if start_r == target_r and start_c == target_c:
        return True

    line_points = bresenham_line_algo(start_r, start_c, target_r, target_c)
    map_height, map_width = map_for_los_check.shape

    if not line_points: return False # Should not happen if start/target are valid

    # Check all points on the line *except the start point* for being an obstacle.
    # The target point itself being an obstacle doesn't block LoS TO it, 
    # but LoS THROUGH it would be blocked if it were an intermediate point.
    for i in range(1, len(line_points)): 
        pr, pc = line_points[i]
        
        if not (0 <= pr < map_height and 0 <= pc < map_width):
            return False 
            
        if map_for_los_check[pr, pc] == obstacle_value:
            return False 

        # --- Conservative check for "corner cutting" through obstacles ---
        # This is more relevant if your path planning allows diagonal moves but LoS should be stricter.
        # If robot only moves cardinally, Bresenham path itself is what matters.
        # This check is for when a Bresenham step implies a diagonal "shortcut" visually.
        if i > 0: # Check against the previous point on the line
            prev_r, prev_c = line_points[i-1]
            dr = pr - prev_r
            dc = pc - prev_c
            
            # If the step was diagonal (dr != 0 and dc != 0)
            if dr != 0 and dc != 0:
                # Check the two cells that form the "corners" of this diagonal step
                # If both of these "corner" cells are obstacles, the diagonal path is blocked
                # by a "wall" of two diagonally adjacent obstacles.
                corner_cell1_r, corner_cell1_c = prev_r + dr, prev_r # (pr, prev_c)
                corner_cell2_r, corner_cell2_c = prev_r, prev_c + dc # (prev_r, pc)

                is_corner1_obstacle = False
                if (0 <= corner_cell1_r < map_height and 0 <= corner_cell1_c < map_width):
                    if map_for_los_check[corner_cell1_r, corner_cell1_c] == obstacle_value:
                        is_corner1_obstacle = True
                # else: corner1 is out of bounds, effectively not blocking within map

                is_corner2_obstacle = False
                if (0 <= corner_cell2_r < map_height and 0 <= corner_cell2_c < map_width):
                    if map_for_los_check[corner_cell2_r, corner_cell2_c] == obstacle_value:
                        is_corner2_obstacle = True
                # else: corner2 is out of bounds

                if is_corner1_obstacle and is_corner2_obstacle:
                    # print(f"LoS conservative: Diagonal cut through obstacles at ({prev_r},{prev_c})->({pr},{pc}) blocked by ({corner_cell1_r},{corner_cell1_c}) and ({corner_cell2_r},{corner_cell2_c})")
                    return False
    return True