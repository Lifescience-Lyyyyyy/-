import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def bresenham_line(start, end):
    x0, y0 = start
    x1, y1 = end
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points

def a_star_search(map_grid, start, goal, OBSTACLE_VAL=2):
    if start is None or goal is None or not (0 <= start[0] < map_grid.shape[0]) or not (0 <= goal[0] < map_grid.shape[0]):
        return None
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    close_set, came_from = set(), {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = [(fscore[start], start)]
    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]
        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if not (0 <= neighbor[0] < map_grid.shape[0] and 0 <= neighbor[1] < map_grid.shape[1]):
                continue
            if map_grid[neighbor[0], neighbor[1]] == OBSTACLE_VAL:
                continue
            tentative_g_score = gscore[current] + 1
            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None