import heapq
import numpy as np
try:
    from environment import FREE, OBSTACLE
except ImportError:
    FREE, OBSTACLE = 1, 2


def heuristic(a, b):
    """曼哈顿距离启发式函数 for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(known_map, start, goal):
    """
    A* 寻路算法。
    known_map: 机器人当前已知的地图。
    start: (r, c) 起始点。
    goal: (r, c) 目标点。
    返回一个 (r, c) 元组列表构成的路径，如果无路径则返回 None。
    """
    rows, cols = known_map.shape
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4方向连接

    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    oheap = []

    heapq.heappush(oheap, (fscore[start], start))

    while oheap:
        current = heapq.heappop(oheap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] # 返回反转后的路径

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue # 超出边界

            # 严格禁止穿过已知的障碍物
            if known_map[neighbor[0], neighbor[1]] == OBSTACLE:
                continue
            
            # 允许路径穿过 FREE 空间。对于探索任务，目标通常是 FREE 空间，
            # A* 会自动处理路径上的 UNKNOWN 空间（将其视为可通行）。
            
            tentative_g_score = gscore[current] + 1 # 移动到邻居的成本为1

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None # 未找到路径