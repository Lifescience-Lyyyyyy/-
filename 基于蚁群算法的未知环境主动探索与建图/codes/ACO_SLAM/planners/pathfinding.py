import heapq
import numpy as np
from environment import FREE, OBSTACLE 

def heuristic(a, b):
    """Manhattan distance heuristic for A*."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(known_map, start, goal):
    """
    A* pathfinding algorithm.
    known_map: The robot's current understanding of the map.
    start: (r, c) tuple for starting position.
    goal: (r, c) tuple for goal position.
    Returns a list of (r, c) tuples representing the path, or None if no path.
    """
    rows, cols = known_map.shape
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] 

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
            return path[::-1] 

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue 

            if known_map[neighbor[0], neighbor[1]] == OBSTACLE:
                continue


            tentative_g_score = gscore[current] + 1 

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None 