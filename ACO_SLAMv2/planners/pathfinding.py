# planners/pathfinding.py
import heapq
import numpy as np
from environment import FREE, OBSTACLE # Assuming UNKNOWN can be traversed initially for planning frontiers

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
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4-connectivity
                # Add [(1,1), (1,-1), (-1,1), (-1,-1)] for 8-connectivity

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
            path.append(start) # Add start to the path
            return path[::-1] # Return reversed path

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue # Out of bounds

            # Allow path through FREE or UNKNOWN (if goal is a frontier in UNKNOWN)
            # But strictly not through OBSTACLE
            if known_map[neighbor[0], neighbor[1]] == OBSTACLE:
                continue
            
            # If goal is UNKNOWN (frontier), allow planning through UNKNOWN
            # otherwise, only through FREE for general movement
            # For frontier planning, goal is usually FREE but adjacent to UNKNOWN
            # The check for OBSTACLE above is the main one.

            tentative_g_score = gscore[current] + 1 # Cost of moving to neighbor is 1

            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                continue

            if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))

    return None # No path found