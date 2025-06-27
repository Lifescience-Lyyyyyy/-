import numpy as np
import random
import asyncio
from .a_star import a_star_search, heuristic

FREE, OBSTACLE, UNKNOWN = 1, 2, 0

def bresenham_line(start, end):
    x0, y0 = start; x1, y1 = end; points = []
    dx = abs(x1 - x0); dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1; sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0));
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 >= dy: err += dy; x0 += sx
        if e2 <= dx: err += dx; y0 += sy
    return points

class PathPlanningACO:
    def __init__(self, map_grid, start_pos, goal_pos, params, websocket):
        self.map_grid = map_grid; self.height, self.width = map_grid.shape
        self.start_pos, self.goal_pos = tuple(start_pos), tuple(goal_pos)
        self.params = params; self.websocket = websocket
        self.n_ants = int(params.get("n_ants_pp", 20)); self.n_iterations = int(params.get("n_iterations_pp", 50))
        self.alpha = float(params.get("alpha", 1.0)); self.beta = float(params.get("beta", 2.5))
        self.evaporation_rate = float(params.get("evaporation", 0.3)); self.use_elitist = params.get("use_elitist", True)
        self.pheromone_max = 20.0; self.pheromone_min = 0.01; self.q_deposit = 100.0; self.elitist_weight = 2.0
        self.pheromone_map = np.full((self.height, self.width), self.pheromone_min, dtype=float)
        self.heuristic_map = np.zeros((self.height, self.width), dtype=float)
        self._calculate_heuristic_map()
        self.heuristic_map[self.map_grid == OBSTACLE] = 0.0
        self.pheromone_map[self.map_grid == OBSTACLE] = 0.0
        self.best_path_overall, self.best_len_overall = [], float('inf')
    def _calculate_heuristic_map(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.map_grid[r,c] != OBSTACLE:
                    dist = heuristic((r, c), self.goal_pos)
                    self.heuristic_map[r, c] = 1.0 / (dist + 0.1)
    def _select_next_step(self, r, c, visited):
        moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]; possible_steps = []
        for dr, dc in moves:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width and self.map_grid[nr, nc] != OBSTACLE and (nr, nc) not in visited:
                tau = self.pheromone_map[nr, nc] ** self.alpha; eta = self.heuristic_map[nr, nc] ** self.beta
                possible_steps.append({'pos': (nr, nc), 'score': tau * eta})
        if not possible_steps: return None
        total_score = sum(s['score'] for s in possible_steps)
        if total_score <= 1e-9: return random.choice(possible_steps)['pos']
        rand_val = random.random() * total_score; current_sum = 0.0
        for step in possible_steps:
            current_sum += step['score']
            if current_sum >= rand_val: return step['pos']
        return possible_steps[-1]['pos']
    async def find_path(self):
        for iteration in range(self.n_iterations):
            iter_paths = []
            for _ in range(self.n_ants):
                r, c = self.start_pos; path = [(r, c)]; visited = {(r, c)}
                for _ in range(self.height * self.width):
                    if (r, c) == self.goal_pos: break
                    next_pos = self._select_next_step(r, c, visited)
                    if next_pos is None: break
                    r, c = next_pos; path.append((r, c)); visited.add((r, c))
                if path and path[-1] == self.goal_pos:
                    iter_paths.append(path)
                    if len(path) < self.best_len_overall: self.best_len_overall, self.best_path_overall = len(path), list(path)
            self.pheromone_map *= (1.0 - self.evaporation_rate)
            for path in iter_paths:
                delta = self.q_deposit / len(path)
                for r_cell, c_cell in path: self.pheromone_map[r_cell, c_cell] += delta
            if self.use_elitist and self.best_path_overall:
                delta_elite = self.elitist_weight * (self.q_deposit / self.best_len_overall)
                for r_cell, c_cell in self.best_path_overall: self.pheromone_map[r_cell, c_cell] += delta_elite
            self.pheromone_map = np.clip(self.pheromone_map, self.pheromone_min, self.pheromone_max)
            await self.websocket.send_json({ "type": "planning_update", "iteration": iteration + 1, "total_iterations": self.n_iterations, "ant_paths": iter_paths, "pheromone_map": self.pheromone_map.tolist() })
            await asyncio.sleep(0.05)
        return self.best_path_overall if self.best_path_overall else None

class ExplorationEngine:
    def __init__(self, true_grid, start_pos, params, websocket):
        self.true_grid = true_grid; self.height, self.width = true_grid.shape
        self.robot_pos = start_pos; self.params = params; self.websocket = websocket
        self.known_grid = np.full_like(true_grid, UNKNOWN); self.steps = 0
        self.sensor_range = int(params.get("sensor_range", 5))
        self.n_ants = int(params.get("n_ants_exp", 15)); self.n_iterations = int(params.get("n_iterations_exp", 10))
        self.alpha = float(params.get("alpha", 1.0)); self.beta = float(params.get("beta", 5.0))
        self.evaporation_rate = float(params.get("evaporation", 0.5)); self.pheromone_map = np.full_like(true_grid, 0.5, dtype=float)

    def _sense(self):
        """Simulates the robot's sensor. Correctly handles occlusion."""
        r_rob, c_rob = self.robot_pos
        r_min, r_max = max(0, r_rob - self.sensor_range), min(self.height, r_rob + self.sensor_range + 1)
        c_min, c_max = max(0, c_rob - self.sensor_range), min(self.width, c_rob + self.sensor_range + 1)

        for r_target in range(r_min, r_max):
            for c_target in range(c_min, c_max):
                if (r_rob - r_target)**2 + (c_rob - c_target)**2 <= self.sensor_range**2:
                    line = bresenham_line(self.robot_pos, (r_target, c_target))
                    is_occluded = any(self.true_grid[p] == OBSTACLE for p in line[1:-1])
                    
                    if not is_occluded:
                        for p in line:
                            self.known_grid[p] = self.true_grid[p]

    def _find_frontiers(self):
        frontiers = [];
        for r in range(self.height):
            for c in range(self.width):
                if self.known_grid[r,c] == FREE:
                    for dr, dc in [(0,1),(0,-1),(1,0),(-1,0)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < self.height and 0 <= nc < self.width and self.known_grid[nr,nc] == UNKNOWN:
                            frontiers.append((r,c)); break
        return frontiers
    def _update_pheromones(self, frontiers):
        if not frontiers: return
        self.pheromone_map *= (1.0 - self.evaporation_rate)
        for r,c in frontiers: self.pheromone_map[r, c] += 1.0 / (heuristic((r,c), self.robot_pos) + 1.0)
        self.pheromone_map = np.clip(self.pheromone_map, 0.01, 10.0)
    def _choose_next_frontier(self, frontiers):
        if not frontiers: return None; best_target = None; max_score = -1
        sampled_frontiers = random.sample(frontiers, min(len(frontiers), 20))
        for f in sampled_frontiers:
            path = a_star_search(self.known_grid, self.robot_pos, f, OBSTACLE_VAL=OBSTACLE)
            if path:
                score = self.pheromone_map[f] / len(path)
                if score > max_score: max_score = score; best_target = f
        return best_target if best_target else random.choice(frontiers)
    async def explore(self):
        total_area = self.height * self.width; max_steps = total_area * 2
        while self.steps < max_steps:
            self._sense(); frontiers = self._find_frontiers()
            progress = np.sum(self.known_grid != UNKNOWN) / total_area * 100
            await self.websocket.send_json({ "type": "exploration_update", "known_grid": self.known_grid.tolist(), "robot_pos": self.robot_pos, "frontiers": frontiers, "steps": self.steps, "progress": progress })
            if not frontiers: await self.websocket.send_json({"type": "exploration_end", "reason": "探索完成！", "known_grid": self.known_grid.tolist(), "robot_pos": self.robot_pos, "steps": self.steps, "progress": 100.0}); return
            self._update_pheromones(frontiers); target = self._choose_next_frontier(frontiers)
            if target is None: await self.websocket.send_json({"type": "exploration_end", "reason": "找不到可达的探索边界", "known_grid": self.known_grid.tolist(), "robot_pos": self.robot_pos, "steps": self.steps, "progress": progress}); return
            path = a_star_search(self.known_grid, self.robot_pos, target, OBSTACLE_VAL=OBSTACLE)
            if not path or len(path) < 2:
                paths = [p for p in [a_star_search(self.known_grid, self.robot_pos, f, OBSTACLE) for f in random.sample(frontiers, min(len(frontiers), 10))] if p]
                if not paths: await self.websocket.send_json({"type": "exploration_end", "reason": "机器人被困住了", "known_grid": self.known_grid.tolist(), "robot_pos": self.robot_pos, "steps": self.steps, "progress": progress}); return
                path = min(paths, key=len)
            for pos in path[1:]:
                self.robot_pos = pos; self.steps += 1; self._sense()
                await self.websocket.send_json({ "type": "robot_move", "pos": self.robot_pos, "known_grid": self.known_grid.tolist(), "path_to_target": path, "steps": self.steps })
                await asyncio.sleep(0.03)
        final_progress = np.sum(self.known_grid != UNKNOWN) / total_area * 100
        await self.websocket.send_json({"type": "exploration_end", "reason": "达到最大步数", "known_grid": self.known_grid.tolist(), "robot_pos": self.robot_pos, "steps": self.steps, "progress": final_progress})