# planners/aco_known_map_planner.py
import numpy as np
import random

try:
    from environment import FREE, OBSTACLE # UNKNOWN not strictly needed for known map
except ImportError:
    FREE, OBSTACLE = 1, 2
    print("Warning (aco_known_map_planner.py): Could not import environment constants. Using default values.")

try:
    from .pathfinding import heuristic as manhattan_heuristic
except ImportError: 
    try:
        from pathfinding import heuristic as manhattan_heuristic
    except ImportError:
        def manhattan_heuristic(a,b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        print("Warning (aco_known_map_planner.py): Could not import manhattan_heuristic. Using basic fallback.")


class ACOKnownMapPlanner:
    def __init__(self, static_known_map, map_height, map_width,
                 start_pos, goal_pos,
                 n_ants=20, n_iterations=50,
                 alpha=1.0, beta=2.0,      
                 evaporation_rate=0.1,   
                 q_deposit=100.0,          
                 pheromone_min=0.01, pheromone_max=10.0,
                 visualize_ants_callback=None, # Callback signature: (start, map, ant_paths_THIS_iter, env, pheromone_map_CURRENT)
                 allow_diagonal_movement=False,
                 use_elitist_ant_system=True, 
                 elitist_weight_factor=1.0):  
        
        self.static_known_map = static_known_map 
        self.height = map_height
        self.width = map_width
        self.start_pos = tuple(start_pos) 
        self.goal_pos = tuple(goal_pos)   
        
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha  
        self.beta = beta    
        self.evaporation_rate = evaporation_rate 
        self.q_deposit = q_deposit 
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        
        self.visualize_ants_callback = visualize_ants_callback
        self.allow_diagonal_movement = allow_diagonal_movement
        self.use_elitist_ant_system = use_elitist_ant_system
        self.elitist_weight_factor = elitist_weight_factor 

        initial_pheromone = (pheromone_min + pheromone_max) / 2.0 
        self.pheromone_map = np.full((self.height, self.width), 
                                     initial_pheromone, dtype=float)
        
        self.heuristic_map = np.zeros((self.height, self.width), dtype=float)
        self._calculate_heuristic_map()
        
        self.heuristic_map[self.static_known_map == OBSTACLE] = 0.0
        self.pheromone_map[self.static_known_map == OBSTACLE] = self.pheromone_min 

        self.best_path_found_overall = [] 
        self.best_path_length_overall = float('inf') 
        
    def _calculate_heuristic_map(self):
        for r in range(self.height):
            for c in range(self.width):
                if self.static_known_map[r,c] != OBSTACLE:
                    dist = manhattan_heuristic((r, c), self.goal_pos)
                    self.heuristic_map[r, c] = 1.0 / (dist + 0.1) 

    def _get_allowed_moves(self):
        if self.allow_diagonal_movement:
            return [(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0)]
        else:
            return [(0,1), (0,-1), (1,0), (-1,0)] 

    def _select_next_step_for_ant(self, current_r, current_c, ant_visited_in_path):
        possible_next_steps = []
        allowed_moves = self._get_allowed_moves()
        for dr, dc in allowed_moves:
            next_r, next_c = current_r + dr, current_c + dc
            if not (0 <= next_r < self.height and 0 <= next_c < self.width): continue
            if self.static_known_map[next_r, next_c] == OBSTACLE: continue
            if (next_r, next_c) in ant_visited_in_path: continue 
            tau_val = self.pheromone_map[next_r, next_c] ** self.alpha
            eta_val = self.heuristic_map[next_r, next_c] ** self.beta
            if tau_val > 0 and eta_val > 0: 
                 prob_numerator = tau_val * eta_val
                 if prob_numerator > 1e-9:
                    possible_next_steps.append({'pos': (next_r, next_c), 'score': prob_numerator})
        if not possible_next_steps: return None
        total_score_sum = sum(s['score'] for s in possible_next_steps)
        if total_score_sum <= 1e-9: return random.choice(possible_next_steps)['pos']
        rand_val = random.random() * total_score_sum
        current_sum = 0.0
        for step_info in possible_next_steps:
            current_sum += step_info['score']
            if current_sum >= rand_val: return step_info['pos']
        return possible_next_steps[-1]['pos']

    def find_path(self):
        # This list is NO LONGER for collecting all iterations' paths for a single callback at the end.
        # Instead, callback is called per iteration.
        # We might still want to collect all paths if some other part of the code needs it.
        # For now, let's assume the callback handles immediate visualization.

        for iteration in range(self.n_iterations):
            current_iteration_all_ant_paths = [] 
            
            for _ in range(self.n_ants):
                ant_r, ant_c = self.start_pos
                current_ant_path_nodes = [(ant_r, ant_c)] 
                current_ant_path_visited_set = {self.start_pos} 
                max_steps_for_ant = (self.height + self.width) * 2 
                for _ in range(max_steps_for_ant):
                    if (ant_r, ant_c) == self.goal_pos: break 
                    next_pos = self._select_next_step_for_ant(ant_r, ant_c, current_ant_path_visited_set)
                    if next_pos is None: break 
                    ant_r, ant_c = next_pos
                    current_ant_path_nodes.append((ant_r, ant_c))
                    current_ant_path_visited_set.add((ant_r, ant_c))
                current_iteration_all_ant_paths.append(current_ant_path_nodes)

                if current_ant_path_nodes and current_ant_path_nodes[-1] == self.goal_pos:
                    path_len = len(current_ant_path_nodes) - 1
                    if path_len < self.best_path_length_overall:
                        self.best_path_length_overall = path_len
                        self.best_path_found_overall = list(current_ant_path_nodes)
            
            # --- Pheromone Evaporation ---
            self.pheromone_map *= (1.0 - self.evaporation_rate)
            non_obstacle_mask = (self.static_known_map != OBSTACLE)
            self.pheromone_map[non_obstacle_mask] = np.maximum(
                self.pheromone_map[non_obstacle_mask], self.pheromone_min)

            # --- Pheromone Deposition (by all ants that found the goal in this iteration) ---
            for ant_path in current_iteration_all_ant_paths:
                if not ant_path or ant_path[-1] != self.goal_pos: continue 
                path_length = len(ant_path) -1 
                if path_length == 0: continue
                delta_pheromone_per_cell = self.q_deposit / path_length 
                for r_cell, c_cell in ant_path:
                    if self.static_known_map[r_cell,c_cell] != OBSTACLE:
                        self.pheromone_map[r_cell, c_cell] += delta_pheromone_per_cell
            
            # --- Elitist Ant System: Extra Pheromone for the Best Path Found So Far ---
            if self.use_elitist_ant_system and self.best_path_found_overall:
                if self.best_path_length_overall > 0 : 
                    elite_deposit_amount = self.elitist_weight_factor * (self.q_deposit / self.best_path_length_overall)
                    for r_cell, c_cell in self.best_path_found_overall:
                        if self.static_known_map[r_cell, c_cell] != OBSTACLE:
                            self.pheromone_map[r_cell, c_cell] += elite_deposit_amount
            
            # --- Clip Pheromones to Max Bound ---
            self.pheromone_map[non_obstacle_mask] = np.minimum(
                self.pheromone_map[non_obstacle_mask], self.pheromone_max)
            
            # --- Call visualization callback AFTER pheromones for THIS iteration are updated ---
            if self.visualize_ants_callback:
                 self.visualize_ants_callback(
                     start_node_arg=self.start_pos, 
                     map_grid_ref_arg=self.static_known_map, 
                     # Pass only the paths from the CURRENT iteration
                     ant_paths_this_iteration_arg=current_iteration_all_ant_paths, 
                     environment_ref_arg=None, # Or self.environment if callback needs it
                     # Pass the CURRENT state of the pheromone map
                     pheromone_map_to_display_arg=np.copy(self.pheromone_map) 
                 )
            # print(f"KnownMapACO Iter {iteration+1} done. Best path: {self.best_path_length_overall}")


        final_path = self._construct_path_from_pheromones_greedy()
        if final_path:
            return self.goal_pos, final_path
        elif self.best_path_found_overall:
            return self.goal_pos, self.best_path_found_overall
        else:
            return None, None

    def _construct_path_from_pheromones_greedy(self):
        # ... (this method remains the same as in the previous complete version) ...
        path = [self.start_pos]
        current_r, current_c = self.start_pos
        max_greedy_steps = (self.height + self.width) * 10 
        allowed_moves = self._get_allowed_moves()

        for _ in range(max_greedy_steps):
            if (current_r, current_c) == self.goal_pos:
                return path

            best_next_greedy_step = None
            max_score_greedy = -1.0
            
            for dr, dc in allowed_moves:
                next_r, next_c = current_r + dr, current_c + dc
                if not (0 <= next_r < self.height and 0 <= next_c < self.width): continue
                if self.static_known_map[next_r, next_c] == OBSTACLE: continue
                if (next_r, next_c) in path: continue

                score = (self.pheromone_map[next_r, next_c] ** self.alpha) * \
                        (self.heuristic_map[next_r, next_c] ** self.beta)

                if score > max_score_greedy:
                    max_score_greedy = score
                    best_next_greedy_step = (next_r, next_c)
            
            if best_next_greedy_step is None:
                return None 
            
            current_r, current_c = best_next_greedy_step
            path.append((current_r, current_c))
        
        return None