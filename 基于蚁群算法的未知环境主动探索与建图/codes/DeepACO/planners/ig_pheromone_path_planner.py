import numpy as np
import random
from .base_planner import BasePlanner 
from environment import UNKNOWN, FREE, OBSTACLE
from .geometry_utils import check_line_of_sight 
from .pathfinding import heuristic as manhattan_heuristic
from collections import deque # For BFS in clustering

class IGPheromonePathPlanner(BasePlanner): # Renaming to reflect clustering might be good later
    def __init__(self, environment,
                 n_ants=15, n_iterations=10,
                 alpha=1.0, beta_ant_to_target_frontier=2.0, 
                 
                 evaporation_rate=0.05,
                 pheromone_initial=0.1, pheromone_min=0.01, pheromone_max=10.0,
                 
                 q_deposit_factor=1.0,    
                 ig_calculation_sensor_range=5, 
                 
                 ant_max_steps_to_frontier=75, 

                 max_pheromone_nav_steps=50,
                 
                 visualize_ants_callback=None,
                 allow_diagonal_movement=False,
                 # --- Clustering Parameters ---
                 enable_frontier_clustering=True, # Toggle for clustering
                 frontier_clustering_distance=3,  # Max Manhattan distance for points in same cluster
                 cluster_ig_scaling_factor=1.0,   # Multiplier for cluster size effect on IG (e.g., 1.0 for linear, 0.5 for sqrt like)
                 select_core_ig_point_in_cluster=True # Whether to select a core point in the cluster based on IG
                 ):
        
        super().__init__(environment)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha 
        self.beta_ant_to_target_frontier = beta_ant_to_target_frontier
        
        self.evaporation_rate = evaporation_rate
        self.pheromone_map = np.full((self.environment.height, self.environment.width), 
                                     pheromone_initial, dtype=float)
        self.pheromone_min = pheromone_min
        self.pheromone_max = pheromone_max
        
        self.q_deposit_factor = q_deposit_factor
        self.ig_calculation_sensor_range = ig_calculation_sensor_range
        self.ant_max_path_steps = ant_max_steps_to_frontier # Name was ant_max_steps_to_frontier

        self.max_pheromone_nav_steps = max_pheromone_nav_steps
        
        self.visualize_ants_callback = visualize_ants_callback
        self.allow_diagonal_movement = allow_diagonal_movement

        # Clustering parameters
        self.enable_frontier_clustering = enable_frontier_clustering
        self.frontier_clustering_distance = frontier_clustering_distance
        self.cluster_ig_scaling_factor = cluster_ig_scaling_factor # 1.0 for linear, 0.5 for sqrt
        self.select_core_ig_point_in_cluster = select_core_ig_point_in_cluster
        self._update_pheromones_for_obstacles(None)

    # --- Helper methods _update_pheromones_for_obstacles, _get_allowed_moves, 
    # --- _calculate_ig_at_frontier (renamed from _calculate_ig_with_los),
    # --- _ant_select_next_step_towards_target (all same as previous complete version) ---
    def _update_pheromones_for_obstacles(self, known_map=None):
        map_to_check = known_map
        if map_to_check is None: 
            if hasattr(self.environment, 'grid_true') and self.environment.grid_true is not None:
                map_to_check = self.environment.grid_true
            else: return 
        obstacle_mask = (map_to_check == OBSTACLE)
        self.pheromone_map[obstacle_mask] = 0.0 

    def _get_allowed_moves(self):
        if self.allow_diagonal_movement:
            return [(dr,dc) for dr in [-1,0,1] for dc in [-1,0,1] if not (dr==0 and dc==0)]
        else:
            return [(0,1), (0,-1), (1,0), (-1,0)]

    def _calculate_ig_at_point(self, r_prospective, c_prospective, known_map_snapshot, sensor_range): # Renamed
        ig = 0
        for dr_ig in range(-sensor_range, sensor_range + 1):
            for dc_ig in range(-sensor_range, sensor_range + 1):
                r_target, c_target = r_prospective + dr_ig, c_prospective + dc_ig
                if self.environment.is_within_bounds(r_target, c_target):
                    if known_map_snapshot[r_target, c_target] == UNKNOWN:
                        if check_line_of_sight(r_prospective, c_prospective, 
                                               r_target, c_target, 
                                               known_map_snapshot):
                            ig += 1
        return ig

    def _ant_select_next_step_towards_target(self, current_r, current_c, 
                                             ant_target_pos, # Can be a frontier or cluster representative
                                             ant_path_visited_set, known_map_snapshot):
        possible_next_steps_info = []; allowed_moves = self._get_allowed_moves()
        for dr, dc in allowed_moves:
            next_r, next_c = current_r + dr, current_c + dc
            if not self.environment.is_within_bounds(next_r, next_c): continue
            if known_map_snapshot[next_r, next_c] == OBSTACLE: continue
            if (next_r, next_c) in ant_path_visited_set: continue 
            tau_val = self.pheromone_map[next_r, next_c] ** self.alpha
            dist_to_ant_target = manhattan_heuristic((next_r, next_c), ant_target_pos)
            eta_val = (1.0 / (dist_to_ant_target + 0.1)) ** self.beta_ant_to_target_frontier
            prob_score = tau_val * eta_val
            if prob_score > 1e-9: possible_next_steps_info.append({'pos': (next_r, next_c), 'score': prob_score})
        if not possible_next_steps_info: return None
        total_score_sum = sum(info['score'] for info in possible_next_steps_info)
        if total_score_sum <= 1e-9: return random.choice(possible_next_steps_info)['pos']
        selection_rand_val = random.random() * total_score_sum; cumulative_sum = 0.0
        for step_info in possible_next_steps_info:
            cumulative_sum += step_info['score']
            if cumulative_sum >= selection_rand_val: return step_info['pos']
        return possible_next_steps_info[-1]['pos']

    # --- New Clustering and Cluster Evaluation Methods ---
    def _cluster_frontiers(self, frontiers):
        """Clusters frontiers based on Manhattan distance."""
        if not frontiers: return []
        clusters = []
        visited = [False] * len(frontiers)
        for i, fr_i in enumerate(frontiers):
            if visited[i]: continue
            current_cluster = [fr_i]
            visited[i] = True
            queue = deque([fr_i])
            head = 0
            while head < len(queue): # Manual queue for index access if needed, or use deque
                curr_fr = queue[head]; head+=1
                for j, fr_j in enumerate(frontiers):
                    if not visited[j]:
                        dist = manhattan_heuristic(curr_fr, fr_j)
                        if dist <= self.frontier_clustering_distance:
                            visited[j] = True
                            queue.append(fr_j) # Add to deque
                            current_cluster.append(fr_j) # Add to current_cluster list
            clusters.append(current_cluster)
        return clusters

    def _get_cluster_representative_and_ig(self, cluster, robot_pos, known_map_snapshot):
        """
        Selects a representative point for a cluster and calculates its aggregated IG.
        Representative: Closest reachable frontier in the cluster to the robot.
        Aggregated IG: IG from representative * (num_points_in_cluster ^ scaling_factor).
        Returns: (representative_pos, path_to_representative, aggregated_ig, cost_to_representative) or (None,None,0,inf)
        """
        if not cluster: return None, None, 0.0, float('inf')

        best_rep_fr = None
        path_to_best_rep = None
        min_cost_to_rep = float('inf')

        # Find the closest reachable frontier in the cluster to be the representative
        for fr_in_cluster in cluster:
            path = self._is_reachable_and_get_path(robot_pos, fr_in_cluster, known_map_snapshot)
            if path:
                cost = len(path) -1
                if cost < min_cost_to_rep:
                    min_cost_to_rep = cost
                    best_rep_fr = fr_in_cluster
                    path_to_best_rep = path
        
        if best_rep_fr is None: # No frontier in cluster is reachable from robot_pos
            return None, None, 0.0, float('inf')

        ig_from_representative = self._calculate_ig_at_point(
            best_rep_fr[0], best_rep_fr[1], known_map_snapshot, self.ig_calculation_sensor_range
        )
        
        # Scale IG by cluster size
        # cluster_ig_scaling_factor = 1.0 for linear, 0.5 for sqrt, etc.
        size_factor = len(cluster) ** self.cluster_ig_scaling_factor
        aggregated_ig = ig_from_representative * size_factor
        
        actual_path_cost = min_cost_to_rep if min_cost_to_rep > 0 else 1e-5

        return best_rep_fr, path_to_best_rep, aggregated_ig, actual_path_cost


    def _run_ants_for_clustered_pheromones(self, robot_start_pos, known_map_snapshot, frontier_clusters, outer_planning_cycle_num):
        """
        Ants select a TARGET CLUSTER, then path towards its REPRESENTATIVE POINT.
        Pheromone deposit is based on the CLUSTER'S aggregated IG.
        """
        all_iterations_ant_paths_for_viz = []

        if not frontier_clusters: # No clusters for ants to target
            if self.visualize_ants_callback: # Send empty paths if viz is on
                 self.visualize_ants_callback(
                    start_node_arg=robot_start_pos, map_grid_ref_arg=known_map_snapshot,
                    ant_paths_all_iterations_data_arg=[[] for _ in range(self.n_iterations)], 
                    environment_ref_arg=self.environment, 
                    pheromone_map_to_display_arg=np.copy(self.pheromone_map),
                    current_iteration_num_arg=outer_planning_cycle_num, 
                    total_iterations_arg=self.n_iterations)
            return

        # --- Pre-calculate utility for each cluster for ants to pick a target cluster ---
        # This utility can be simpler than robot's final choice, e.g., just AggregatedIG / CostToClusterRep
        cluster_utilities_for_ants = []
        for i, cluster in enumerate(frontier_clusters):
            rep_pos, path_to_rep, agg_ig, cost_to_rep = self._get_cluster_representative_and_ig(
                cluster, robot_start_pos, known_map_snapshot
            )
            if rep_pos and cost_to_rep != float('inf'):
                utility = agg_ig / cost_to_rep if cost_to_rep > 0 else agg_ig / 1e-5
                cluster_utilities_for_ants.append({'cluster_idx': i, 'representative': rep_pos, 'utility': utility, 'agg_ig': agg_ig})
        
        if not cluster_utilities_for_ants: # No valid clusters for ants
             if self.visualize_ants_callback: self.visualize_ants_callback(robot_start_pos, known_map_snapshot,[[] for _ in range(self.n_iterations)],self.environment,np.copy(self.pheromone_map),outer_planning_cycle_num,self.n_iterations)
             return


        for iteration_idx in range(self.n_iterations): 
            ant_paths_this_inner_iter_for_viz = []
            paths_data_for_deposit = [] 

            for _ in range(self.n_ants):
                ant_r, ant_c = robot_start_pos; current_ant_path = [(ant_r, ant_c)]; ant_path_visited_set = {robot_start_pos}
                
                # Ant selects a TARGET CLUSTER based on pre-calculated utilities (e.g., roulette wheel)
                if not cluster_utilities_for_ants: break
                
                # Roulette wheel selection for target cluster
                total_cluster_utility = sum(cu['utility'] for cu in cluster_utilities_for_ants if cu['utility'] > 0)
                chosen_cluster_info = None
                if total_cluster_utility <= 1e-9:
                    chosen_cluster_info = random.choice(cluster_utilities_for_ants)
                else:
                    rand_val = random.random() * total_cluster_utility
                    current_sum = 0
                    for cu_info in cluster_utilities_for_ants:
                        if cu_info['utility'] > 0:
                            current_sum += cu_info['utility']
                            if current_sum >= rand_val:
                                chosen_cluster_info = cu_info
                                break
                    if chosen_cluster_info is None: chosen_cluster_info = random.choice(cluster_utilities_for_ants) # Fallback
                
                target_representative_for_ant = chosen_cluster_info['representative']
                cluster_agg_ig_for_deposit = chosen_cluster_info['agg_ig']


                for _ in range(self.ant_max_path_steps): # Ant tries to reach the representative
                    if (ant_r, ant_c) == target_representative_for_ant: 
                        paths_data_for_deposit.append({'path': list(current_ant_path), 'ig_metric': cluster_agg_ig_for_deposit})
                        break 
                    next_pos = self._ant_select_next_step_towards_target(
                        ant_r, ant_c, target_representative_for_ant, 
                        ant_path_visited_set, known_map_snapshot
                    )
                    if next_pos is None: break 
                    ant_r, ant_c = next_pos; current_ant_path.append((ant_r, ant_c)); ant_path_visited_set.add((ant_r, ant_c))
                
                if self.visualize_ants_callback: ant_paths_this_inner_iter_for_viz.append(current_ant_path) 
            
            if self.visualize_ants_callback and ant_paths_this_inner_iter_for_viz:
                all_iterations_ant_paths_for_viz.append(ant_paths_this_inner_iter_for_viz)

            self.pheromone_map *= (1.0 - self.evaporation_rate)
            self.pheromone_map = np.maximum(self.pheromone_map, self.pheromone_min) 
            self._update_pheromones_for_obstacles(known_map_snapshot) 

            for data in paths_data_for_deposit:
                path, ig_metric = data['path'], data['ig_metric']
                if not path or len(path) <=1 or ig_metric <= 0: continue
                path_len = len(path) -1; 
                if path_len == 0 : path_len = 1 
                pheromone_to_deposit_on_path = (self.q_deposit_factor * ig_metric) / path_len
                for r_cell, c_cell in path:
                    if known_map_snapshot[r_cell, c_cell] != OBSTACLE:
                        self.pheromone_map[r_cell, c_cell] += pheromone_to_deposit_on_path
            
            self.pheromone_map = np.clip(self.pheromone_map, self.pheromone_min, self.pheromone_max) 
            self._update_pheromones_for_obstacles(known_map_snapshot)
        
        if self.visualize_ants_callback and all_iterations_ant_paths_for_viz:
             self.visualize_ants_callback(
                 start_node_arg=robot_start_pos, map_grid_ref_arg=known_map_snapshot,
                 ant_paths_all_iterations_data_arg=all_iterations_ant_paths_for_viz, 
                 environment_ref_arg=self.environment, 
                 pheromone_map_to_display_arg=np.copy(self.pheromone_map),
                 current_iteration_num_arg=outer_planning_cycle_num, 
                 total_iterations_arg=self.n_iterations )

    def _select_core_ig_point_from_cluster(self, cluster_points, known_map_snapshot):
        """
        From a cluster of frontier points, select the "best" one based on:
        1. Individual IG of points in the cluster.
        2. For points with max IG, how many of their neighbors are also high-IG/frontier points.
        Returns: The selected core (r,c) point, or None if cluster is empty/invalid.
        """
        if not cluster_points:
            return None

        # 1. Calculate IG for all points in the cluster
        igs_in_cluster = {}
        for fr_p in cluster_points:
            igs_in_cluster[fr_p] = self._calculate_ig_at_point(
                fr_p[0], fr_p[1], known_map_snapshot, self.ig_calculation_sensor_range
            )
        
        if not igs_in_cluster: return None # Should not happen if cluster_points is not empty

        # 2. Find the maximum IG in this cluster
        max_ig_in_cluster = 0
        if igs_in_cluster.values(): # Check if there are any values
             max_ig_in_cluster = max(igs_in_cluster.values())
        
        if max_ig_in_cluster <= 0: # No point in this cluster offers positive IG
            # Fallback: return a point from the cluster, e.g., the first one,
            # or let the calling function handle this (e.g. by _get_cluster_representative_and_ig)
            # For now, if no positive IG, this cluster is not very attractive for this refined selection.
            # We can return a random point from the cluster, or the one used for agg_ig calculation.
            # To keep it simple, if there's no strong IG signal, picking a "central" or "easy to reach"
            # point from the cluster (which _get_cluster_representative_and_ig does) is reasonable.
            # This function focuses on finding a "hotspot" within a cluster.
            # If no hotspot, then the representative point is as good as any.
             return random.choice(cluster_points) if cluster_points else None


        # 3. Identify all points in the cluster that have this maximum IG
        high_ig_points_in_cluster = {pt for pt, ig_val in igs_in_cluster.items() if ig_val >= max_ig_in_cluster * 0.9} # Allow slight tolerance
        if not high_ig_points_in_cluster: # Should have at least one if max_ig_in_cluster > 0
            return random.choice(cluster_points) if cluster_points else None


        # 4. For each high-IG point, count how many of its neighbors are also high-IG points (or general frontiers in cluster)
        best_core_point = None
        max_high_ig_neighbors = -1

        allowed_moves_local = self._get_allowed_moves() # 4 or 8 directions

        for core_candidate_r, core_candidate_c in high_ig_points_in_cluster:
            current_high_ig_neighbor_count = 0
            for dr, dc in allowed_moves_local:
                nr, nc = core_candidate_r + dr, core_candidate_c + dc
                if (nr, nc) in high_ig_points_in_cluster: # Check if neighbor is also a high-IG point
                    current_high_ig_neighbor_count += 1
                # Optionally, also count if neighbor is just *any* frontier in the original cluster
                # elif (nr, nc) in cluster_points:
                #    current_high_ig_neighbor_count += 0.5 # Lesser weight for general frontier neighbor
            
            if current_high_ig_neighbor_count > max_high_ig_neighbors:
                max_high_ig_neighbors = current_high_ig_neighbor_count
                best_core_point = (core_candidate_r, core_candidate_c)
            elif current_high_ig_neighbor_count == max_high_ig_neighbors and best_core_point is not None:
                # Tie-breaking: choose the one closer to the geometric center of high_ig_points_in_cluster
                # Or simpler: just keep the first one found.
                # For now, keep the first one with max neighbors.
                pass 
        
        if best_core_point:
            return best_core_point
        elif high_ig_points_in_cluster: # If no point has high-IG neighbors, but high-IG points exist
            return random.choice(list(high_ig_points_in_cluster)) # Pick one of them
        elif cluster_points : # Fallback to random point in cluster if all else fails
            return random.choice(cluster_points)
        return None

    def plan_next_action(self, robot_pos, known_map, **kwargs):
        self._update_pheromones_for_obstacles(known_map)
        
        all_frontiers = self.find_frontiers(known_map)
        if not all_frontiers: 
            return self._final_fallback_plan(robot_pos, known_map), {} # Return empty info dict

        frontier_clusters = []
        if self.enable_frontier_clustering:
            frontier_clusters = self._cluster_frontiers(all_frontiers)
            if not frontier_clusters:
                frontier_clusters = [[fr] for fr in all_frontiers] 
        else: 
            frontier_clusters = [[fr] for fr in all_frontiers]

        if not frontier_clusters:
             return self._final_fallback_plan(robot_pos, known_map), {}

        known_map_snapshot = np.copy(known_map)
        outer_planning_cycle = kwargs.get('total_planning_cycles', 0)
        
        self._run_ants_for_clustered_pheromones(
            robot_pos, known_map_snapshot, 
            frontier_clusters, 
            outer_planning_cycle
        )
        
        best_utility = -float('inf')
        best_target_for_robot = None 
        best_path_for_robot = None
        
        # For visualization of chosen representatives or core points for each cluster
        chosen_points_per_cluster_for_viz = [None] * len(frontier_clusters)

        for idx, cluster in enumerate(frontier_clusters):
            if not cluster: continue
            
            # Step 1: Get a representative point for pathing and the cluster's aggregated IG
            representative_pos_for_cost, path_to_rep, cluster_agg_ig, cost_to_rep = \
                self._get_cluster_representative_and_ig(cluster, robot_pos, known_map_snapshot)

            if representative_pos_for_cost is None or cluster_agg_ig <= 0 or cost_to_rep == float('inf'):
                continue

            # Step 2: If enabled, refine target within this cluster to a "core IG point"
            actual_target_in_cluster = representative_pos_for_cost # Default to representative
            if self.select_core_ig_point_in_cluster:
                core_point = self._select_core_ig_point_from_cluster(cluster, known_map_snapshot)
                if core_point:
                    # Now, ensure this core_point is reachable and get path to it
                    path_to_core_point = self._is_reachable_and_get_path(robot_pos, core_point, known_map_snapshot)
                    if path_to_core_point:
                        actual_target_in_cluster = core_point
                        # Cost should be to this new actual_target_in_cluster
                        cost_to_rep = len(path_to_core_point) -1 # Update cost
                        if cost_to_rep <=0: cost_to_rep = 1e-5
                        path_to_rep = path_to_core_point # Update path
                    # else: core point not reachable, stick with original representative
            
            chosen_points_per_cluster_for_viz[idx] = actual_target_in_cluster


            # Robot's utility for choosing this cluster (targeting actual_target_in_cluster)
            pheromone_at_target = self.pheromone_map[actual_target_in_cluster[0], actual_target_in_cluster[1]]
            
            # The IG used for utility should be the cluster_agg_ig, as it represents the cluster's potential
            # If actual_target_in_cluster is different from representative_pos_for_cost,
            # cluster_agg_ig was based on representative_pos_for_cost. This is a slight disconnect.
            # For now, use cluster_agg_ig.
            ig_for_utility = cluster_agg_ig 

            w_ig_robot = 1.0; w_pher_robot = 1.0; w_cost_robot = 1.0 # Example weights

            utility = (ig_for_utility ** w_ig_robot) * \
                      (pheromone_at_target ** w_pher_robot) / \
                      ((cost_to_rep if cost_to_rep > 0 else 1e-5) ** w_cost_robot)

            if utility > best_utility:
                best_utility = utility
                best_target_for_robot = actual_target_in_cluster 
                best_path_for_robot = path_to_rep

        planning_info_for_viz = {
            'frontier_clusters': frontier_clusters,
            'cluster_representatives': chosen_points_per_cluster_for_viz # Now stores the actual target considered
        }

        if best_target_for_robot:
            self._last_chosen_frontier_target = best_target_for_robot 
            return best_target_for_robot, best_path_for_robot, planning_info_for_viz
        else:
            self._last_chosen_frontier_target = None
            # Even if fallback, return cluster info if available for visualization
            fallback_target, fallback_path = self._final_fallback_plan(robot_pos, known_map)
            return fallback_target, fallback_path, planning_info_for_viz