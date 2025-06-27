import numpy as np
import random
try:
    from environment_SLAM import FREE, OBSTACLE, LANDMARK, UNKNOWN
except ImportError:
    FREE, OBSTACLE, LANDMARK, UNKNOWN = 1, 2, 3, 0

class RobotSlamSimplified:
    def __init__(self, start_pos, sensor_range=5,
                 # Motion noise: std dev of noise added to intended move
                 motion_noise_std=0.1,  
                 # Observation noise: std dev of noise added to relative landmark position
                 observation_pos_noise_std=0.2, 
                 observation_noise_prob_flip=0.01):
        
        # True state (God's view)
        self.true_pos = np.array(start_pos, dtype=float) 
        
        # Robot's estimated state (belief)
        self.believed_pos = np.copy(self.true_pos)
        self.pose_covariance = np.eye(2) * 0.01 # Start with very low uncertainty
        
        self.actual_pos_history = [list(self.believed_pos)]

        self.sensor_range = sensor_range
        self.motion_noise_std = motion_noise_std 
        self.observation_pos_noise_std = observation_pos_noise_std
        self.observation_noise_prob_flip = observation_noise_prob_flip
        
        # SLAM landmark database: { landmark_id: {'pos': est_pos, 'cov': est_cov} }
        self.landmark_database = {}

    def get_position(self):
        """Returns the robot's believed integer grid position for high-level planning."""
        return tuple(self.believed_pos.astype(int))

    def sense(self, environment):
        """Senses surroundings, updates map, and returns landmark observations."""
        observed_landmarks_in_frame = []
        r_believed, c_believed = self.believed_pos[0], self.believed_pos[1]
        
        # Simulate sensing in a square around the true position
        for dr in range(-self.sensor_range, self.sensor_range + 1):
            for dc in range(-self.sensor_range, self.sensor_range + 1):
                # Target cell in true world coordinates
                r_target_true, c_target_true = int(round(self.true_pos[0] + dr)), int(round(self.true_pos[1] + dc))
                if not environment.is_within_bounds(r_target_true, c_target_true): continue
                
                true_state = environment.get_true_map_state(r_target_true, c_target_true)
                is_obstacle_type = (true_state in [OBSTACLE, LANDMARK])

                # Apply observation noise (misclassification)
                if random.random() < self.observation_noise_prob_flip:
                    is_obstacle_obs = not is_obstacle_type
                else:
                    is_obstacle_obs = is_obstacle_type

                # Update probabilistic map based on robot's believed position
                r_target_believed = int(round(r_believed + dr))
                c_target_believed = int(round(c_believed + dc))
                if environment.is_within_bounds(r_target_believed, c_target_believed):
                    environment.update_probabilistic_map(r_target_believed, c_target_believed, is_obstacle_obs)

                # If the true cell is a landmark and we didn't misread it
                if true_state == LANDMARK and is_obstacle_obs:
                    landmark_id = (r_target_true, c_target_true) # True position is its unique ID
                    
                    # Robot observes a noisy version of the landmark's position relative to its BELIEVED pose
                    # True relative vector:
                    true_relative_vec = np.array([r_target_true, c_target_true]) - self.true_pos
                    # Add noise to the observation
                    obs_noise = np.random.normal(0, self.observation_pos_noise_std, size=2)
                    noisy_relative_vec = true_relative_vec + obs_noise
                    
                    # Robot calculates landmark's world position based on its own believed pose
                    observed_landmark_world_pos = self.believed_pos + noisy_relative_vec
                    
                    observed_landmarks_in_frame.append({'id': landmark_id, 'pos': observed_landmark_world_pos})
        return observed_landmarks_in_frame
    
    def attempt_move_one_step(self, next_step_pos_int, environment):
        """Moves robot with noise, updates believed pose and covariance."""
        # 1. Calculate intended move vector based on integer grid steps
        intended_move_vec = np.array(next_step_pos_int) - self.get_position()

        # 2. Add Motion Noise to create the actual move vector
        noise_vec = np.random.normal(0, self.motion_noise_std, size=2)
        actual_move_vec = intended_move_vec + noise_vec
        
        # 3. Calculate new true position
        new_true_pos = self.true_pos + actual_move_vec

        # 4. Check for collision at the true new position
        r_true_int, c_true_int = int(round(new_true_pos[0])), int(round(new_true_pos[1]))
        if not environment.is_within_bounds(r_true_int, c_true_int) or \
           environment.get_true_map_state(r_true_int, c_true_int) in [OBSTACLE, LANDMARK]:
            self.pose_covariance += np.eye(2) * 0.01 # Uncertainty increases slightly on failed move
            return False 

        # 5. Update true position and robot's belief
        self.true_pos = new_true_pos
        # Robot's belief is updated by the intended (odometry) move, NOT the noisy true move
        self.believed_pos = self.believed_pos + intended_move_vec 
        self.actual_pos_history.append(list(self.believed_pos))

        # Increase pose uncertainty due to motion (simplified model)
        motion_cov_increase = np.eye(2) * (self.motion_noise_std * np.linalg.norm(intended_move_vec))**2
        self.pose_covariance += motion_cov_increase
        self.pose_covariance = np.clip(self.pose_covariance, -10, 10) # Prevent crazy values
        return True

    def correct_pose_and_landmarks(self, landmark_id, observed_pos):
        """Simplified SLAM correction step (like a simple Kalman Filter update)."""

        known_landmark_data = self.landmark_database[landmark_id]
        believed_landmark_pos = known_landmark_data['pos']
        innovation = observed_pos - believed_landmark_pos
        
        correction_factor = 0.1 
        self.believed_pos -= innovation * correction_factor
        
        self.pose_covariance *= (1.0 - correction_factor * 0.5) 
        
        known_landmark_data['pos'] = (known_landmark_data['pos'] * (1-correction_factor) + observed_pos * correction_factor)
        known_landmark_data['cov'] *= (1.0 - correction_factor * 0.5)