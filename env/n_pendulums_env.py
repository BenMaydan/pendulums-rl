import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

class NPendulumEnv(gym.Env):
    """
    N-Pendulum on a Motor-Driven Cart Environment using RK4 Integration.
    The cart's movement dictates the pendulums' movement, but the pendulums 
    do not exert a back-force on the cart.
    """
    def __init__(self, 
                 n_pendulums=2,
                 cart_mass=1.0,
                 masses:None | int | float | NDArray[np.float64]=None,
                 lengths:None | int | float | NDArray[np.float64]=None,
                 com_distances:None | int | float | NDArray[np.float64]=None,
                 inertias:None | int | float | NDArray[np.float64]=None,
                 viscous_friction:None | int | float | NDArray[np.float64]=0.05,
                 target_configs:None | NDArray[np.float64]=None,
                 cart_sigma=0.48768,
                 config_sigma=1.0,
                 config_cos_weight=0.3,
                 vel_sigma=1.5,
                 dt=0.02,
                 pole_length=2.4384,
                 max_g=2.0,
                 max_cart_vel=1.0,
                 reward_weight_angle=0.6,
                 reward_weight_vel=0.3,
                 reward_weight_cart=0.1,
                 early_termination_cart_pos_allowed=False,
                 early_termination_angle_allowed=False,
                 early_termination_angle_vel_allowed=False,
    ):
        super(NPendulumEnv, self).__init__()
        
        self.N = n_pendulums
        self.cart_mass = cart_mass
        self.dt = dt
        self.g = 9.81

        m_def = 0.88
        l_def = 0.5
        
        # Dynamic property resolution
        if masses is None:
            self.masses = np.full(self.N, m_def, dtype=np.float64)
        elif isinstance(masses, (int, float)):
            self.masses = np.full(self.N, float(masses), dtype=np.float64)
        else:
            self.masses = np.array(masses, dtype=np.float64)
            
        if lengths is None:
            self.lengths = np.full(self.N, l_def, dtype=np.float64)
        elif isinstance(lengths, (int, float)):
            self.lengths = np.full(self.N, float(lengths), dtype=np.float64)
        else:
            self.lengths = np.array(lengths, dtype=np.float64)
            
        if com_distances is None:
            self.com_distances = self.lengths / 2.0
        elif isinstance(com_distances, (int, float)):
            self.com_distances = np.full(self.N, float(com_distances), dtype=np.float64)
        else:
            self.com_distances = np.array(com_distances, dtype=np.float64)
            
        if inertias is None:
            self.inertias = (1.0 / 12.0) * self.masses * (self.lengths ** 2)
        elif isinstance(inertias, (int, float)):
            self.inertias = np.full(self.N, float(inertias), dtype=np.float64)
        else:
            self.inertias = np.array(inertias, dtype=np.float64)
        
        if viscous_friction is None:
            self.viscous_friction = np.full(self.N, 0.05, dtype=np.float64)
        elif isinstance(viscous_friction, (int, float)):
            self.viscous_friction = np.full(self.N, float(viscous_friction), dtype=np.float64)
        else:
            self.viscous_friction = np.array(viscous_friction, dtype=np.float64)
        
        # Reward parameters
        self.target_configs = target_configs if target_configs is not None else [np.zeros(self.N)]
        self.config_sigma = config_sigma
        self.config_cos_weight = config_cos_weight
        self.cart_sigma = cart_sigma
        
        self.pole_length = pole_length
        
        self.vel_sigma = vel_sigma

        # reward weighting for sum of bell curves
        self.reward_weight_angle = reward_weight_angle
        self.reward_weight_vel = reward_weight_vel
        self.reward_weight_cart = reward_weight_cart

        self.early_termination_cart_pos_allowed = early_termination_cart_pos_allowed
        self.early_termination_angle_allowed = early_termination_angle_allowed
        self.early_termination_angle_vel_allowed = early_termination_angle_vel_allowed
        
        # Action space: Normalized [-1.0, 1.0]. Scaled to physical force in step()
        # Max force calculated from configurable max_g and total mass
        self.max_g = max_g
        self.max_cart_vel = max_cart_vel
        total_mass = self.cart_mass + np.sum(self.masses)
        self.max_force = self.max_g * self.g * total_mass
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
        # Observation space: time step t-1 and time step t.
        # Each step: cart pos (1), cart vel (1), sin(theta) (N), cos(theta) (N), angular vel (N), sin(target) (N), cos(target) (N), gravity (1).
        # Shape is (2 * (5 * self.N + 3),)
        obs_size = 5 * self.N + 3
        high = np.inf * np.ones(2 * obs_size, dtype=np.float64)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)
        
        self.state = None
        self.prev_features = np.zeros(obs_size, dtype=np.float64)
        self.current_step_count = 0
        self.max_steps = int(10.0 / self.dt)
        self.current_target_config = self.target_configs[0] if self.target_configs else np.zeros(self.N)
        self.current_init_noise = 0.05  # Curriculum learning: starts at 0.05
        self.current_init_offset = 0.0
        self.eval_mode = False
        
        self.ep_max_angle_diff = 0.0
        self.ep_max_joint_vel = 0.0
        self.ep_max_cart_pos_perc = 0.0
        
        self._precompute_constants()

    def get_env_kwargs(self):
        """Returns the fully resolved environment configuration for saving."""
        return {
            "n_pendulums": self.N,
            "cart_mass": self.cart_mass,
            "masses": self.masses.tolist(),
            "lengths": self.lengths.tolist(),
            "com_distances": self.com_distances.tolist(),
            "inertias": self.inertias.tolist(),
            "viscous_friction": self.viscous_friction.tolist(),
            "target_configs": [list(cfg) for cfg in self.target_configs],
            "cart_sigma": self.cart_sigma,
            "config_sigma": self.config_sigma,
            "config_cos_weight": self.config_cos_weight,
            "vel_sigma": self.vel_sigma,
            "dt": self.dt,
            "pole_length": self.pole_length,
            "max_g": self.max_g,
            "max_cart_vel": self.max_cart_vel,
        }

    def get_target_config(self):
        """Returns the current target configuration for the environment."""
        return self.current_target_config

    def set_target_config(self, target):
        """Sets the current target configuration."""
        self.current_target_config = target

    def set_eval(self):
        """Disables step truncation for continuous evaluation/visualization."""
        self.eval_mode = True

    def set_train(self):
        """Enables standard step truncation for training."""
        self.eval_mode = False

    def set_init_noise(self, noise, offset=0.0):
        """Updates the initialization noise spread dynamically for curriculum learning."""
        self.current_init_noise = noise
        self.current_init_offset = offset

    def set_gravity(self, g: float):
        """Updates the gravity constant dynamically for curriculum learning."""
        self.g = g
        total_mass = self.cart_mass + np.sum(self.masses)
        self.max_force = self.max_g * self.g * total_mass

    def set_early_termination(self, type: str, allowed: bool):
        """Enable or disable early termination dynamically."""
        assert type in ["cart_pos", "angle", "angle_vel"]
        match type:
            case "cart_pos":
                self.early_termination_cart_pos_allowed = allowed
            case "angle":
                self.early_termination_angle_allowed = allowed
            case "angle_vel":
                self.early_termination_angle_vel_allowed = allowed

    def _get_features(self):
        """Builds the features for a single time step."""
        if self.state is None:
            return np.zeros(5 * self.N + 2, dtype=np.float64)
            
        x = self.state[0]
        x_dot = self.state[self.N+1]
        theta = self.state[1:self.N+1]
        theta_dot = self.state[self.N+2:]
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        cos_target = np.cos(self.current_target_config)
        sin_target = np.sin(self.current_target_config)
        
        # Normalize x position to roughly [-1.0, 1.0] based on the track limits.
        # This keeps the neural network's input stable regardless of physical track size.
        limit = max(self.pole_length / 2.0, 0.001)  # Prevent division by zero
        x_norm = x / limit
        
        # TODO: decide whether to keep this scaling or not.
        # Scale velocities to be dynamically invariant to changing gravity.
        # A pendulum naturally swings sqrt(g) times faster as gravity increases.
        velocity_scale = np.sqrt(9.81 / max(self.g, 0.001))
        scaled_x_dot = x_dot * velocity_scale
        scaled_theta_dot = theta_dot * velocity_scale
        
        # We append scaled gravity so the neural network learns a smooth contextual policy
        g_norm = self.g / 9.81
        
        return np.concatenate(([x_norm, scaled_x_dot], sin_theta, cos_theta, scaled_theta_dot, sin_target, cos_target, [g_norm]))

    def _get_obs(self):
        """Builds the observation by concatenating t-1 and t features."""
        current_features = self._get_features()
        return np.concatenate((self.prev_features, current_features))

    def _precompute_constants(self):
        """Precomputes mass distribution matrices for the RK4 integration."""
        d = np.zeros((self.N, self.N))
        for k in range(self.N):
            for m in range(self.N):
                if m < k:
                    d[k, m] = self.lengths[m]
                elif m == k:
                    d[k, m] = self.com_distances[k]
                else:
                    d[k, m] = 0.0

        self.W = np.zeros(self.N)
        for i in range(self.N):
            self.W[i] = np.sum(self.masses * d[:, i])

        self.M_prime = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.M_prime[i, j] = np.sum(self.masses * d[:, i] * d[:, j])

    def _get_derivatives(self, state, action):
        """Builds the decoupled M matrix and F vector, returning state derivatives."""
        x, theta = state[0], state[1:self.N+1]
        x_dot, theta_dot = state[self.N+1], state[self.N+2:]
        
        # 1. Decoupled Cart Equation
        total_cart_force = action
        
        limit = self.pole_length / 2.0

        # Hard stop logic for cart acceleration
        if x <= -limit and x_dot <= 0 and total_cart_force < 0:
            x_ddot = 0.0
            x_dot = 0.0
        elif x >= limit and x_dot >= 0 and total_cart_force > 0:
            x_ddot = 0.0
            x_dot = 0.0
        else:
            x_ddot = total_cart_force / self.cart_mass
        
        # 2. Pendulum Equations (N x N System)
        M_pend = np.zeros((self.N, self.N))
        F_pend = np.zeros(self.N)
        
        for i in range(self.N):
            for j in range(self.N):
                # Construct the reduced mass matrix for pendulums only
                M_pend[i, j] = self.M_prime[i, j] * np.cos(theta[i] - theta[j])
                if i == j:
                    M_pend[i, j] += self.inertias[i]
                    
            # 3. Forcing terms on the pendulums
            centrifugal = np.sum(self.M_prime[i, :] * np.sin(theta[i] - theta) * theta_dot**2)
            gravity = self.W[i] * self.g * np.sin(theta[i])
            
            # The cart's acceleration acts as a fictitious force pushing the pendulums
            cart_forcing = -self.W[i] * np.cos(theta[i]) * x_ddot
            
            F_pend[i] = cart_forcing - centrifugal + gravity
            
            # Add Viscous Friction (air resistance / bearing damping)
            rel_vel = theta_dot[i] - (theta_dot[i-1] if i > 0 else 0.0)
            F_pend[i] -= self.viscous_friction[i] * rel_vel
            if i < self.N - 1:
                rel_vel_next = theta_dot[i+1] - theta_dot[i]
                F_pend[i] += self.viscous_friction[i+1] * rel_vel_next
            
        # 4. Solve for angular accelerations: M_pend * theta_ddot = F_pend
        theta_ddot = np.linalg.solve(M_pend, F_pend)
        
        # Return state derivative: [velocities, accelerations]
        return np.concatenate(([x_dot], theta_dot, [x_ddot], theta_ddot))

    def _rk4_step(self, state, action):
        """Performs a single Runge-Kutta 4th order integration step."""
        k1 = self._get_derivatives(state, action)
        k2 = self._get_derivatives(state + 0.5 * self.dt * k1, action)
        k3 = self._get_derivatives(state + 0.5 * self.dt * k2, action)
        k4 = self._get_derivatives(state + self.dt * k3, action)
        
        return state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def step(self, action):
        """Applies the motor force, integrates physics, and calculates rewards."""
        self.prev_features = self._get_features()
        
        # Scale the normalized action [-1.0, 1.0] to a physical force
        normalized_action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        force = normalized_action * self.max_force
        
        # Apply RK4 integration
        self.state = self._rk4_step(self.state, force)
        
        # Enforce hard stop position and velocity
        limit = self.pole_length / 2.0
        if self.state[0] <= -limit:
            self.state[0] = -limit
            if self.state[self.N+1] < 0:
                self.state[self.N+1] = 0.0
        elif self.state[0] >= limit:
            self.state[0] = limit
            if self.state[self.N+1] > 0:
                self.state[self.N+1] = 0.0
                
        # Check for numerical instability (NaN, Inf, or extremely large values)
        if not np.all(np.isfinite(self.state)) or np.any(np.abs(self.state) > 1e6):
            self.state = np.zeros_like(self.state)
            return self._get_obs(), 0.0, True, False, {"error": "numerical instability"}
        
        self.current_step_count += 1
        x = self.state[0]
        theta = self.state[1:self.N+1]
        theta_dot = self.state[self.N+2:]
        
        # Calculate angle diff from target
        diff = (theta - self.current_target_config + np.pi) % (2 * np.pi) - np.pi

        # Update episode max metrics
        max_diff = np.max(np.abs(diff))
        max_vel = np.max(np.abs(theta_dot))
        cart_limit = max(self.pole_length / 2.0, 0.001)
        cart_pos_perc = (abs(x) / cart_limit) * 100.0

        self.ep_max_angle_diff = max(self.ep_max_angle_diff, float(max_diff))
        self.ep_max_joint_vel = max(self.ep_max_joint_vel, float(max_vel))
        self.ep_max_cart_pos_perc = max(self.ep_max_cart_pos_perc, float(cart_pos_perc))

        # We do a weighted sum of a bell curve and linear reward to discourage dead gradient
        gauss_reward_angle = np.exp(-np.sum(diff**2) / (2 * self.config_sigma**2))
        linear_reward_angle = np.mean((np.pi - np.abs(diff)) / np.pi)
        reward_angle = self.config_cos_weight * linear_reward_angle + (1 - self.config_cos_weight) * gauss_reward_angle

        reward_cart = np.exp(-x**2 / (2 * self.cart_sigma**2)) # Bell curve for how close the cart is to the center [0 to 1]

        # TODO: decide whether to keep this scaling or not.
        # Scale velocity to normalize the penalty across all gravity levels
        velocity_scale = np.sqrt(9.81 / max(self.g, 0.001))
        scaled_theta_dot = theta_dot * velocity_scale
        reward_vel = np.exp(-np.sum(scaled_theta_dot**2) / (2 * self.vel_sigma**2))

        # The key to avoiding the "Valley of Death" local minimum at the bottom:
        # We make the velocity and cart rewards ACT AS MULTIPLIERS to the angle reward.
        # If the pendulum is at the bottom, `reward_angle` is near 0, so the total reward is near 0.
        # As it swings to the top, it unlocks bonuses for being still and centered.
        multiplier = 1.0 + (self.reward_weight_vel * reward_vel) + (self.reward_weight_cart * reward_cart)
        max_multiplier = 1.0 + self.reward_weight_vel + self.reward_weight_cart
        
        total_reward = (reward_angle * multiplier) / max_multiplier * self.dt
        
        # Check early termination if allowed (falling beyond ~90 degrees)
        terminated = False
        early_termination_allowed = self.early_termination_cart_pos_allowed or self.early_termination_angle_allowed or self.early_termination_angle_vel_allowed
        if early_termination_allowed and not self.eval_mode:
            if self.early_termination_cart_pos_allowed and abs(x) > (self.pole_length / 2.0) * 0.95:
                terminated = True
            if self.early_termination_angle_allowed and np.any(np.abs(diff) > np.pi / 4.0):
                terminated = True
            vel_limits = np.array([15.0 * (i + 1) for i in range(self.N)])
            if self.early_termination_angle_vel_allowed and np.any(np.abs(scaled_theta_dot) > vel_limits):
                terminated = True
                
        truncated = False if self.eval_mode else (self.current_step_count >= self.max_steps)
        
        info = {
            "episode_max_angle_diff": self.ep_max_angle_diff,
            "episode_max_joint_vel": self.ep_max_joint_vel,
            "episode_max_cart_pos_perc": self.ep_max_cart_pos_perc,
            "init_noise": self.current_init_noise,
            "init_offset": self.current_init_offset,
            "gravity": self.g,
            "is_terminated": terminated,
        }
        
        return self._get_obs(), total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        
        # Pick a random target configuration every episode
        if len(self.target_configs) > 0:
            idx = self.np_random.integers(0, len(self.target_configs))
            self.current_target_config = self.target_configs[idx]
            
        self.state = np.zeros(2 + 2 * self.N, dtype=np.float64)
        
        self.ep_max_angle_diff = 0.0
        self.ep_max_joint_vel = 0.0
        self.ep_max_cart_pos_perc = 0.0
        
        noise = self.np_random.uniform(low=-self.current_init_noise, high=self.current_init_noise, size=(self.N,))
        self.state[1:self.N+1] = (self.current_target_config + self.current_init_offset + noise + np.pi) % (2 * np.pi) - np.pi
        
        self.prev_features = self._get_features()
        return self._get_obs(), {}

    def get_joint_angles(self):
        """Returns absolute angles for rendering."""
        if self.state is None:
            return np.zeros(self.N)
        return self.state[1:self.N+1]