import gymnasium as gym
import numpy as np
from gymnasium import spaces

class NPendulumEnv(gym.Env):
    """
    N-Pendulum on a Motor-Driven Cart Environment using RK4 Integration.
    The cart's movement dictates the pendulums' movement, but the pendulums 
    do not exert a back-force on the cart.
    """
    def __init__(self, 
                 n_pendulums=2,
                 cart_mass=1.0,
                 masses=None,
                 lengths=None,
                 com_distances=None,
                 inertias=None,
                 viscous_friction=0.0,
                 target_configs=None,
                 config_tolerance_pct=0.05,
                 cart_sigma=0.2,
                 config_sigma=0.1,
                 dt=0.02,
                 pole_length=1.0,
                 edge_spring_k=500.0,
                 edge_spring_damp=10.0):
        super(NPendulumEnv, self).__init__()
        
        self.N = n_pendulums
        self.cart_mass = cart_mass
        self.dt = dt
        self.g = 9.81
        
        # Default physical parameters if none provided
        self.masses = np.array(masses if masses is not None else [1.0] * self.N, dtype=np.float64)
        self.lengths = np.array(lengths if lengths is not None else [1.0] * self.N, dtype=np.float64)
        self.com_distances = np.array(com_distances if com_distances is not None else [0.5] * self.N, dtype=np.float64)
        self.inertias = np.array(inertias if inertias is not None else [0.1] * self.N, dtype=np.float64)
        
        if isinstance(viscous_friction, (int, float)):
            self.viscous_friction = np.full(self.N, viscous_friction, dtype=np.float64)
        else:
            self.viscous_friction = np.array(viscous_friction, dtype=np.float64)
        
        # Reward parameters
        self.target_configs = target_configs if target_configs is not None else [np.zeros(self.N)]
        self.config_tolerance = config_tolerance_pct * 2 * np.pi
        self.config_sigma = config_sigma
        self.cart_sigma = cart_sigma
        
        self.pole_length = pole_length
        self.edge_spring_k = edge_spring_k
        self.edge_spring_damp = edge_spring_damp
        
        # Action space: Normalized [-1.0, 1.0]. Will be scaled up to real physical force inside step()
        self.max_force = 100.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float64)
        
        # Observation space: N angle diffs, N angular velocities, 1 cart position
        # Shape is (2 * self.N + 1,)
        high = np.inf * np.ones(2 * self.N + 1, dtype=np.float64)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float64)
        
        self.state = None
        self.current_step_count = 0
        self.max_steps = int(10.0 / self.dt)
        self.air_time = 0.0
        self.current_target_config = self.target_configs[0] if self.target_configs else np.zeros(self.N)
        self._precompute_constants()

    def get_target_config(self):
        """Returns the current target configuration for the environment."""
        return self.current_target_config

    def set_target_config(self, target):
        """Sets the current target configuration."""
        self.current_target_config = target

    def _get_obs(self):
        """Builds the observation: angle diffs, angular velocities, cart x."""
        if self.state is None:
            return np.zeros(2 * self.N + 1, dtype=np.float64)
            
        x = self.state[0]
        theta = self.state[1:self.N+1]
        theta_dot = self.state[self.N+2:]
        
        # Calculate angle difference from the target config (closest angle)
        angle_diff = (theta - self.current_target_config + np.pi) % (2 * np.pi) - np.pi
        
        return np.concatenate((angle_diff, theta_dot, [x]))

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
        
        # Apply damped spring force if cart goes beyond pole limits
        limit = self.pole_length / 2.0
        if x < -limit:
            spring_force = -self.edge_spring_k * (x - (-limit))
            damping_force = -self.edge_spring_damp * x_dot
            total_cart_force += (spring_force + damping_force)
        elif x > limit:
            spring_force = -self.edge_spring_k * (x - limit)
            damping_force = -self.edge_spring_damp * x_dot
            total_cart_force += (spring_force + damping_force)

        # The motor and springs determine the cart acceleration.
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
        # Scale the normalized action [-1.0, 1.0] mathematically up to physical forces
        normalized_action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        force = normalized_action * self.max_force
        
        # Apply RK4 integration
        self.state = self._rk4_step(self.state, force)
        
        # Check for numerical instability (NaN, Inf, or extremely large values)
        if not np.all(np.isfinite(self.state)) or np.any(np.abs(self.state) > 1e6):
            self.state = np.zeros_like(self.state)
            return self._get_obs(), 0.0, True, False, {"error": "numerical instability"}
        
        self.current_step_count += 1
        x = self.state[0]
        theta = self.state[1:self.N+1]
        
        # Calculate angle diff from target
        diff = (theta - self.current_target_config + np.pi) % (2 * np.pi) - np.pi
        
        if np.all(np.abs(diff) <= self.config_tolerance):
            # Target is met
            self.air_time += 1.0
            
            # Rewarded for staying in the center ONLY when joints are within buffer
            reward_cart = np.exp(-x**2 / (2 * self.cart_sigma**2))
            total_reward = reward_cart + self.air_time
        else:
            # Target is not met
            self.air_time = 0.0
            
            # Punished with linear increase for every joint that is off
            # It increases by how off the angle is from the target (sum of absolute diffs)
            total_reward = -np.sum(np.abs(diff))
        
        # State no longer terminates out-of-bounds due to spring bounce.
        terminated = False
        truncated = self.current_step_count >= self.max_steps
        
        return self._get_obs(), total_reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step_count = 0
        self.air_time = 0.0
        
        # Pick a random target configuration every episode
        if len(self.target_configs) > 0:
            idx = self.np_random.integers(0, len(self.target_configs))
            self.current_target_config = self.target_configs[idx]
            
        self.state = np.zeros(2 + 2 * self.N, dtype=np.float64)
        self.state[1:self.N+1] = self.np_random.uniform(low=-0.05, high=0.05, size=(self.N,))
        return self._get_obs(), {}

    def get_joint_angles(self):
        """Returns absolute angles for rendering."""
        if self.state is None:
            return np.zeros(self.N)
        return self.state[1:self.N+1]