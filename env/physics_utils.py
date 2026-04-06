import numpy as np
from env.n_pendulums_env import NPendulumEnv

def inches_to_meters(inches: float) -> float:
    """
    Converts inches to meters.
    
    Args:
        inches (float): The value in inches.
        
    Returns:
        float: The value in meters.
    """
    return inches * 0.0254

def compute_masses(env: NPendulumEnv, cross_sectional_area: float, density: float) -> np.ndarray:
    """
    Computes the mass of each pendulum in the environment.
    
    Args:
        env (NPendulumEnv): The N-Pendulum environment.
        
    Returns:
        np.ndarray: An array containing the mass of each pendulum.
    """
    return np.array([density * cross_sectional_area * length for length in env.lengths])

def compute_max_viscous_friction(zeta: float, env_kwargs: dict = None) -> float:
    """
    Computes the maximum recommended viscous friction for the N-Pendulum environment 
    given a specific target damping ratio (zeta).
    
    Args:
        zeta (float): The desired damping ratio (e.g. 0.4 to 0.5 for curriculum start).
        env_kwargs (dict): The keyword arguments passed to NPendulumEnv.
        
    Returns:
        float: The maximum viscous friction scalar value to use.
    """
    if env_kwargs is None:
        env_kwargs = {}
        
    # Instantiate a temporary environment to safely resolve any missing 
    # defaults (masses, lengths, inertias, etc.)
    env = NPendulumEnv(**env_kwargs)
    
    masses = env.masses
    inertias = env.inertias
    com_distances = env.com_distances
    g = env.g
    
    c_critical = np.zeros(env.N)
    
    for i in range(env.N):
        # Calculate inertia around the pivot using parallel axis theorem: I_pivot = I_cm + m * h^2
        i_pivot = inertias[i] + masses[i] * (com_distances[i] ** 2)
        
        # Weight equivalent for torque (W = m * h)
        w_i = masses[i] * com_distances[i]
        
        # Critical damping for this joint as an independent simple pendulum: c_c = 2 * sqrt(I_pivot * W * g)
        c_critical[i] = 2.0 * np.sqrt(i_pivot * w_i * g)
        
    env.close()
    
    # The maximum friction per joint is the critical damping scaled by zeta
    c_max = zeta * c_critical
    
    # Return the minimum of the max allowed frictions so that the lightest/fastest 
    # joint doesn't get overdamped if the user applies a single scalar friction to all.
    return float(np.min(c_max))
