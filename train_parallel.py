import os
import argparse
import itertools
import numpy as np
import gymnasium as gym
import glob
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

from env.n_pendulums_env import NPendulumEnv
from env.physics_utils import inches_to_meters, compute_masses, compute_max_viscous_friction

class CurriculumCallback(BaseCallback):
    def __init__(self, start_noise: float = 0.05, max_noise: float = np.pi / 4.0, 
                 flat_phase_steps: int = 1_000_000, 
                 ramp_phase_steps: int = 98_000_000, 
                 verbose: int = 0):
        super().__init__(verbose)
        self.start_noise = start_noise
        self.max_noise = max_noise
        self.flat_phase_steps = flat_phase_steps
        self.ramp_phase_steps = ramp_phase_steps

    def _on_step(self) -> bool:
        ramp_start = self.flat_phase_steps
        ramp_end = self.flat_phase_steps + self.ramp_phase_steps
        
        if self.num_timesteps <= ramp_start:
            current_noise = self.start_noise
            current_offset = 0.0
            early_term = True
        elif self.num_timesteps >= ramp_end:
            current_noise = self.max_noise
            current_offset = np.pi
            early_term = False
        else:
            fraction = (self.num_timesteps - ramp_start) / self.ramp_phase_steps
            current_noise = fraction * self.max_noise
            current_offset = np.pi
            early_term = False
            
        self.training_env.env_method("set_init_noise", current_noise, current_offset)
        self.training_env.env_method("set_early_termination", early_term)
            
        return True

class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                if "episode_max_angle_diff" in info:
                    self.logger.record("episode_metrics/max_angle_diff", info["episode_max_angle_diff"])
                    self.logger.record("episode_metrics/max_joint_vel", info["episode_max_joint_vel"])
                    self.logger.record("episode_metrics/max_cart_pos_perc", info["episode_max_cart_pos_perc"])
                    if "init_noise" in info:
                        self.logger.record("episode_metrics/init_noise", info["init_noise"])
                        self.logger.record("episode_metrics/init_offset", info["init_offset"])
        return True

class KeepLatestCheckpointsCallback(CheckpointCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = 'rl_model', keep_last: int = 5, verbose: int = 0):
        super().__init__(save_freq=save_freq, save_path=save_path, name_prefix=name_prefix, verbose=verbose)
        self.keep_last = keep_last

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            search_pattern = os.path.join(self.save_path, f"{self.name_prefix}_*_steps.zip")
            checkpoint_files = glob.glob(search_pattern)
            checkpoint_files.sort(key=os.path.getmtime)
            while len(checkpoint_files) > self.keep_last:
                try:
                    os.remove(checkpoint_files.pop(0))
                except OSError:
                    pass
        return continue_training

def make_env(rank, seed=0, **env_kwargs):
    """
    Utility function for multiprocessed env.
    
    :param rank: (int) index of the subprocess
    :param seed: (int) the initial seed for RNG
    :param env_kwargs: (dict) keyword arguments to pass to the environment
    """
    def _init():
        env = NPendulumEnv(**env_kwargs)
        env = Monitor(env)
        # Seed the environment using the seed + rank
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser(description="Parallel training for N-Pendulum Environment")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments to run")
    parser.add_argument("--total_timesteps", type=int, default=200_000_000, help="Total training timesteps")
    parser.add_argument("--n_pendulums", type=int, default=2, help="Number of pendulums in the environment")
    parser.add_argument("--log_dir", type=str, default="./logs/", help="Directory to save logs and checkpoints")
    args = parser.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    # Generate all target configurations (0.0 = UP, np.pi = DOWN)
    # Include all combinations except where all pendulums are pointing down
    all_configs = list(itertools.product([0.0, np.pi], repeat=args.n_pendulums))
    # target_configs = [np.array(config) for config in all_configs if not all(c == np.pi for c in config)]
    target_configs = [np.array(config) for config in all_configs if all(c == 0 for c in config)]

    # Define environment parameters
    env_kwargs = {
        "n_pendulums": args.n_pendulums,
        "viscous_friction": 0.05,
        "pole_length": 2.4384,
        "cart_sigma": 0.48768,
        "edge_spring_k": 500.0,
        "target_configs": target_configs,
        "early_termination_allowed": True,
    }

    # Calculate your starting viscosity based on zeta = 0.45 (highly stable, but controllable)
    max_viscosity = compute_max_viscous_friction(zeta=0.45, env_kwargs=env_kwargs)
    print(f"Computed Curriculum Starting Friction: {max_viscosity:.4f}")
    # env_kwargs["viscous_friction"] = max_viscosity

    # Initialize a temporary environment to resolve all defaults and dynamic physics constraints
    temp_env = NPendulumEnv(**env_kwargs)
    fully_resolved_kwargs = temp_env.get_env_kwargs()
    temp_env.close()

    # # Compute dynamic kwargs
    # densities = {
    #     'aluminum': 2700.0,
    #     'steel': 7850.0,
    #     'brass': 8500.0,
    #     'copper': 8960.0,
    #     'titanium': 4500.0,
    #     'tungsten': 19300.0,
    # }
    # masses = compute_masses(
    #     temp_env,
    #     cross_sectional_area=np.pi * inches_to_meters(1.0) ** 2,
    #     density=densities['aluminum']
    # )
    # print(f"Computed Masses: {masses}")
    # env_kwargs["masses"] = list(masses)


    # # WE COMPUTED DYNAMIC KWARGS, SO WE NEED TO RE-RESOLVE THE KWARGS
    # temp_env = NPendulumEnv(**env_kwargs)
    # fully_resolved_kwargs = temp_env.get_env_kwargs()
    # temp_env.close()


    with open(os.path.join(args.log_dir, "env_config.json"), "w") as f:
        json.dump(fully_resolved_kwargs, f, indent=4)

    print(f"Initializing {args.num_envs} parallel environments with {args.n_pendulums} pendulums...")
    
    # SubprocVecEnv runs each environment in a separate process
    vec_env = SubprocVecEnv([make_env(i, **env_kwargs) for i in range(args.num_envs)])
    
    # Save checkpoints periodically
    checkpoint_callback = KeepLatestCheckpointsCallback(
        save_freq=max(100_000 // args.num_envs, 1),
        save_path=args.log_dir,
        name_prefix=f'ppo_npendulum_{args.n_pendulums}',
        keep_last=10,
    )
    
    curriculum_callback = CurriculumCallback(
        flat_phase_steps=1_000_000,
        ramp_phase_steps=98_000_000
    )
    
    tensorboard_callback = TensorboardLoggingCallback()
    
    callback_list = CallbackList([checkpoint_callback, curriculum_callback, tensorboard_callback])

    print("Creating PPO model...")
    # Initialize the PPO agent. The algorithm can be changed to SAC, TD3, etc.
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(args.log_dir, "tensorboard")
    )

    print("Starting training...")
    try:
        model.learn(total_timesteps=args.total_timesteps, callback=callback_list)
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final model...")
    finally:
        # Save the final model
        final_model_path = os.path.join(args.log_dir, f"ppo_npendulum_{args.n_pendulums}_final")
        model.save(final_model_path)
        vec_env.close()
        print(f"Done. Final model saved to {final_model_path}.zip")

if __name__ == "__main__":
    main()
