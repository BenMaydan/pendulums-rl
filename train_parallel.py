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
from stable_baselines3.common.callbacks import CheckpointCallback

from env.n_pendulums_env import NPendulumEnv

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
        "pole_length": 1.0,
        "edge_spring_k": 500.0,
        "target_configs": target_configs,
    }

    env_kwargs_to_save = env_kwargs.copy()
    env_kwargs_to_save["target_configs"] = [t.tolist() for t in env_kwargs_to_save["target_configs"]]
    with open(os.path.join(args.log_dir, "env_config.json"), "w") as f:
        json.dump(env_kwargs_to_save, f, indent=4)

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
        model.learn(total_timesteps=args.total_timesteps, callback=checkpoint_callback)
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
