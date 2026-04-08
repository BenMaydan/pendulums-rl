import os
import argparse
import itertools
import numpy as np
import gymnasium as gym
import glob
import json
from collections import deque

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

from env.n_pendulums_env import NPendulumEnv
from env.physics_utils import inches_to_meters, compute_masses, compute_max_viscous_friction

class CurriculumCallback(BaseCallback):
    def __init__(self, start_noise: float = 0.05, max_noise: float = np.pi / 4.0, 
                 flat_phase_steps: int = 1_000_000, 
                 ramp_phase_steps: int = 98_000_000, 
                 start_gravity: float = 1.0, max_gravity: float = 9.81,
                 verbose: int = 0):
        super().__init__(verbose)
        self.start_noise = start_noise
        self.max_noise = max_noise
        self.flat_phase_steps = flat_phase_steps
        self.ramp_phase_steps = ramp_phase_steps
        self.start_gravity = start_gravity
        self.max_gravity = max_gravity
        self.last_noise = None
        self.last_offset = None
        self.last_gravity = None
        self.last_early_term = None

    def _on_step(self) -> bool:
        # Check every 1000 steps to alleviate IPC bottleneck
        if self.num_timesteps % 1000 != 0 and self.num_timesteps > 1:
            return True
            
        ramp_start = self.flat_phase_steps
        ramp_end = self.flat_phase_steps + self.ramp_phase_steps
        
        if self.num_timesteps <= ramp_start:
            current_noise = self.start_noise
            current_offset = 0.0
            current_gravity = self.max_gravity
            early_term = True
        elif self.num_timesteps >= ramp_end:
            current_noise = self.max_noise
            current_offset = np.pi
            current_gravity = self.max_gravity
            early_term = False
        else:
            fraction = (self.num_timesteps - ramp_start) / self.ramp_phase_steps
            current_noise = fraction * self.max_noise
            current_offset = np.pi
            current_gravity = self.start_gravity + fraction * (self.max_gravity - self.start_gravity)
            early_term = False
            
        if self.last_noise != current_noise or self.last_offset != current_offset:
            self.training_env.env_method("set_init_noise", current_noise, current_offset)
            self.last_noise = current_noise
            self.last_offset = current_offset
            
        if self.last_gravity != current_gravity:
            self.training_env.env_method("set_gravity", current_gravity)
            self.last_gravity = current_gravity
            
        if self.last_early_term != early_term:
            self.training_env.env_method("set_early_termination", "angle", early_term)
            self.last_early_term = early_term
            
        return True

class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.termination_history = deque(maxlen=100)
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                is_terminated = info.get("is_terminated", False)
                self.termination_history.append(float(is_terminated))
                
                if "episode_max_angle_diff" in info:
                    self.logger.record("episode_metrics/max_angle_diff", info["episode_max_angle_diff"])
                    self.logger.record("episode_metrics/max_joint_vel", info["episode_max_joint_vel"])
                    self.logger.record("episode_metrics/max_cart_pos_perc", info["episode_max_cart_pos_perc"])
                    if "init_noise" in info:
                        self.logger.record("episode_metrics/init_noise", info["init_noise"])
                        self.logger.record("episode_metrics/init_offset", info["init_offset"])
                        self.logger.record("episode_metrics/gravity", info.get("gravity", 9.81))
        
        if len(self.termination_history) > 0:
            self.logger.record("episode_metrics/termination_rate", sum(self.termination_history) / len(self.termination_history))
            
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

def truncate_tb_logs(tb_log_dir: str, max_step: int):
    """
    Truncates all TensorBoard event files in the given directory to only contain events up to `max_step`.
    This prevents overlapping graphs when training is resumed from a checkpoint and the previous run had progressed further.
    """
    try:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        from tensorboard.summary.writer.record_writer import RecordWriter
    except ImportError:
        print("TensorBoard could not be imported. Skipping log truncation.")
        return

    event_files = glob.glob(os.path.join(tb_log_dir, "**", "events.out.tfevents.*"), recursive=True)
    
    for event_file in event_files:
        if event_file.endswith(".temp"): continue
        
        filtered_events = []
        try:
            for event in EventFileLoader(event_file).Load():
                if event.step <= max_step:
                    filtered_events.append(event)
        except Exception as e:
            print(f"Warning: Could not read {event_file}: {e}")
            continue
            
        temp_file = event_file + ".temp"
        try:
            with open(temp_file, "wb") as f:
                writer = RecordWriter(f)
                for event in filtered_events:
                    writer.write(event.SerializeToString())
            os.replace(temp_file, event_file)
            print(f"Truncated {os.path.basename(event_file)} to step {max_step}")
        except Exception as e:
            print(f"Warning: Could not write truncated logs to {event_file}: {e}")

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
    parser.add_argument("--model_type", type=str, default="PPO", choices=["A2C", "DDPG", "DQN", "PPO", "SAC", "TD3"], help="RL Algorithm to use")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint model to resume from")
    args = parser.parse_args()

    dir_path = os.path.normpath(args.log_dir)
    parent, base = os.path.split(dir_path)
    if base:
        args.log_dir = os.path.join(parent, f"{args.model_type.lower()}_{base}")
    else:
        args.log_dir = os.path.join(parent, f"{args.model_type.lower()}")

    os.makedirs(args.log_dir, exist_ok=True)

    # Generate all target configurations (0.0 = UP, np.pi = DOWN)
    # Include all combinations except where all pendulums are pointing down
    all_configs = list(itertools.product([0.0, np.pi], repeat=args.n_pendulums))
    # target_configs = [np.array(config) for config in all_configs if not all(c == np.pi for c in config)]
    target_configs = [np.array(config) for config in all_configs if all(c == 0 for c in config)]

    # Define environment parameters
    env_kwargs = {
        "n_pendulums": args.n_pendulums,
        "dt": 0.01,
        "viscous_friction": 0.05,
        "pole_length": 5.0,
        "masses": 2.0,
        "lengths": 1.0,
        "inertias": 0.5,
        "cart_sigma": 0.48768,
        "max_cart_vel": 10.0,
        "target_configs": target_configs,
        "early_termination_cart_pos_allowed": True,
        "early_termination_angle_allowed": False,
        "early_termination_angle_vel_allowed": True,
    }

    # Calculate your starting viscosity based on zeta = 0.45 (highly stable, but controllable)
    max_viscosity = compute_max_viscous_friction(zeta=0.45, env_kwargs=env_kwargs)
    print(f"Computed Curriculum Starting Friction: {max_viscosity:.4f}")
    # env_kwargs["viscous_friction"] = max_viscosity

    # Initialize a temporary environment to resolve all defaults and dynamic physics constraints
    temp_env = NPendulumEnv(**env_kwargs)
    fully_resolved_kwargs = temp_env.get_env_kwargs()
    temp_env.close()

    fully_resolved_kwargs["model_type"] = args.model_type
    with open(os.path.join(args.log_dir, "train_config.json"), "w") as f:
        json.dump(fully_resolved_kwargs, f, indent=4)

    print(f"Initializing {args.num_envs} parallel environments with {args.n_pendulums} pendulums...")
    
    # SubprocVecEnv runs each environment in a separate process
    vec_env = SubprocVecEnv([make_env(i, **env_kwargs) for i in range(args.num_envs)])
    
    # Save checkpoints periodically
    checkpoint_callback = KeepLatestCheckpointsCallback(
        save_freq=max(100_000 // args.num_envs, 1),
        save_path=args.log_dir,
        name_prefix=args.model_type.lower(),
        keep_last=10,
    )
    
    curriculum_callback = CurriculumCallback(
        flat_phase_steps=2_000_000,
        ramp_phase_steps=98_000_000
    )
    
    tensorboard_callback = TensorboardLoggingCallback()
    
    callback_list = CallbackList([checkpoint_callback, curriculum_callback, tensorboard_callback])

    # Scale gamma appropriately so the change in dt doesn't mess up anything
    old_gamma = 0.99
    old_dt = 0.02
    new_gamma = old_gamma ** (fully_resolved_kwargs["dt"] / old_dt)
    print(f"New gamma: {new_gamma}")

    print(f"Creating {args.model_type} model...")
    RLClass = {"A2C": A2C, "DDPG": DDPG, "DQN": DQN, "PPO": PPO, "SAC": SAC, "TD3": TD3}[args.model_type]
    
    if args.resume_path:
        print(f"Resuming model from {args.resume_path}...")
        model = RLClass.load(
            args.resume_path,
            env=vec_env,
            custom_objects={"gamma": new_gamma},
            tensorboard_log=os.path.join(args.log_dir, "tensorboard")
        )
        
        # Prevent TensorBoard graph overlaps by truncating old events
        tb_log_dir = os.path.join(args.log_dir, "tensorboard")
        if os.path.exists(tb_log_dir):
            print("Cleaning up old overlapping TensorBoard logs (if any)...")
            truncate_tb_logs(tb_log_dir, model.num_timesteps)
    else:
        # Initialize the agent.
        model = RLClass(
            "MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=os.path.join(args.log_dir, "tensorboard"),
            gamma=new_gamma,
        )

    print("Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps, 
            callback=callback_list,
            reset_num_timesteps=not args.resume_path
        )
    except KeyboardInterrupt:
        print("Training interrupted by user. Saving final model...")
    finally:
        # Save the final model
        final_model_path = os.path.join(args.log_dir, f"{args.model_type.lower()}_final")
        model.save(final_model_path)
        vec_env.close()
        print(f"Done. Final model saved to {final_model_path}.zip")

if __name__ == "__main__":
    main()
