import pygame
import math
import sys
import time
import argparse
import numpy as np
import os
import glob
import json

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
# Import the custom Gymnasium environment
from env.n_pendulums_env import NPendulumEnv

# --- Constants & Setup ---
WIDTH, HEIGHT = 800, 600
CART_WIDTH, CART_HEIGHT = 80, 40
CART_Y = HEIGHT // 2

# Scale & Coordinate Constants
TRACK_START_X = int(WIDTH * 0.2) 
TRACK_END_X = int(WIDTH * 0.8)   
TRACK_WIDTH_PX = TRACK_END_X - TRACK_START_X
# PIXELS_PER_METER will be calculated dynamically

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (220, 50, 50)
BLUE = (50, 150, 220)
GREEN = (50, 200, 50)
PURPLE = (150, 50, 220)

def physical_to_pixel_x(physical_x, ppm):
    # Map physical x (where 0 is center) to pixel coordinates
    return int((WIDTH / 2) + physical_x * ppm)

def pixel_to_physical_x(pixel_x, ppm):
    # Map pixel coordinates back to physical x [-limit, limit]
    return (pixel_x - (WIDTH / 2)) / ppm

def main():
    parser = argparse.ArgumentParser(description="Visualize N-Pendulum")
    parser.add_argument("--model_path", type=str, default="./logs/", help="Path to the trained SB3 model (.zip) or directory")
    parser.add_argument("--n_pendulums", "-p", type=int, default=2, help="Number of pendulums")
    parser.add_argument("--gravity", "-g", type=float, default=9.81, help="The gravity to simulate at.")
    parser.add_argument("--init_offset", "-o", type=float, default=0, help="The initial angle offset when resetting the pendulums")
    parser.add_argument("--init_noise", "-n", type=float, default=0.05, help="The initial angle noise when resetting the pendulums")
    parser.add_argument("--jitter_prob", type=float, default=None, help="Probability of cart jitter (overrides config)")
    parser.add_argument("--jitter_force", type=float, default=None, help="Force multiplier for cart jitter (overrides config)")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("N-Pendulum AI Environment")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    model = None
    n_pend = args.n_pendulums
    gravity = args.gravity
    env_kwargs = {
        "n_pendulums": n_pend,
        "dt": 0.01,
        "viscous_friction": 0.05,
        "pole_length": 5.0,
        "masses": 2.0,
        "lengths": 1.0,
        "inertias": 0.5,
        "cart_sigma": 0.48768
    }
    
    ai_mode = False
    if args.model_path:
        model_file = args.model_path
        if os.path.isdir(args.model_path):
            zip_files = glob.glob(os.path.join(args.model_path, "*.zip"))
            if zip_files:
                model_file = max(zip_files, key=os.path.getmtime)
                print(f"Found latest model in directory: {model_file}")
            else:
                model_file = None
                print(f"No .zip models found in {args.model_path}")
                
        if model_file and os.path.isfile(model_file):
            # Check for train_config.json in the same directory
            model_dir = os.path.dirname(os.path.abspath(model_file))
            config_path = os.path.join(model_dir, "train_config.json")
            model_type = "PPO"
            if os.path.exists(config_path):
                print(f"Loading environment config from {config_path}...")
                with open(config_path, "r") as f:
                    loaded_config = json.load(f)
                    model_type = loaded_config.get("model_type", "PPO")
                    env_kwargs.update(loaded_config)
                    env_kwargs.pop("model_type", None)
                    if "target_configs" in env_kwargs:
                        env_kwargs["target_configs"] = [np.array(t) for t in env_kwargs["target_configs"]]
                    if "n_pendulums" in env_kwargs:
                        n_pend = env_kwargs["n_pendulums"]
            else:
                env_kwargs["n_pendulums"] = n_pend

            ai_mode = True
            print(f"Loading {model_type} model from {model_file}...")
            RLClass = {"A2C": A2C, "DDPG": DDPG, "DQN": DQN, "PPO": PPO, "SAC": SAC, "TD3": TD3}.get(model_type, PPO)
            model = RLClass.load(model_file)

    # Apply manual CLI overrides to env properties
    if args.jitter_prob is not None:
        env_kwargs["cart_jitter_prob"] = args.jitter_prob
    if args.jitter_force is not None:
        env_kwargs["cart_jitter_force"] = args.jitter_force

    # Initialize the actual Gym Environment
    env = NPendulumEnv(**env_kwargs)
    init_noise, init_offset = args.init_noise, args.init_offset
    env.set_init_noise(init_noise, init_offset)
    env.set_gravity(gravity)
    env.reset()

    if not ai_mode:
        env.set_eval()
    
    ppm = TRACK_WIDTH_PX / env.pole_length

    # Decouple physics framerate from rendering framerate
    target_fps = 60
    physics_steps_per_frame = max(1, int(round((1.0 / target_fps) / env.dt)))
    fps = int(1.0 / (env.dt * physics_steps_per_frame))

    dragging_cart = False
    mouse_pixel_x = WIDTH // 2
    applied_force = 0.0

    button_width, button_height = 120, 40
    button_rect = pygame.Rect(
        WIDTH - button_width - 20, 
        HEIGHT - button_height - 20, 
        button_width, 
        button_height
    )

    term_button_rect = pygame.Rect(
        WIDTH - button_width - 20, 
        HEIGHT - (button_height * 2) - 30, 
        button_width, 
        button_height
    )
    
    reset_button_rect = pygame.Rect(
        WIDTH - button_width - 20, 
        HEIGHT - (button_height * 3) - 40, 
        button_width, 
        button_height
    )
    
    early_term = env.early_termination_angle_allowed or env.early_termination_cart_pos_allowed or env.early_termination_angle_vel_allowed

    running = True
    while running:
        clock.tick(fps)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                
                # Toggle AI Mode
                if button_rect.collidepoint(mouse_x, mouse_y):
                    ai_mode = not ai_mode
                
                # Toggle Early Termination
                if term_button_rect.collidepoint(mouse_x, mouse_y):
                    early_term = not early_term
                    env.set_early_termination("cart_pos", early_term)
                    env.set_early_termination("angle", early_term)
                    env.set_early_termination("angle_vel", early_term)
                    # For early termination to work, we need to ensure eval_mode is False
                    if early_term:
                        env.set_train()
                    elif not ai_mode:
                        env.set_eval()
                
                # Immediate Reset
                if reset_button_rect.collidepoint(mouse_x, mouse_y):
                    env.reset()
                    dragging_cart = False
                    applied_force = 0.0
                
                # Check if clicking on the cart
                cart_px = physical_to_pixel_x(env.state[0], ppm)
                cart_rect = pygame.Rect(cart_px - CART_WIDTH//2, CART_Y - CART_HEIGHT//2, CART_WIDTH, CART_HEIGHT)
                
                if cart_rect.collidepoint(mouse_x, mouse_y) and not ai_mode:
                    dragging_cart = True
                    mouse_pixel_x = mouse_x

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging_cart = False
                applied_force = 0.0 

            elif event.type == pygame.MOUSEMOTION:
                if dragging_cart:
                    mouse_pixel_x = event.pos[0]

        # --- Force Calculation & Physics Step (Multiple Times for Real-Time) ---
        for _ in range(physics_steps_per_frame):
            cart_x = env.state[0]
            cart_v = env.state[env.N + 1]

            if dragging_cart:
                # Instead of manually setting position, use a PD Controller 
                # to calculate the force needed to move the cart to the mouse.
                target_physical_x = pixel_to_physical_x(mouse_pixel_x, ppm)
                
                # Enforce rigid physical drag limits
                limit = env.pole_length / 2.0
                target_physical_x = max(-limit, min(limit, target_physical_x))
                
                kp = 300.0  # Proportional gain (pull strength)
                kd = 30.0   # Derivative gain (dampening)
                
                applied_force = kp * (target_physical_x - cart_x) - kd * cart_v
                
                # Clip physical force bounds, but NOT action spaces here
                applied_force = np.clip(applied_force, -env.max_force, env.max_force)
                
            elif ai_mode:
                if model is not None:
                    # Use the trained SB3 policy to predict the action
                    action, _states = model.predict(env._get_obs(), deterministic=True)
                    # Clip the action identically to how the physics environment does it so the text isn't misleading
                    clipped_action = np.clip(action[0], -1.0, 1.0)
                    # Output from SB3 is now safely in normalized units [-1.0, 1.0], scale to physical representation for visuals
                    applied_force = float(clipped_action) * env.max_force
                else:
                    # Fallback if no model loaded
                    applied_force = env.max_force * math.sin(time.time() * 4.0)
            else:
                applied_force = 0.0

            # Step the actual environment
            # Mathematically re-compress action to [-1.0, 1.0] expected by our normalized environments
            normalized_action_step = applied_force / env.max_force
            action = np.array([normalized_action_step], dtype=np.float32)
            
            # Capture the extra truncated param manually if we want Visualizer to bounce bounds like train process
            _, _, terminated, truncated, _ = env.step(action)

            # Automatically reset if it goes fundamentally off rails (should rarely happen now with hard stops)
            if terminated or truncated:
                env.reset()
                dragging_cart = False
                break

        # --- Drawing ---
        screen.fill(WHITE)

        # Draw Track (Sliding Pole)
        pole_start = int(WIDTH * 0.05)
        pole_end = int(WIDTH * 0.95)
        pygame.draw.line(screen, GRAY, (pole_start, CART_Y), (pole_end, CART_Y), 6)
        
        # Draw Hard Stops
        stop_left = TRACK_START_X - CART_WIDTH//2
        stop_right = TRACK_END_X + CART_WIDTH//2
        pygame.draw.line(screen, BLACK, (stop_left, CART_Y - 20), (stop_left, CART_Y + 20), 8)
        pygame.draw.line(screen, BLACK, (stop_right, CART_Y - 20), (stop_right, CART_Y + 20), 8)

        # Draw Cart
        cart_pixel_x = physical_to_pixel_x(env.state[0], ppm)

        cart_rect = pygame.Rect(cart_pixel_x - CART_WIDTH//2, CART_Y - CART_HEIGHT//2, CART_WIDTH, CART_HEIGHT)
        pygame.draw.rect(screen, BLUE, cart_rect)
        pygame.draw.rect(screen, BLACK, cart_rect, 2)

        # Draw Pendulums
        angles = env.get_joint_angles()
        current_x, current_y = cart_pixel_x, CART_Y
        
        for i, angle in enumerate(angles):
            link_length_px = env.lengths[i] * ppm
            
            # Subtracting from Y because PyGame's Y-axis goes down, 
            # and angle=0 is the upright (unstable) position in your physics
            next_x = current_x + link_length_px * math.sin(angle)
            next_y = current_y - link_length_px * math.cos(angle)
            
            pygame.draw.line(screen, BLACK, (current_x, current_y), (next_x, next_y), 4)
            pygame.draw.circle(screen, RED, (int(next_x), int(next_y)), 8)
            
            current_x, current_y = next_x, next_y

        # Draw AI Button
        btn_color = GREEN if ai_mode else GRAY
        pygame.draw.rect(screen, btn_color, button_rect)
        pygame.draw.rect(screen, BLACK, button_rect, 2)
        
        btn_text = "Stop AI" if ai_mode else "Start AI"
        text_surf = font.render(btn_text, True, BLACK)
        text_rect = text_surf.get_rect(center=button_rect.center)
        screen.blit(text_surf, text_rect)

        # Draw Term Button
        term_color = GREEN if early_term else GRAY
        pygame.draw.rect(screen, term_color, term_button_rect)
        pygame.draw.rect(screen, BLACK, term_button_rect, 2)

        term_text = "Term: ON" if early_term else "Term: OFF"
        term_surf = font.render(term_text, True, BLACK)
        term_rect = term_surf.get_rect(center=term_button_rect.center)
        screen.blit(term_surf, term_rect)

        # Draw Reset Button
        pygame.draw.rect(screen, RED, reset_button_rect)
        pygame.draw.rect(screen, BLACK, reset_button_rect, 2)

        reset_text = "Reset"
        reset_surf = font.render(reset_text, True, WHITE)
        reset_rect = reset_surf.get_rect(center=reset_button_rect.center)
        screen.blit(reset_surf, reset_rect)

        # Draw Status Texts
        status_text = f"Mode: {'AI' if ai_mode else 'Manual'} | Cart Pos: {env.state[0]:.3f} m"
        status_surf = font.render(status_text, True, BLACK)
        screen.blit(status_surf, (20, 20))

        # Draw Force Text
        force_text = f"Applied Force: {applied_force:.2f} N"
        force_color = PURPLE if dragging_cart else BLACK 
        force_surf = font.render(force_text, True, force_color)
        screen.blit(force_surf, (20, 50))

        # Draw Episode Metrics
        metrics_text = f"Max Angle Diff: {env.ep_max_angle_diff:.2f} rad | Max Vel: {env.ep_max_joint_vel:.2f} rad/s | Max Cart: {env.ep_max_cart_pos_perc:.1f}%"
        metrics_surf = font.render(metrics_text, True, BLACK)
        screen.blit(metrics_surf, (20, 80))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()