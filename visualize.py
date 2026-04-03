import pygame
import math
import sys
import time
import argparse
import numpy as np

from stable_baselines3 import PPO
# Import the custom Gymnasium environment
from env.n_pendulums_env import NPendulumEnv

# --- Constants & Setup ---
WIDTH, HEIGHT = 800, 600
# Set FPS to 50 to match the environment's internal dt = 0.02s
FPS = 50 
LINK_LENGTH_PX = 60 
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

def draw_spring(surface, color, start_pos, end_pos, coils=5, width=10):
    start_x, start_y = start_pos
    end_x, end_y = end_pos
    
    length = end_x - start_x
    if length == 0:
        return
        
    points = [(start_x, start_y)]
    for i in range(1, coils + 1):
        x = start_x + length * (i / (coils + 1))
        y = start_y + (width if i % 2 == 1 else -width)
        points.append((x, y))
        
    points.append((end_x, end_y))
    pygame.draw.lines(surface, color, False, points, 2)


def main():
    parser = argparse.ArgumentParser(description="Visualize N-Pendulum")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained SB3 model (.zip)")
    parser.add_argument("--n_pendulums", type=int, default=3, help="Number of pendulums")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("N-Pendulum AI Environment")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    model = None
    n_pend = args.n_pendulums
    if args.model_path:
        print(f"Loading model from {args.model_path}...")
        model = PPO.load(args.model_path)
        # Infer n_pendulums from the model's expected observation space
        # shape is (2 * n_pendulums + 1,)
        n_pend = (model.observation_space.shape[0] - 1) // 2
        print(f"Inferred n_pendulums = {n_pend} from model")

    # Initialize the actual Gym Environment
    env = NPendulumEnv(n_pendulums=n_pend, viscous_friction=0.05, pole_length=1.0, edge_spring_k=500.0)
    env.reset()
    
    ppm = TRACK_WIDTH_PX / env.pole_length

    ai_mode = False
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

    running = True
    while running:
        clock.tick(FPS)

        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                
                # Toggle AI Mode
                if button_rect.collidepoint(mouse_x, mouse_y):
                    ai_mode = not ai_mode
                
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

        # --- Force Calculation & Physics Step ---
        cart_x = env.state[0]
        cart_v = env.state[env.N + 1]

        if dragging_cart:
            # Instead of manually setting position, use a PD Controller 
            # to calculate the force needed to move the cart to the mouse.
            target_physical_x = pixel_to_physical_x(mouse_pixel_x, ppm)
            
            # Allow dragging slightly past limits to feel the spring
            limit = env.pole_length / 2.0
            target_physical_x = max(-limit - 0.5, min(limit + 0.5, target_physical_x))
            
            kp = 300.0  # Proportional gain (pull strength)
            kd = 30.0   # Derivative gain (dampening)
            
            applied_force = kp * (target_physical_x - cart_x) - kd * cart_v
            
            # Clip physical force bounds, but NOT action spaces here
            applied_force = np.clip(applied_force, -env.max_force, env.max_force)
            
        elif ai_mode:
            if model is not None:
                # Use the trained SB3 policy to predict the action
                action, _states = model.predict(env._get_obs(), deterministic=True)
                # Output from SB3 is now in normalized units [-1.0, 1.0], scale to physical representation for visuals
                applied_force = float(action[0]) * env.max_force
            else:
                # Fallback if no model loaded
                applied_force = 40.0 * math.sin(time.time() * 4.0)
        else:
            applied_force = 0.0

        # Step the actual environment
        # Mathematically re-compress action to [-1.0, 1.0] expected by our normalized environments
        normalized_action_step = applied_force / env.max_force
        action = np.array([normalized_action_step], dtype=np.float32)
        
        # Capture the extra truncated param manually if we want Visualizer to bounce bounds like train process
        _, _, terminated, truncated, _ = env.step(action)

        # Automatically reset if it goes fundamentally off rails (should rarely happen now with springs)
        if terminated or truncated:
            env.reset()
            dragging_cart = False

        # --- Drawing ---
        screen.fill(WHITE)

        # Draw Track (Sliding Pole)
        pole_start = int(WIDTH * 0.05)
        pole_end = int(WIDTH * 0.95)
        pygame.draw.line(screen, GRAY, (pole_start, CART_Y), (pole_end, CART_Y), 6)
        
        # Draw Spring Anchors
        anchor_left = TRACK_START_X - 60
        anchor_right = TRACK_END_X + 60
        pygame.draw.line(screen, BLACK, (anchor_left, CART_Y - 20), (anchor_left, CART_Y + 20), 4)
        pygame.draw.line(screen, BLACK, (anchor_right, CART_Y - 20), (anchor_right, CART_Y + 20), 4)

        # Draw Cart
        cart_pixel_x = physical_to_pixel_x(env.state[0], ppm)
        
        # Left Spring
        left_rest_x = TRACK_START_X - CART_WIDTH//2
        left_tip_x = min(left_rest_x, cart_pixel_x - CART_WIDTH//2)
        draw_spring(screen, BLACK, (anchor_left, CART_Y), (left_tip_x, CART_Y), coils=6)

        # Right Spring
        right_rest_x = TRACK_END_X + CART_WIDTH//2
        right_tip_x = max(right_rest_x, cart_pixel_x + CART_WIDTH//2)
        draw_spring(screen, BLACK, (right_tip_x, CART_Y), (anchor_right, CART_Y), coils=6)

        cart_rect = pygame.Rect(cart_pixel_x - CART_WIDTH//2, CART_Y - CART_HEIGHT//2, CART_WIDTH, CART_HEIGHT)
        pygame.draw.rect(screen, BLUE, cart_rect)
        pygame.draw.rect(screen, BLACK, cart_rect, 2)

        # Draw Pendulums
        angles = env.get_joint_angles()
        current_x, current_y = cart_pixel_x, CART_Y
        
        for angle in angles:
            # Subtracting from Y because PyGame's Y-axis goes down, 
            # and angle=0 is the upright (unstable) position in your physics
            next_x = current_x + LINK_LENGTH_PX * math.sin(angle)
            next_y = current_y - LINK_LENGTH_PX * math.cos(angle)
            
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

        # Draw Status Texts
        status_text = f"Mode: {'AI' if ai_mode else 'Manual'} | Cart Pos: {env.state[0]:.3f} m"
        status_surf = font.render(status_text, True, BLACK)
        screen.blit(status_surf, (20, 20))

        # Draw Force Text
        force_text = f"Applied Force: {applied_force:.2f} N"
        force_color = PURPLE if dragging_cart else BLACK 
        force_surf = font.render(force_text, True, force_color)
        screen.blit(force_surf, (20, 50))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()