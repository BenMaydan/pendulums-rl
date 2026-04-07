import pygame
import math
import sys
import argparse
import numpy as np

WIDTH, HEIGHT = 800, 600
CART_WIDTH, CART_HEIGHT = 80, 40
CART_Y = HEIGHT // 2
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (220, 50, 50)
BLUE = (50, 150, 220)

def main():
    parser = argparse.ArgumentParser(description="Visualize early termination max angular velocities")
    parser.add_argument("--n_pendulums", type=int, default=2, help="Number of pendulums")
    parser.add_argument("--fps", type=int, default=60, help="Frames per second")
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Velocity Limit Visualization N={args.n_pendulums}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    dt = 1.0 / args.fps
    
    # Velocity limits from n_pendulums_env.py: 15.0 * (i + 1)
    vel_limits = np.array([15.0 * (i + 1) for i in range(args.n_pendulums)])
    
    # Start all pendulums at upright (0)
    angles = np.zeros(args.n_pendulums)
    
    # The visualization pole length logic from visualize.py
    # using a fixed pixel proxy length to look clean
    # Typically 1 meter = 60 pixels approx (based on 5m track = ~300px scale)
    # We will just use 100 pixels per joint for easy visibility
    lengths_px = [100.0] * args.n_pendulums

    cart_x = WIDTH // 2

    running = True
    while running:
        clock.tick(args.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update angles based on their maximum velocity
        angles += vel_limits * dt

        screen.fill(WHITE)

        # Draw cart
        cart_rect = pygame.Rect(cart_x - CART_WIDTH//2, CART_Y - CART_HEIGHT//2, CART_WIDTH, CART_HEIGHT)
        pygame.draw.rect(screen, BLUE, cart_rect)
        pygame.draw.rect(screen, BLACK, cart_rect, 2)

        # Draw pendulums
        current_x, current_y = cart_x, CART_Y
        for i, angle in enumerate(angles):
            link_length_px = lengths_px[i]
            
            # Subtracted Y because PyGame's Y-axis goes down, and 0=upright
            next_x = current_x + link_length_px * math.sin(angle)
            next_y = current_y - link_length_px * math.cos(angle)
            
            pygame.draw.line(screen, BLACK, (current_x, current_y), (next_x, next_y), 4)
            pygame.draw.circle(screen, RED, (int(next_x), int(next_y)), 8)
            
            current_x, current_y = next_x, next_y

        # Text overlay
        y_offset = 20
        screen.blit(font.render("Showing Early Termination Maximum Angular Velocities", True, BLACK), (20, y_offset))
        y_offset += 30
        for i, vel in enumerate(vel_limits):
            text = f"Joint {i+1} Limit: {vel:.1f} rad/s"
            screen.blit(font.render(text, True, BLACK), (20, y_offset))
            y_offset += 25
            
        screen.blit(font.render(f"FPS: {args.fps}", True, BLACK), (20, HEIGHT - 30))

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
