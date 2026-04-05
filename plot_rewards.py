import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# === 1. Define input variable ranges ===
x_arr = np.linspace(-2.5, 2.5, 500)
dot_arr = np.linspace(-15.0, 15.0, 500)

def get_cart_reward(x, cart_sigma):
    return np.exp(-(x**2) / (2 * cart_sigma**2))

def get_vel_reward(theta_dot, vel_sigma):
    return np.exp(-(theta_dot**2) / (2 * vel_sigma**2))

fig, (ax_cart, ax_vel) = plt.subplots(1, 2, figsize=(12, 6))
plt.subplots_adjust(bottom=0.25, wspace=0.3)

init_cart_sigma = 0.48768
init_vel_sigma = 2.0

# --- Plot Cart ---
line_cart, = ax_cart.plot(x_arr, get_cart_reward(x_arr, init_cart_sigma), lw=2, color='orange')
ax_cart.set_title("Cart Position Reward")
ax_cart.set_xlabel("Cart Position x (m)")
ax_cart.set_ylim(-0.05, 1.05)
ax_cart.grid(True, alpha=0.3)

# --- Plot Velocity ---
line_vel, = ax_vel.plot(dot_arr, get_vel_reward(dot_arr, init_vel_sigma), lw=2, color='purple')
ax_vel.set_title("Angular Velocity Reward")
ax_vel.set_xlabel("Angular Velocity (rad/s)")
ax_vel.set_ylim(-0.05, 1.05)
ax_vel.grid(True, alpha=0.3)

# Add Sliders
axcolor = 'lightgoldenrodyellow'
ax_cart_s   = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
ax_vel_s    = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)

s_cart_sigma    = Slider(ax_cart_s, 'cart_sigma', 0.01, 2.0, valinit=init_cart_sigma)
s_vel_sigma     = Slider(ax_vel_s, 'vel_sigma', 0.1, 10.0, valinit=init_vel_sigma)

def update(val):
    c_cart = s_cart_sigma.val
    c_vel = s_vel_sigma.val
    
    line_cart.set_ydata(get_cart_reward(x_arr, c_cart))
    line_vel.set_ydata(get_vel_reward(dot_arr, c_vel))
    
    fig.canvas.draw_idle()

s_cart_sigma.on_changed(update)
s_vel_sigma.on_changed(update)

plt.suptitle("N-Pendulum Reward Explorer (Cart & Velocity)", fontsize=16)
plt.show()
