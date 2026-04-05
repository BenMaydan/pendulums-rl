import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# === 1. Setup Data Grid ===
# Reduced resolution to 50x50 to keep 3 simultaneous 3D subplots responsive
diff = np.linspace(-np.pi, np.pi, 50)
D1, D2 = np.meshgrid(diff, diff)

def get_3d_gauss(d1, d2, config_sigma, weight):
    sum_sq = d1**2 + d2**2
    return (1.0 - weight) * np.exp(-sum_sq / (2 * config_sigma**2))

def get_3d_cos(d1, d2, weight):
    cos_1 = (1.0 + np.cos(d1)) / 2.0
    cos_2 = (1.0 + np.cos(d2)) / 2.0
    return weight * ((cos_1 + cos_2) / 2.0)

# === 2. Setup Figure and 3 Subplots ===
fig = plt.figure(figsize=(18, 6))
ax_gauss = fig.add_subplot(131, projection='3d')
ax_cos = fig.add_subplot(132, projection='3d')
ax_sum = fig.add_subplot(133, projection='3d')
plt.subplots_adjust(bottom=0.35, wspace=0.1)

init_sigma = 1.25
init_weight = 0.5

# Calculate initial surfaces
Z_gauss = get_3d_gauss(D1, D2, init_sigma, init_weight)
Z_cos = get_3d_cos(D1, D2, init_weight)
Z_sum = Z_gauss + Z_cos

# Plot surfaces
surf_gauss = [ax_gauss.plot_surface(D1, D2, Z_gauss, cmap='plasma', edgecolor='none')]
surf_cos = [ax_cos.plot_surface(D1, D2, Z_cos, cmap='viridis', edgecolor='none')]
surf_sum = [ax_sum.plot_surface(D1, D2, Z_sum, cmap='inferno', edgecolor='none')]

for ax, title in zip([ax_gauss, ax_cos, ax_sum], ["Gaussian Contribution", "Cosine Contribution", "Hybrid Sum"]):
    ax.set_title(title)
    ax.set_xlabel("Joint 1 Error (rad)")
    ax.set_ylabel("Joint 2 Error (rad)")
    ax.set_zlabel("Reward")
    ax.set_zlim(0, 1.05)

# === 3. Add Interactive Sliders ===
axcolor = 'lightgoldenrodyellow'
ax_sigma_s = plt.axes([0.15, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_weight_s = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)

s_sigma = Slider(ax_sigma_s, 'config_sigma', 0.01, 3.0, valinit=init_sigma)
s_weight = Slider(ax_weight_s, 'Cosine Weight', 0.0, 1.0, valinit=init_weight)

def update(val):
    c_sig = s_sigma.val
    c_weight = s_weight.val
    
    Z_g = get_3d_gauss(D1, D2, c_sig, c_weight)
    Z_c = get_3d_cos(D1, D2, c_weight)
    Z_s = Z_g + Z_c
    
    surf_gauss[0].remove()
    surf_gauss[0] = ax_gauss.plot_surface(D1, D2, Z_g, cmap='plasma', edgecolor='none')
    
    surf_cos[0].remove()
    surf_cos[0] = ax_cos.plot_surface(D1, D2, Z_c, cmap='viridis', edgecolor='none')
    
    surf_sum[0].remove()
    surf_sum[0] = ax_sum.plot_surface(D1, D2, Z_s, cmap='inferno', edgecolor='none')
    
    fig.canvas.draw_idle()

s_sigma.on_changed(update)
s_weight.on_changed(update)

# === 4. Custom View Controls for all subplots ===
azim_angle_val = -60
elev_angle_val = 30
for ax in [ax_gauss, ax_cos, ax_sum]:
    ax.view_init(elev=elev_angle_val, azim=azim_angle_val)

def update_view():
    for ax in [ax_gauss, ax_cos, ax_sum]:
        ax.view_init(elev=elev_angle_val, azim=azim_angle_val)
    fig.canvas.draw_idle()

def azim_plus(event):
    global azim_angle_val
    azim_angle_val += 15
    update_view()

def azim_minus(event):
    global azim_angle_val
    azim_angle_val -= 15
    update_view()

def elev_plus(event):
    global elev_angle_val
    elev_angle_val += 15
    update_view()

def elev_minus(event):
    global elev_angle_val
    elev_angle_val -= 15
    update_view()

# Add rotation buttons
btn_ax_ap = plt.axes([0.3, 0.05, 0.08, 0.05])
btn_ap = Button(btn_ax_ap, 'Azimuth +')
btn_ap.on_clicked(azim_plus)

btn_ax_am = plt.axes([0.4, 0.05, 0.08, 0.05])
btn_am = Button(btn_ax_am, 'Azimuth -')
btn_am.on_clicked(azim_minus)

btn_ax_ep = plt.axes([0.52, 0.05, 0.08, 0.05])
btn_ep = Button(btn_ax_ep, 'Elev +')
btn_ep.on_clicked(elev_plus)

btn_ax_em = plt.axes([0.62, 0.05, 0.08, 0.05])
btn_em = Button(btn_ax_em, 'Elev -')
btn_em.on_clicked(elev_minus)

plt.show()
