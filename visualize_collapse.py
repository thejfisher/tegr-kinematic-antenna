import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import RadioButtons
import time
import os

# Set global dark theme
plt.style.use('dark_background')

plt.ion()
fig = plt.figure(figsize=(12, 9))
# Leave room on the left for the radio buttons
fig.subplots_adjust(left=0.25)
ax = fig.add_subplot(111, projection='3d')

# Make the 3D pane backgrounds completely black
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
# Dim the grid lines so particles stand out
ax.xaxis._axinfo["grid"].update({"color": (0.3, 0.3, 0.3, 1)})
ax.yaxis._axinfo["grid"].update({"color": (0.3, 0.3, 0.3, 1)})
ax.zaxis._axinfo["grid"].update({"color": (0.3, 0.3, 0.3, 1)})

# Interactive Radio Buttons for Metric Selection
rax = plt.axes([0.02, 0.6, 0.18, 0.35], facecolor='black')
radio = RadioButtons(rax, ('Gamma', 'Hue (Phase)', 'Mass', 'Momentum', 'Time', 'Z-Position', 'Gravity'))
for label in radio.labels:
    label.set_color('white')

current_metric = 'Gamma'
zoom_level = 25.0
force_redraw = False

def update_metric(label):
    global current_metric, force_redraw
    current_metric = label
    force_redraw = True

radio.on_clicked(update_metric)

def on_scroll(event):
    global zoom_level
    if event.button == 'up':
        zoom_level *= 0.8   # Scroll up to zoom in
    elif event.button == 'down':
        zoom_level *= 1.25  # Scroll down to zoom out
    
    # Immediately apply limits to feel responsive
    ax.set_xlim([-zoom_level, zoom_level])
    ax.set_ylim([-zoom_level, zoom_level])
    ax.set_zlim([-zoom_level, zoom_level])
    fig.canvas.draw_idle()

fig.canvas.mpl_connect('scroll_event', on_scroll)

print("Starting live visualization of checkpoint_latest.npz...")
print("Leave this window open. Use the radio buttons on the left to change color modes.")
print("Scroll your mouse wheel to zoom in and out!")

last_tick = -1
cbar = None

while True:
    if os.path.exists("checkpoint_latest.npz"):
        try:
            data = np.load("checkpoint_latest.npz")
            chunk = data['chunk']
            tick = data['tick']
            
            if (tick != last_tick or force_redraw) and len(chunk) > 0:
                force_redraw = False
                last_frame = chunk[-1]
                x = last_frame[:, 1]
                y = last_frame[:, 2]
                z = last_frame[:, 3]
                
                # Extract metrics
                r_sq = x**2 + y**2 + z**2
                gravity = 50000.0 / np.maximum(r_sq, 1e-6)
                
                metrics = {
                    'Gamma': (last_frame[:, 9], 'plasma'),
                    'Hue (Phase)': (last_frame[:, 8], 'hsv'),
                    'Mass': (last_frame[:, 7], 'cool'),
                    'Momentum': (np.linalg.norm(last_frame[:, 4:7], axis=1), 'magma'),
                    'Time': (last_frame[:, 0], 'viridis'),
                    'Z-Position': (z, 'inferno'),
                    'Gravity': (gravity, 'cividis')
                }
                
                c_data, cmap = metrics[current_metric]
                
                ax.clear()
                # Re-apply dark panes because ax.clear() resets them
                ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
                ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
                ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
                
                # Plot particles super bright (alpha=1.0, edgecolors='none', larger size)
                sc = ax.scatter(x, y, z, c=c_data, cmap=cmap, s=10, alpha=1.0, edgecolors='none')
                
                # Manage colorbar safely
                if cbar is None:
                    # Create a dedicated axis for the colorbar so it doesn't shrink the 3D plot
                    cax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
                    cbar = fig.colorbar(sc, cax=cax)
                else:
                    cbar.update_normal(sc)
                cbar.set_label(f'{current_metric} Value', color='white')
                cbar.ax.yaxis.set_tick_params(color='white')
                plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
                
                ax.set_title(f"Teleparallel Collapse (N={len(x)})\nTick: {tick} | Displaying: {current_metric}", color='white')
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")
                ax.set_xlim([-zoom_level, zoom_level])
                ax.set_ylim([-zoom_level, zoom_level])
                ax.set_zlim([-zoom_level, zoom_level])
                
                plt.draw()
                plt.pause(0.5)
                last_tick = tick
            else:
                plt.pause(1.0)
                
        except Exception as e:
            plt.pause(0.5)
    else:
        plt.pause(2.0)
