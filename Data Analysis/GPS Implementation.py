import matplotlib   # switch backend before importing pyplot
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def gps_harmonic(p):
    """Compute the harmonic-mean GPS for a list of positive values."""
    p = np.array(p, dtype=float)
    return len(p) / np.sum(1.0 / p)

# Initial slider values
init_p = [10, 20, 30, 40]

# Build the figure + axes
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(left=0.1, bottom=0.30)

# Draw bars for p1…p4
bars = ax.bar(
    range(4),
    init_p,
    tick_label=[f"p{i+1}" for i in range(4)],
    color="skyblue"
)

# Draw the initial GPS line
initial_g = gps_harmonic(init_p)
gps_line = ax.axhline(
    initial_g,
    color="red",
    linestyle="--",
    linewidth=2
)

# Set labels and title
ax.set_ylim(0, max(init_p) * 1.5)
ax.set_ylabel("Value")
ax.set_title("Interactive p₁–p₄ with GPS (harmonic mean)")

# Create four sliders, one per pᵢ
sliders = []
for i in range(4):
    ax_slider = plt.axes([0.1, 0.25 - 0.04 * i, 0.8, 0.03])
    slider = Slider(
        ax_slider,
        label=f"p{i+1}",
        valmin=1,
        valmax=100,
        valinit=init_p[i],
        valstep=1,
        valfmt="%0.0f"
    )
    sliders.append(slider)

# Update callback
def update(val):
    # 1) Read all four slider values
    p_vals = [s.val for s in sliders]

    # 2) Update bar heights
    for bar, newh in zip(bars, p_vals):
        bar.set_height(newh)

    # 3) Recompute GPS and update the dashed line
    g = gps_harmonic(p_vals)
    xs = gps_line.get_xdata()             # original x-coords ([xmin, xmax])
    gps_line.set_ydata([g] * len(xs))     # new y-coords must match length of xs

    # 4) Optionally adjust the y-axis to fit everything
    ax.set_ylim(0, max(p_vals + [g]) * 1.5)

    # 5) Trigger a redraw
    fig.canvas.draw_idle()
    plt.pause(0.001)

# Hook up the sliders to our callback
for s in sliders:
    s.on_changed(update)

# Show the interactive window
plt.show()
