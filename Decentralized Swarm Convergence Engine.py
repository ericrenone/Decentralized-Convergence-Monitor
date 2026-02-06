import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D

# --- Custom blue → green → red colormap ---
colors = [(0, 0, 1), (0, 1, 0), (1, 0, 0)]  # blue → green → red
cmap = LinearSegmentedColormap.from_list('BGR', colors, N=256)

# --- FER ENGINE WITH DRAMATIC SWARM ---
class FERMasterEngine:
    def __init__(self, n_nodes=1000):
        self.n = n_nodes
        self.states = np.random.uniform(-15, 15, (n_nodes, 2))
        self.history = {"AUPRC": [], "FISHER_TR": [], "ENTROPY": [], "KL_VEL": [], "F_BETA": []}

    def step(self, frame):
        mean = np.mean(self.states, axis=0)
        alpha = 2.0 / (1 + frame * 0.02)  # high early dynamics
        prev_states = self.states.copy()
        self.states += alpha * (mean - self.states) + 0.05 * np.random.randn(self.n, 2)

        # Metrics
        tr = np.sum(np.var(self.states, axis=0))
        vel_per_node = np.linalg.norm(self.states - prev_states, axis=1)
        vel = np.mean(vel_per_node)
        u, s, vh = np.linalg.svd(self.states - mean)
        p = s**2 / (np.sum(s**2) + 1e-9)
        entropy = -np.sum(p * np.log(p + 1e-9))
        auprc = 1.0 - 0.5 * np.exp(-frame * 0.05)
        f_beta = (1.25 * auprc * (1 - vel)) / (0.25 * auprc + (1 - vel))

        for k, v in zip(self.history.keys(), [auprc, tr, entropy, vel, f_beta]):
            self.history[k].append(v)

        # Z height = energy per node
        dist_from_mean = np.linalg.norm(self.states - mean, axis=1)
        z_metric = tr + vel_per_node + dist_from_mean

        # Color metric: normalize distance for colormap
        color_metric = (dist_from_mean - np.min(dist_from_mean)) / (np.max(dist_from_mean) - np.min(dist_from_mean) + 1e-9)

        # Combined energy for zooming
        energy = np.mean(z_metric)

        return self.states, z_metric, color_metric, mean, tr, vel, entropy, energy

# --- FIGURE & AXES ---
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d', facecolor='#050505')
engine = FERMasterEngine()

# --- ANIMATION ---
def animate(frame):
    ax.cla()
    nodes, z_metric, color_metric, mean, tr, vel, entropy, energy = engine.step(frame)

    # --- energy-based dynamic zoom ---
    zoom_factor = max(1.5, 20 * energy / np.max(z_metric))
    ax.set_xlim(mean[0]-zoom_factor, mean[0]+zoom_factor)
    ax.set_ylim(mean[1]-zoom_factor, mean[1]+zoom_factor)
    ax.set_zlim(0, max(5.0, np.max(z_metric)*1.05))

    # Labels and title
    ax.set_xlabel("X", color='white')
    ax.set_ylabel("Y", color='white')
    ax.set_zlabel("Z (Energy Height)", color='white')
    ax.set_title(f"DRONE SWARM → MONOLITH | STEP {frame}", color='white', fontsize=14, weight='bold')
    ax.set_facecolor('#050505')

    # Ghost attractor at center
    ax.scatter(mean[0], mean[1], 0.1, s=50, color='#00ffcc', alpha=0.05)

    # --- Scatter with high-contrast BGR colormap ---
    sc = ax.scatter(
        nodes[:,0], nodes[:,1], z_metric,
        c=color_metric, cmap=cmap, s=8, alpha=0.9, edgecolors='face'
    )

    # Optional surface mesh for texture effect
    if frame % 10 == 0:
        try:
            ax.plot_trisurf(nodes[:,0], nodes[:,1], z_metric,
                            cmap=cmap, alpha=0.15, linewidth=0)
        except:
            pass

    # Trailing lines to mean (sampled)
    for i in range(0, engine.n, engine.n//50):
        ax.plot([mean[0], nodes[i,0]], [mean[1], nodes[i,1]], [0, z_metric[i]],
                color='#00ffcc', alpha=0.03)

    # Dynamic metrics overlay
    ax.text2D(0.02, 0.95,
              f"TR={tr:.2f} | KL_VEL={vel:.2f} | ENTROPY={entropy:.2f} | ENERGY={energy:.2f}",
              color='white', transform=ax.transAxes, fontsize=10)

    return sc,

# --- Slow and extended animation for clarity ---
ani = FuncAnimation(fig, animate, frames=400, interval=80)
plt.tight_layout()
print("[*] DRONE SWARM → MONOLITH VISUALIZATION ONLINE. HIGH-CONTRAST BGR ACTIVE...")
plt.show()
