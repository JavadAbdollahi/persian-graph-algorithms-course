# -----------------------------------------------------------------------------
# File:         kahn_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         July 2025
# -----------------------------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import os

"""
Description:  This script animates Kahn's algorithm for topological sorting.
              It displays the graph traversal, the state of the queue (nodes
              with in-degree 0), and the final sorted list.
"""
# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

# Create a directed graph
G = nx.DiGraph()
edges = [('CS1', 'DS'), ('DM', 'Algo'), ('DS', 'Algo'), ('DS', 'GT'), ('Algo', 'GT')]
nodes = ['CS1', 'DM', 'DS', 'Algo', 'GT']
G.add_nodes_from(nodes)
G.add_edges_from(edges)

# --- 2. Kahn's Algorithm with History ---
def kahn_traversal_with_history(graph):
    in_degrees = {u: d for u, d in graph.in_degree()}
    queue = deque([u for u in graph.nodes() if in_degrees[u] == 0])
    sorted_list = []
    history = []

    # Initial state
    history.append((None, list(queue), list(sorted_list), dict(in_degrees)))

    while queue:
        u = queue.popleft()
        sorted_list.append(u)
        history.append((u, list(queue), list(sorted_list), dict(in_degrees)))

        for v in sorted(graph.neighbors(u)):
            in_degrees[v] -= 1
            if in_degrees[v] == 0:
                queue.append(v)
            history.append((u, list(queue), list(sorted_list), dict(in_degrees)))

    # Final state
    history.append((None, list(queue), list(sorted_list), dict(in_degrees)))
    return history

kahn_history = kahn_traversal_with_history(G)

# --- 3. Animation Setup ---
pos = nx.circular_layout(G)
fig = plt.figure(figsize=(18, 9))
ax_graph = fig.add_subplot(1, 2, 1)
ax_data = fig.add_subplot(1, 2, 2)
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Update Function ---
def update(frame):
    current_node, queue_state, sorted_list_state, in_degrees_state = kahn_history[frame]

    # Graph subplot
    ax_graph.clear()
    ax_graph.set_title("Graph State", fontsize=20)
    node_colors = []
    for node in G.nodes():
        if node in sorted_list_state:
            node_colors.append('#d3d3d3')
        elif node in queue_state:
            node_colors.append('#add8e6')
        else:
            node_colors.append('lightgreen')
    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=3000, font_size=18, width=1.5, edge_color='gray',
            arrows=True, arrowstyle='->', arrowsize=20)

    # Data subplot
    ax_data.clear()
    ax_data.set_title("Algorithm State", fontsize=20)
    ax_data.axis('off')
    ax_data.text(0.1, 0.9, "Sorted List (L):", fontsize=16, weight='bold')
    ax_data.text(0.1, 0.8, ", ".join(sorted_list_state) if sorted_list_state else "[]",
                 fontsize=16, bbox=dict(boxstyle="round,pad=0.5", fc="#d3d3d3"))
    ax_data.text(0.1, 0.65, "Queue (S):", fontsize=16, weight='bold')
    ax_data.text(0.1, 0.55, ", ".join(queue_state) if queue_state else "[]",
                 fontsize=16, bbox=dict(boxstyle="round,pad=0.5", fc="#add8e6"))
    ax_data.text(0.1, 0.4, "In-Degrees:", fontsize=16, weight='bold')
    degree_text = "\n".join([f"{node}: {deg}" for node, deg in in_degrees_state.items()])
    ax_data.text(0.1, 0.05, degree_text, fontsize=14, va='bottom', fontfamily='monospace')

    if current_node:
        fig.suptitle(f'Step {frame}: Processing Node {current_node}', fontsize=22, y=0.98)
    else:
        fig.suptitle(f'Step {frame}: Initial/Final State', fontsize=22, y=0.98)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 5. Save frames folder ---
frames_dir = "kahn_animation"
os.makedirs(frames_dir, exist_ok=True)

# Save each frame as PNG
for i in range(len(kahn_history)):
    update(i)
    frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
    plt.savefig(frame_path)

num_frames = len(kahn_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1200, repeat=False)

try:
    gif_path = os.path.join(frames_dir, "animation.gif")
    print(f"Saving animation to '{gif_path}'... This may take a moment.")
    ani.save(gif_path, writer='pillow', fps=0.8)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")

# --- 6. Create Animation ---
num_frames = len(kahn_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1200, repeat=False)

print("Displaying Kahn's algorithm animation...")
plt.show()

# --- 7. Save as GIF ---
try:
    print("Saving animation to 'kahn_animation.gif'...")
    ani.save('kahn_animation.gif', writer='pillow', fps=0.8)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")