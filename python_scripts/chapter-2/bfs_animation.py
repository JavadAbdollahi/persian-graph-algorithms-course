# ----------------------------------------------------------------------------- 
# File: bfs_animation.py # Project: Graph Algorithms Course Notes 
# Author: Mohammad Javad Abdolahi # GitHub: https://github.com/JavadAbdollahi 
# Supervisor: Dr. Behnaz Omoomi 
# Date: June 2025 
# -----------------------------------------------------------------------------

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
FOLDER_NAME = "bfs_animation"

# --- 2. Graph Creation ---
G = nx.Graph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')]
G.add_edges_from(edges)
start_node = 'A'

# --- 3. BFS with History ---
def bfs_traversal_with_history(graph, source):
    visited = {source}
    queue = deque([source])
    history = []
    history.append((None, set(visited), list(queue)))
    
    while queue:
        current_node = queue.popleft()
        history.append((current_node, set(visited), list(queue)))
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                history.append((current_node, set(visited), list(queue)))
    return history

bfs_history = bfs_traversal_with_history(G, start_node)

# --- 4. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig = plt.figure(figsize=(16, 8))
ax_graph = fig.add_subplot(1, 2, 1)
ax_queue = fig.add_subplot(1, 2, 2)
fig.text(0.99, 0.01, GITHUB_URL,
         fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

def update(frame):
    current_node, visited_nodes, queue_state = bfs_history[frame]
    ax_graph.clear()
    ax_queue.clear()

    # Graph coloring
    node_colors = []
    for node in G.nodes():
        if node == current_node:
            node_colors.append('#ff4747')
        elif node in visited_nodes:
            node_colors.append('#add8e6')
        else:
            node_colors.append('#d3d3d3')
    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=2500, font_size=25, width=2.0, edge_color='gray')
    ax_graph.set_title("Graph Traversal", fontsize=20)

    # Queue subplot
    ax_queue.set_title("Queue State", fontsize=20)
    ax_queue.set_xlim(0, 1)
    ax_queue.set_ylim(-0.5, 5)
    if not queue_state:
        ax_queue.text(0.5, 2.5, "Queue is empty", ha='center', va='center', fontsize=18, color='gray')
    else:
        for i, node in enumerate(queue_state):
            y_pos = 4 - i * 0.7
            ax_queue.text(0.5, y_pos, str(node), ha='center', va='center',
                          fontsize=20, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
        ax_queue.text(0.1, 4, "Front", ha='center', va='center', fontsize=15, color='gray')
        ax_queue.text(0.9, 4 - (len(queue_state) - 1) * 0.7, "Rear", ha='center', va='center', fontsize=15, color='gray')

    ax_queue.axis('off')
    if current_node:
        fig.suptitle(f'Step {frame}: Processing Node {current_node}', fontsize=22, y=0.98)
    else:
        fig.suptitle(f'Step {frame}: Initial State', fontsize=22, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 5. Create Folder ---
os.makedirs(FOLDER_NAME, exist_ok=True)

# --- 6. Save Frames Individually ---
num_frames = len(bfs_history)
for i in range(num_frames):
    update(i)
    frame_filename = os.path.join(FOLDER_NAME, f"frame_{i:03d}.png")
    fig.savefig(frame_filename)
print(f"Saved {num_frames} frames in '{FOLDER_NAME}' folder.")

# --- 7. Save GIF ---
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1500, repeat=False)
gif_path = os.path.join(FOLDER_NAME, "animation.gif")
ani.save(gif_path, writer='pillow', fps=0.5)
print(f"Saved GIF as '{gif_path}'")
