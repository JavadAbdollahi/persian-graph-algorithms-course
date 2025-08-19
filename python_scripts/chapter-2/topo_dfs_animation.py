# -----------------------------------------------------------------------------
# File:         topo_dfs_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         July 2025
# -----------------------------------------------------------------------------
import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

G = nx.DiGraph()
edges = [('CS1', 'DS'), ('DM', 'Algo'), ('DS', 'Algo'), ('DS', 'GT'), ('Algo', 'GT')]
nodes = ['CS1', 'DM', 'DS', 'Algo', 'GT']
G.add_nodes_from(nodes)
G.add_edges_from(edges)
all_nodes_in_order = sorted(list(G.nodes()))

# --- 2. DFS with History ---
history = []
visited = set()
topological_order = deque()

def dfs_visit_with_history(graph, node):
    visited.add(node)
    history.append({'processing': node, 'visited': set(visited), 'order': list(topological_order)})
    for neighbor in sorted(graph.neighbors(node)):
        if neighbor not in visited:
            dfs_visit_with_history(graph, neighbor)
    topological_order.appendleft(node)
    history.append({'processing': node, 'visited': set(visited), 'order': list(topological_order), 'finished': node})

for node in all_nodes_in_order:
    if node not in visited:
        dfs_visit_with_history(G, node)

history.append({'processing': None, 'visited': set(visited), 'order': list(topological_order), 'finished': None})

# --- 3. Animation Setup ---
pos = nx.circular_layout(G)
fig = plt.figure(figsize=(18, 9))
ax_graph = fig.add_subplot(1, 2, 1)
ax_data = fig.add_subplot(1, 2, 2)
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

def update(frame):
    state = history[frame]
    current_node = state.get('processing')
    visited_nodes = state.get('visited')
    sorted_list_state = state.get('order')
    finished_node = state.get('finished')

    ax_graph.clear()
    ax_graph.set_title("Graph Traversal (DFS)", fontsize=20)

    node_colors = []
    for node in G.nodes():
        if node == finished_node:
            node_colors.append('#2ca02c')
        elif node == current_node and not finished_node:
            node_colors.append('#ff7f0e')
        elif node in visited_nodes:
            node_colors.append('#add8e6')
        else:
            node_colors.append('#d3d3d3')
    
    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=3000, font_size=18, width=1.5, edge_color='gray',
            arrows=True, arrowstyle='->', arrowsize=20)

    ax_data.clear()
    ax_data.set_title("Algorithm State", fontsize=20)
    ax_data.axis('off')
    ax_data.text(0.5, 0.9, "Topological Order (L):", fontsize=16, weight='bold', ha='center')
    sorted_text = " -> ".join(sorted_list_state) if sorted_list_state else "[]"
    ax_data.text(0.5, 0.8, sorted_text, fontsize=16, ha='center', bbox=dict(boxstyle="round,pad=0.5", fc="#d3d3d3"))
    action_text = f"Node {finished_node} finished. Prepending to list." if finished_node else f"Visiting node {current_node}..." if current_node else ""
    ax_data.text(0.5, 0.5, action_text, fontsize=16, ha='center', style='italic')

    fig.suptitle(f'Step {frame}: DFS-based Topological Sort', fontsize=22, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 4. Save Frames and GIF ---
output_folder = 'topo_dfs_animation'
os.makedirs(output_folder, exist_ok=True)

num_frames = len(history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1500, repeat=False)

# Save individual frames
for i, frame in enumerate(range(num_frames)):
    update(frame)
    fig.savefig(os.path.join(output_folder, f"frame_{i:03d}.png"))

# Save GIF
ani.save(os.path.join(output_folder, 'animation.gif'), writer='pillow', fps=0.6)

print(f"Animation and frames saved in folder '{output_folder}'")
plt.show()
