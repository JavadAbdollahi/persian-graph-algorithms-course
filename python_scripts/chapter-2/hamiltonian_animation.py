# -----------------------------------------------------------------------------
# File:         hamiltonian_animation.py
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

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
G = nx.cubical_graph()
start_node = 0
num_nodes = G.number_of_nodes()

# --- 2. Backtracking Algorithm with History Tracking ---
history = []

def solve_hamiltonian_cycle(graph, path):
    current_node = path[-1]
    history.append({'path': list(path), 'backtrack_edge': None})

    if len(path) == num_nodes:
        if graph.has_edge(current_node, path[0]):
            path.append(path[0])
            history.append({'path': list(path), 'backtrack_edge': None})
            return True
        else:
            return False

    for neighbor in sorted(list(graph.neighbors(current_node))):
        if neighbor not in path:
            path.append(neighbor)
            if solve_hamiltonian_cycle(graph, path):
                return True
            bad_edge = (path[-2], path[-1])
            path.pop()
            history.append({'path': list(path), 'backtrack_edge': bad_edge})
    return False

initial_path = [start_node]
solve_hamiltonian_cycle(G, initial_path)

# --- 3. Animation Setup ---
pos = nx.spring_layout(G, seed=42) 
fig, (ax_graph, ax_data) = plt.subplots(1, 2, figsize=(18, 9))
fig.suptitle("Finding a Hamiltonian Cycle via Backtracking", fontsize=20)
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Animation Update Function ---
def update(frame):
    state = history[frame]
    current_path = state['path']
    backtrack_edge = state['backtrack_edge']

    ax_graph.clear()
    ax_graph.set_title("Graph Traversal", fontsize=16)
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='lightgray', node_size=800)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='gray', width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=12)

    if current_path:
        path_edges = list(zip(current_path, current_path[1:]))
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=path_edges, edge_color='green', width=3.0)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, nodelist=current_path, node_color='lightgreen', node_size=800)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, nodelist=[current_path[-1]], node_color='orange', node_size=900)

    if backtrack_edge:
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=[backtrack_edge], edge_color='red', width=3.5, style='dashed')

    ax_data.clear()
    ax_data.set_title("Algorithm State", fontsize=16)
    ax_data.axis('off')
    ax_data.text(0.05, 0.9, "Stack / Current Path:", fontsize=14, weight='bold')
    path_text = " -> ".join(map(str, current_path))
    ax_data.text(0.5, 0.8, path_text, fontsize=12, ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen", alpha=0.8))
    
    action_text = ""
    if backtrack_edge:
        action_text = f"Dead End. Backtracking from {backtrack_edge[1]}."
    elif len(current_path) == num_nodes + 1:
        action_text = "Hamiltonian Cycle Found!"
    else:
        action_text = "Building path..."
    ax_data.text(0.5, 0.5, action_text, fontsize=14, ha='center', style='italic',
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3))

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 5. Create Folder for Frames and GIF ---
gif_name = 'hamiltonian_animation.gif'
folder_name = gif_name.replace('.gif', '')
os.makedirs(folder_name, exist_ok=True)

num_frames = len(history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=False)

# Save GIF
ani.save(os.path.join(folder_name, gif_name), writer='pillow', fps=2)

# Save frames
for i, frame in enumerate(history):
    update(i)
    frame_filename = f"frame_{i:03d}.png"
    fig.savefig(os.path.join(folder_name, frame_filename))

print(f"Animation and frames saved in folder '{folder_name}'.")
