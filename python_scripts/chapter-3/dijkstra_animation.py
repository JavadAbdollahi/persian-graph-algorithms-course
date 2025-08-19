# -----------------------------------------------------------------------------
# File:         dijkstra_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         July 2025
# -----------------------------------------------------------------------------

import heapq
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os # Import the os module to handle file paths and directories

"""
Description:  This script animates Dijkstra's shortest path algorithm.
              It creates a directory, saves each frame as a PNG image,
              and also saves the final animation as a GIF inside that directory.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
OUTPUT_DIR = "dijkstra_animation_output" # Directory to save the output files

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Graph Definition
G = nx.Graph()
edges_with_weights = [
    ('A', 'B', 1), ('A', 'C', 5), ('B', 'C', 2), ('B', 'D', 2),
    ('B', 'E', 4), ('C', 'D', 1), ('D', 'E', 3)
]
G.add_weighted_edges_from(edges_with_weights)
start_node = 'A'

# --- 2. Dijkstra's Algorithm with History Tracking ---
def dijkstra_with_history(graph, source):
    """Performs Dijkstra's algorithm and returns a history of states for animation."""
    # (The function body is the same as the previous version)
    history = []
    distances = {node: float('infinity') for node in graph.nodes()}
    distances[source] = 0
    pq = [(0, source)]
    visited = set()
    history.append({'current_node': None, 'distances': distances.copy(), 'visited': visited.copy(), 'pq': list(pq), 'message': f'Start Node: {source}'})
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        if current_node in visited: continue
        visited.add(current_node)
        history.append({'current_node': current_node, 'distances': distances.copy(), 'visited': visited.copy(), 'pq': list(pq), 'message': f'Visiting Node {current_node}'})
        for neighbor in graph.neighbors(current_node):
            if neighbor in visited: continue
            weight = graph[current_node][neighbor]['weight']
            new_distance = current_distance + weight
            if new_distance < distances[neighbor]:
                old_dist = distances[neighbor]
                distances[neighbor] = new_distance
                heapq.heappush(pq, (new_distance, neighbor))
                pq.sort()
                old_dist_str = 'inf' if old_dist == float('infinity') else f'{old_dist:.1f}'
                history.append({'current_node': current_node, 'distances': distances.copy(), 'visited': visited.copy(), 'pq': list(pq), 'message': f"Relaxing ({current_node}-{neighbor})"})
    history.append({'current_node': None, 'distances': distances.copy(), 'visited': visited.copy(), 'pq': [], 'message': 'Algorithm Finished'})
    return history

# Get the step-by-step history
dijkstra_history = dijkstra_with_history(G, start_node)

# --- 3. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig = plt.figure(figsize=(18, 9))
ax_graph = fig.add_subplot(1, 2, 1)
ax_data = fig.add_subplot(1, 2, 2)
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)


# --- 4. Animation Update Function ---
def update(frame):
    """Function to update and save each animation frame."""
    # (The update logic is the same, with one added line at the end)
    state = dijkstra_history[frame]
    current_node, distances, visited, pq, message = state['current_node'], state['distances'], state['visited'], state['pq'], state['message']
    ax_graph.clear()
    node_colors = ['#ff6347' if node == current_node else '#add8e6' if node in visited else '#d3d3d3' for node in G.nodes()]
    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors, node_size=2000, font_size=20, width=1.5, edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_graph, font_size=14)
    ax_graph.set_title("Graph Traversal", fontsize=18)
    ax_data.clear()
    ax_data.axis('off')
    ax_data.set_title("Data Structures State", fontsize=18)
    ax_data.text(0.1, 0.95, "Priority Queue (dist, node)", fontsize=14, weight='bold')
    if not pq: ax_data.text(0.1, 0.88, "Empty", color='gray', fontsize=12)
    else:
        for i, (dist, node) in enumerate(pq[:8]): ax_data.text(0.1, 0.88 - i * 0.06, f"({dist:.1f}, '{node}')", fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="#90ee90"))
        if len(pq) > 8: ax_data.text(0.1, 0.88 - 8 * 0.06, "...", fontsize=12)
    ax_data.text(0.05, 0.88, "Min", ha='center', va='center', fontsize=11, color='gray')
    ax_data.text(0.6, 0.95, "Distances", fontsize=14, weight='bold')
    sorted_nodes = sorted(distances.keys())
    for i, node in enumerate(sorted_nodes):
        dist_str = f"{distances[node]:.1f}" if distances[node] != float('infinity') else 'inf'
        ax_data.text(0.6, 0.88 - i * 0.06, f"'{node}': {dist_str}", fontsize=12)
    fig.suptitle(f'Step {frame}: {message}', fontsize=20, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- NEW: Save each frame as a separate image file ---
    frame_filename = os.path.join(OUTPUT_DIR, f'frame_{frame:03d}.png')
    plt.savefig(frame_filename)


# --- 5. Create and Save the Animation ---
num_frames = len(dijkstra_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1200, repeat=False)

try:
    # Save the final animation as a GIF inside the output directory
    gif_filename = os.path.join(OUTPUT_DIR, 'dijkstra_animation.gif')
    print(f"Saving animation to '{gif_filename}'...")
    ani.save(gif_filename, writer='pillow', fps=0.66)
    print("Animation and frames saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")