# -----------------------------------------------------------------------------
# File:         topo_dfs_animation.py
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

"""
Description:  This script animates the DFS-based algorithm for topological sorting.
              It displays the graph traversal and the construction of the 
              topologically sorted list by prepending nodes as they finish.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

# Create a directed graph from the handout example
G = nx.DiGraph()
edges = [('CS1', 'DS'), ('DM', 'Algo'), ('DS', 'Algo'), ('DS', 'GT'), ('Algo', 'GT')]
nodes = ['CS1', 'DM', 'DS', 'Algo', 'GT']
G.add_nodes_from(nodes)
G.add_edges_from(edges)
all_nodes_in_order = sorted(list(G.nodes()))

# --- 2. Custom DFS Implementation to Track History ---
history = []
visited = set()
topological_order = deque()

def dfs_visit_with_history(graph, node):
    """Recursive DFS visit that records history for animation."""
    visited.add(node)
    
    # State: Node is being visited (gray)
    history.append({'processing': node, 'visited': set(visited), 'order': list(topological_order)})

    for neighbor in sorted(graph.neighbors(node)):
        if neighbor not in visited:
            dfs_visit_with_history(graph, neighbor)
    
    # State: Node has finished, prepend it to the order
    topological_order.appendleft(node)
    history.append({'processing': node, 'visited': set(visited), 'order': list(topological_order), 'finished': node})

# Main loop to start DFS from all unvisited nodes
for node in all_nodes_in_order:
    if node not in visited:
        dfs_visit_with_history(G, node)

# Add a final state to show the result
history.append({'processing': None, 'visited': set(visited), 'order': list(topological_order), 'finished': None})

# --- 3. Animation Setup ---
pos = nx.circular_layout(G)
fig = plt.figure(figsize=(18, 9))

ax_graph = fig.add_subplot(1, 2, 1)
ax_data = fig.add_subplot(1, 2, 2)

fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Animation Update Function ---
def update(frame):
    state = history[frame]
    current_node = state.get('processing')
    visited_nodes = state.get('visited')
    sorted_list_state = state.get('order')
    finished_node = state.get('finished')

    # --- Update Graph Subplot ---
    ax_graph.clear()
    ax_graph.set_title("Graph Traversal (DFS)", fontsize=20)
    
    node_colors = []
    for node in G.nodes():
        if node == finished_node:
            node_colors.append('#2ca02c') # Green for finished nodes
        elif node == current_node and not finished_node:
            node_colors.append('#ff7f0e') # Orange for currently visiting node
        elif node in visited_nodes:
            node_colors.append('#add8e6')  # Light blue for visited nodes
        else:
            node_colors.append('#d3d3d3') # Gray for unvisited nodes
            
    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=3000, font_size=18, width=1.5, edge_color='gray',
            arrows=True, arrowstyle='->', arrowsize=20)

    # --- Update Data Subplot ---
    ax_data.clear()
    ax_data.set_title("Algorithm State", fontsize=20)
    ax_data.axis('off')

    # Display Sorted List
    ax_data.text(0.5, 0.9, "Topological Order (L):", fontsize=16, weight='bold', ha='center')
    sorted_text = " -> ".join(sorted_list_state) if sorted_list_state else "[]"
    ax_data.text(0.5, 0.8, sorted_text, fontsize=16, ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="#d3d3d3"))

    # Display current action
    action_text = ""
    if finished_node:
        action_text = f"Node {finished_node} finished. Prepending to list."
    elif current_node:
        action_text = f"Visiting node {current_node}..."
    
    ax_data.text(0.5, 0.5, action_text, fontsize=16, ha='center', style='italic')

    # Set overall title
    fig.suptitle(f'Step {frame}: DFS-based Topological Sort', fontsize=22, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# --- 5. Create and Save the Animation ---
num_frames = len(history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1500, repeat=False)

print("Displaying DFS-based topological sort animation...")
plt.show()

try:
    print("Saving animation to 'topo_dfs_animation.gif'... This may take a moment.")
    ani.save('topo_dfs_animation.gif', writer='pillow', fps=0.6)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")