# -----------------------------------------------------------------------------
# File:         hamiltonian_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         July 2025
# -----------------------------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Description:  This script animates the backtracking algorithm to find a 
              Hamiltonian cycle, with a dual-pane view for the graph
              and the algorithm's state (stack/path).
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

# A smaller graph to make the backtracking process clearer
G = nx.cubical_graph() # A cube graph
start_node = 0
num_nodes = G.number_of_nodes()

# --- 2. Backtracking Algorithm with Enhanced History Tracking ---
history = []

def solve_hamiltonian_cycle(graph, path):
    # The current node is the last one in the path
    current_node = path[-1]

    # Add current state to history for visualization
    history.append({'path': list(path), 'backtrack_edge': None})

    # Base case: if a full path is found
    if len(path) == num_nodes:
        # Check if it's a cycle
        if graph.has_edge(current_node, path[0]):
            path.append(path[0]) # Add the first node to complete the cycle
            history.append({'path': list(path), 'backtrack_edge': None})
            return True # Solution found
        else:
            return False # It's a path, but not a cycle

    # Recursive step
    for neighbor in sorted(list(graph.neighbors(current_node))):
        if neighbor not in path:
            path.append(neighbor)
            if solve_hamiltonian_cycle(graph, path):
                return True # Propagate success signal
            
            # If the recursive call returned False, we must backtrack
            bad_edge = (path[-2], path[-1])
            path.pop()
            history.append({'path': list(path), 'backtrack_edge': bad_edge})
            
    return False

# Initial call to the recursive function
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

    # --- Update Graph Subplot ---
    ax_graph.clear()
    ax_graph.set_title("Graph Traversal", fontsize=16)
    
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='lightgray', node_size=800)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='gray', width=1.0, alpha=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=12)

    if current_path: # Check if path is not empty
        path_edges = list(zip(current_path, current_path[1:]))
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=path_edges, edge_color='green', width=3.0)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, nodelist=current_path, node_color='lightgreen', node_size=800)
        nx.draw_networkx_nodes(G, pos, ax=ax_graph, nodelist=[current_path[-1]], node_color='orange', node_size=900)

    if backtrack_edge:
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=[backtrack_edge], edge_color='red', width=3.5, style='dashed')
        
    # --- Update Data Subplot ---
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

# --- 5. Create and Save the Animation ---
num_frames = len(history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=False)

print("Displaying Hamiltonian Cycle animation...")
plt.show()

try:
    print("Saving animation to 'hamiltonian_animation.gif'... This may take a moment.")
    ani.save('hamiltonian_animation.gif', writer='pillow', fps=2)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")