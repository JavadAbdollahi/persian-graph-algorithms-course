# -----------------------------------------------------------------------------
# File:         prim_animation.py
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
import os

"""
Description:  This script animates Prim's algorithm for finding a Minimum Spanning Tree.
              It shows the MST growing and displays the state of the priority queue.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
OUTPUT_DIR = "prim_animation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Graph Definition (from the handout example)
G = nx.Graph() # Prim's algorithm works on undirected graphs
edges_with_weights = [
    ('A', 'B', 4), ('A', 'C', 2), ('B', 'C', 5), ('B', 'D', 10),
    ('C', 'E', 3), ('D', 'E', 6)
]
G.add_weighted_edges_from(edges_with_weights)
start_node = 'A'

# --- 2. Prim's Algorithm with History Tracking ---
def prim_with_history(graph, source):
    """Performs Prim's algorithm and returns a history of states for animation."""
    history = []
    keys = {node: float('infinity') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    keys[source] = 0
    
    # Priority queue: (key, node, predecessor)
    pq = [(0, source, None)] 
    
    nodes_in_mst = set()
    mst_edges = []
    total_weight = 0

    # Initial state
    history.append({
        'current_node': None,
        'nodes_in_mst': nodes_in_mst.copy(),
        'mst_edges': list(mst_edges),
        'total_weight': total_weight,
        'pq': list(pq),
        'message': 'Initialization'
    })

    while pq:
        key, u, pred = heapq.heappop(pq)
        
        if u in nodes_in_mst:
            continue
            
        nodes_in_mst.add(u)
        if pred is not None:
            mst_edges.append((pred, u))
            total_weight += key

        # State after extracting from PQ
        history.append({
            'current_node': u,
            'nodes_in_mst': nodes_in_mst.copy(),
            'mst_edges': list(mst_edges),
            'total_weight': total_weight,
            'pq': list(pq),
            'message': f'Adding node {u} to MST'
        })

        for v in graph.neighbors(u):
            if v not in nodes_in_mst:
                weight = graph[u][v]['weight']
                if weight < keys[v]:
                    keys[v] = weight
                    predecessors[v] = u
                    heapq.heappush(pq, (weight, v, u))
                    # Sort for consistent display
                    pq.sort()
                    # State after a key update
                    history.append({
                        'current_node': u,
                        'nodes_in_mst': nodes_in_mst.copy(),
                        'mst_edges': list(mst_edges),
                        'total_weight': total_weight,
                        'pq': list(pq),
                        'message': f'Updating key for node {v}'
                    })
    
    # Final state
    history.append({
        'current_node': None,
        'nodes_in_mst': nodes_in_mst.copy(),
        'mst_edges': list(mst_edges),
        'total_weight': total_weight,
        'pq': [],
        'message': 'Algorithm Finished'
    })
    return history

prim_history = prim_with_history(G, start_node)

# --- 3. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig, (ax_graph, ax_data) = plt.subplots(1, 2, figsize=(18, 9))
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)
NODE_SIZE = 2000

# --- 4. Animation Update Function ---
def update(frame):
    ax_graph.clear()
    state = prim_history[frame]
    current_node, nodes_in_mst, mst_edges, total_weight, pq, message = state.values()

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        if node == current_node:
            node_colors.append('#ff6347') # Red for current
        elif node in nodes_in_mst:
            node_colors.append('#add8e6') # Blue for in MST
        else:
            node_colors.append('#d3d3d3') # Gray for not in MST
            
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=NODE_SIZE, node_color=node_colors)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=16, font_color='black')

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='lightgray', width=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=mst_edges, edge_color='blue', width=2.5)
    
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax_graph, font_size=14)
    ax_graph.set_title("Prim's Algorithm", fontsize=18)
    
    # --- Update Data Subplot ---
    ax_data.clear()
    ax_data.axis('off')
    ax_data.set_title("Data Structures State", fontsize=18)
    
    ax_data.text(0.1, 0.95, f"MST Weight: {total_weight}", fontsize=14, weight='bold')
    
    # Display Priority Queue
    ax_data.text(0.1, 0.85, "Priority Queue (key, node)", fontsize=14, weight='bold')
    if not pq:
        ax_data.text(0.1, 0.78, "Empty", color='gray', fontsize=12)
    else:
        for i, (key, node, pred) in enumerate(pq[:10]): # Show top 10 elements
            ax_data.text(0.1, 0.78 - i * 0.07, f"({key}, '{node}')", fontsize=12, 
                         bbox=dict(boxstyle="round,pad=0.3", fc="#90ee90"))
    ax_data.text(0.05, 0.78, "Min", ha='center', va='center', fontsize=11, color='gray')
    
    fig.suptitle(f'Step {frame}: {message}', fontsize=20, y=0.98)
    
    frame_filename = os.path.join(OUTPUT_DIR, f'frame_{frame:03d}.png')
    plt.savefig(frame_filename)

# --- 5. Create and Save the Animation ---
ani = animation.FuncAnimation(fig, update, frames=len(prim_history), interval=1000, repeat=False)
try:
    gif_filename = os.path.join(OUTPUT_DIR, 'prim_animation.gif')
    print(f"Saving animation to '{gif_filename}'...")
    ani.save(gif_filename, writer='pillow', fps=1)
    print("Animation and frames saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")