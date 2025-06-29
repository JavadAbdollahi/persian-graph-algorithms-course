# -----------------------------------------------------------------------------
# File:         bfs_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         June 2025
# -----------------------------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

"""
Description:  This script animates the Breadth-First Search (BFS) algorithm.
              It displays the graph traversal on one side and the state of the
              queue on the other, with a GitHub watermark.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

# --- 1. Graph Creation ---
G = nx.Graph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E')]
G.add_edges_from(edges)
start_node = 'A'

# --- 2. Custom BFS Implementation to Track Queue State ---
# We need to implement BFS manually to get the state of the queue at each step.
def bfs_traversal_with_history(graph, source):
    """
    Performs BFS and yields the state at each step.
    Yields: (current_node, visited_nodes, queue_state)
    """
    visited = {source}
    queue = deque([source])
    history = []
    
    # Initial state before loop starts
    history.append((None, set(visited), list(queue)))
    
    while queue:
        current_node = queue.popleft()
        # State after dequeuing
        history.append((current_node, set(visited), list(queue)))
        
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                # State after discovering a new node
                history.append((current_node, set(visited), list(queue)))
    
    return history

# Get the step-by-step history of the BFS traversal
bfs_history = bfs_traversal_with_history(G, start_node)

# --- 3. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig = plt.figure(figsize=(16, 8))

# Create two subplots: one for the graph, one for the queue
ax_graph = fig.add_subplot(1, 2, 1)
ax_queue = fig.add_subplot(1, 2, 2)

# Add GitHub watermark to the figure using the constant
fig.text(0.99, 0.01, GITHUB_URL,
         fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Animation Update Function ---
def update(frame):
    current_node, visited_nodes, queue_state = bfs_history[frame]

    # --- Update Graph Subplot ---
    ax_graph.clear()
    
    node_colors = []
    for node in G.nodes():
        if node == current_node:
            node_colors.append('#ff4747')  # Red for the currently processing node
        elif node in visited_nodes:
            node_colors.append('#add8e6')  # Light blue for visited nodes
        else:
            node_colors.append('#d3d3d3')  # Gray for unvisited nodes

    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=2500, font_size=25, width=2.0, edge_color='gray')
    
    ax_graph.set_title("Graph Traversal", fontsize=20)


    # --- Update Queue Subplot ---
    ax_queue.clear()
    ax_queue.set_title("Queue State", fontsize=20)
    ax_queue.set_xlim(0, 1)
    ax_queue.set_ylim(-0.5, 5) # Adjust based on max queue size
    
    if not queue_state:
        ax_queue.text(0.5, 2.5, "Queue is empty", ha='center', va='center', fontsize=18, color='gray')
    else:
        # Display queue elements vertically
        for i, node in enumerate(queue_state):
            y_pos = 4 - i * 0.7
            ax_queue.text(0.5, y_pos, str(node), ha='center', va='center',
                          fontsize=20, bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
        
        # Add Front and Rear labels
        ax_queue.text(0.1, 4, "Front", ha='center', va='center', fontsize=15, color='gray')
        ax_queue.text(0.9, 4 - (len(queue_state) - 1) * 0.7, "Rear", ha='center', va='center', fontsize=15, color='gray')

    # Hide axes
    ax_queue.axis('off')

    # Set the overall title for the frame
    if current_node:
        fig.suptitle(f'Step {frame}: Processing Node {current_node}', fontsize=22, y=0.98)
    else:
        fig.suptitle(f'Step {frame}: Initial State', fontsize=22, y=0.98)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# --- 5. Create and Save the Animation ---
num_frames = len(bfs_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1500, repeat=False)

# Show and save
print("Displaying animation...")
plt.show()

try:
    print("Saving animation to 'bfs_animation.gif'... This may take a moment.")
    ani.save('bfs_animation.gif', writer='pillow', fps=0.5)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")