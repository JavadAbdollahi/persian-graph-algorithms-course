# -----------------------------------------------------------------------------
# File:         eulerian_animation.py
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
Description:  This script animates Hierholzer's algorithm for finding an
              Eulerian tour. It displays the graph traversal on one side
              and the state of the stack and the tour list on the other.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

# Define the graph from the example
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), 
         ('B', 'E'), ('C', 'D'), ('C', 'E')]
# Use a MultiGraph to handle multiple edges if needed in other examples
G = nx.Graph()
G.add_edges_from(edges)
start_node = 'A'

# --- 2. Hierholzer's Algorithm with History Tracking ---
def hierholzer_with_history(graph, start):
    # Make a copy to modify
    g = graph.copy()
    
    stack = [start]
    tour = []
    history = []
    
    # Initial state
    history.append({'stack': list(stack), 'tour': list(tour), 'graph_edges': list(g.edges())})

    while stack:
        u = stack[-1] # Look at the top of the stack
        
        # Check if there is an unvisited edge from u
        if g.degree(u) > 0:
            # Find a neighbor and move to it
            v = list(g.neighbors(u))[0]
            stack.append(v)
            g.remove_edge(u, v) # Remove edge to mark as visited
        else:
            # Backtrack and add to tour
            tour.insert(0, stack.pop())

        # Record the state after this step
        history.append({'stack': list(stack), 'tour': list(tour), 'graph_edges': list(g.edges()), 'current_node': u})
        
    return history

# Generate the history of states
euler_history = hierholzer_with_history(G.copy(), start_node)

# --- 3. Animation Setup ---
pos = nx.circular_layout(G)
fig = plt.figure(figsize=(18, 9))

ax_graph = fig.add_subplot(1, 2, 1)
ax_data = fig.add_subplot(1, 2, 2)

fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Animation Update Function ---
def update(frame):
    state = euler_history[frame]
    stack_state = state['stack']
    tour_state = state['tour']
    remaining_edges = state['graph_edges']
    current_node = state.get('current_node')

    # --- Update Graph Subplot ---
    ax_graph.clear()
    ax_graph.set_title("Graph State", fontsize=20)
    
    # Draw nodes and labels
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='lightgray', node_size=2000)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=18, font_weight='bold')
    
    # Draw remaining edges in gray
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=remaining_edges, edge_color='gray', width=1.5, style='dashed')
    
    # Draw the tour found so far in green
    tour_edges = list(zip(tour_state, tour_state[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=tour_edges, edge_color='green', width=2.5)
    
    # Highlight the current node (top of stack)
    if stack_state:
        nx.draw_networkx_nodes(G, pos, nodelist=[stack_state[-1]], ax=ax_graph, node_color='red', node_size=2200)

    # --- Update Data Subplot ---
    ax_data.clear()
    ax_data.set_title("Algorithm State", fontsize=20)
    ax_data.axis('off')

    # Display Stack
    ax_data.text(0.1, 0.9, "Stack (S):", fontsize=16, weight='bold')
    stack_text = " ".join(reversed(stack_state)) if stack_state else "[]"
    ax_data.text(0.5, 0.75, f"Top -> {stack_text}", fontsize=14, ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))

    # Display Tour
    ax_data.text(0.1, 0.5, "Tour (T):", fontsize=16, weight='bold')
    tour_text = " -> ".join(tour_state) if tour_state else "[]"
    ax_data.text(0.5, 0.35, tour_text, fontsize=14, ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))
    
    # Set overall title for the frame
    if current_node:
        fig.suptitle(f'Step {frame}: Processing Node {current_node}', fontsize=22, y=0.98)
    else:
        fig.suptitle(f'Step {frame}: Initial State', fontsize=22, y=0.98)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# --- 5. Create and Save the Animation ---
num_frames = len(euler_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1200, repeat=False)

print("Displaying Eulerian Tour animation...")
plt.show()

try:
    print("Saving animation to 'eulerian_animation.gif'... This may take a moment.")
    ani.save('eulerian_animation.gif', writer='pillow', fps=1)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")