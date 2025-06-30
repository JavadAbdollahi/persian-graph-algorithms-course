# -----------------------------------------------------------------------------
# File:         dfs_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         June 2025
# -----------------------------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
Description:  This script animates the Depth-First Search (DFS) algorithm.
              It displays the graph traversal on one side and the state of the
              explicit stack on the other, with a GitHub watermark.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"

# Create a directed graph from the handout example
G = nx.DiGraph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'E'), ('D', 'E')]
G.add_edges_from(edges)
start_node = 'A'

# --- 2. Custom DFS Implementation to Track Stack State ---
def dfs_traversal_with_history(graph, source):
    """
    Performs iterative DFS and yields the state at each step for animation.
    Yields: (current_node, visited_nodes, stack_state)
    """
    visited = set()
    stack = [source]
    history = []

    while stack:
        # State before popping from stack
        history.append((None, set(visited), list(stack)))
        
        current_node = stack.pop()

        if current_node not in visited:
            visited.add(current_node)
            # State after visiting a new node
            history.append((current_node, set(visited), list(stack)))

            # Add neighbors to the stack in reverse order to process them alphabetically
            # This makes the traversal deterministic and easier to follow.
            neighbors = sorted(list(graph.neighbors(current_node)), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
    
    # Final state with empty stack
    history.append((None, set(visited), []))
    return history

# Get the step-by-step history of the DFS traversal
dfs_history = dfs_traversal_with_history(G, start_node)

# --- 3. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig = plt.figure(figsize=(16, 8))

# Create two subplots: one for the graph, one for the stack
ax_graph = fig.add_subplot(1, 2, 1)
ax_stack = fig.add_subplot(1, 2, 2)

# Add GitHub watermark
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)


# --- 4. Animation Update Function ---
def update(frame):
    current_node, visited_nodes, stack_state = dfs_history[frame]

    # --- Update Graph Subplot ---
    ax_graph.clear()
    
    # Gray for unvisited, lightblue for visited, red for currently processing
    node_colors = ['#add8e6' if n in visited_nodes else '#d3d3d3' for n in G.nodes()]
    if current_node:
        try:
            node_index = list(G.nodes()).index(current_node)
            node_colors[node_index] = '#ff4747'
        except ValueError:
            pass # Node might not be in the list if graph is disconnected

    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=2500, font_size=25, width=1.5, edge_color='gray',
            arrows=True, arrowstyle='->', arrowsize=20)
    
    ax_graph.set_title("Graph Traversal", fontsize=20)

    # --- Update Stack Subplot ---
    ax_stack.clear()
    ax_stack.set_title("Stack State", fontsize=20)
    ax_stack.set_xlim(0, 1)
    ax_stack.set_ylim(-0.5, 5) # Adjust based on max stack size

    if not stack_state:
        ax_stack.text(0.5, 2.5, "Stack is empty", ha='center', va='center', fontsize=18, color='gray')
    else:
        # Display stack elements vertically
        for i, node in enumerate(reversed(stack_state)): # reversed to show top at top
            y_pos = 4 - i * 0.7
            ax_stack.text(0.5, y_pos, str(node), ha='center', va='center',
                          fontsize=20, bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
        
        # Add Top label
        ax_stack.text(0.1, 4, "Top", ha='center', va='center', fontsize=15, color='gray')

    ax_stack.axis('off')

    # Set overall title for the frame
    if current_node:
        fig.suptitle(f'Step {frame}: Visiting Node {current_node}', fontsize=22, y=0.98)
    else:
        # This state happens when a node is about to be popped from the stack
        if stack_state:
            fig.suptitle(f'Step {frame}: Popping {stack_state[-1]} from stack', fontsize=22, y=0.98)
        else:
            fig.suptitle(f'Step {frame}: Initial/Final State', fontsize=22, y=0.98)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# --- 5. Create and Save the Animation ---
num_frames = len(dfs_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1500, repeat=False)

# Show and save
print("Displaying DFS animation...")
plt.show()

try:
    print("Saving DFS animation to 'dfs_animation.gif'... This may take a moment.")
    ani.save('dfs_animation.gif', writer='pillow', fps=0.5)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")