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
import os

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
OUTPUT_FOLDER = "dfs_animation"

# Make folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. Graph Creation ---
G = nx.DiGraph()
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'), ('C', 'E'), ('D', 'E')]
G.add_edges_from(edges)
start_node = 'A'

# --- 3. DFS with History ---
def dfs_traversal_with_history(graph, source):
    visited = set()
    stack = [source]
    history = []

    while stack:
        history.append((None, set(visited), list(stack)))  # state before popping
        current_node = stack.pop()
        if current_node not in visited:
            visited.add(current_node)
            history.append((current_node, set(visited), list(stack)))  # after visiting
            neighbors = sorted(list(graph.neighbors(current_node)), reverse=True)
            for neighbor in neighbors:
                if neighbor not in visited:
                    stack.append(neighbor)
    history.append((None, set(visited), []))  # final empty state
    return history

dfs_history = dfs_traversal_with_history(G, start_node)

# --- 4. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig = plt.figure(figsize=(16, 8))
ax_graph = fig.add_subplot(1, 2, 1)
ax_stack = fig.add_subplot(1, 2, 2)
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 5. Animation Update Function ---
def update(frame):
    current_node, visited_nodes, stack_state = dfs_history[frame]

    # Graph subplot
    ax_graph.clear()
    node_colors = ['#add8e6' if n in visited_nodes else '#d3d3d3' for n in G.nodes()]
    if current_node:
        try:
            node_index = list(G.nodes()).index(current_node)
            node_colors[node_index] = '#ff4747'
        except ValueError:
            pass
    nx.draw(G, pos, ax=ax_graph, with_labels=True, node_color=node_colors,
            node_size=2500, font_size=25, width=1.5, edge_color='gray',
            arrows=True, arrowstyle='->', arrowsize=20)
    ax_graph.set_title("Graph Traversal", fontsize=20)

    # Stack subplot
    ax_stack.clear()
    ax_stack.set_title("Stack State", fontsize=20)
    ax_stack.set_xlim(0, 1)
    ax_stack.set_ylim(-0.5, 5)
    if not stack_state:
        ax_stack.text(0.5, 2.5, "Stack is empty", ha='center', va='center', fontsize=18, color='gray')
    else:
        for i, node in enumerate(reversed(stack_state)):
            y_pos = 4 - i * 0.7
            ax_stack.text(0.5, y_pos, str(node), ha='center', va='center',
                          fontsize=20, bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
        ax_stack.text(0.1, 4, "Top", ha='center', va='center', fontsize=15, color='gray')
    ax_stack.axis('off')

    # Title
    if current_node:
        fig.suptitle(f'Step {frame}: Visiting Node {current_node}', fontsize=22, y=0.98)
    else:
        if stack_state:
            fig.suptitle(f'Step {frame}: Popping {stack_state[-1]} from stack', fontsize=22, y=0.98)
        else:
            fig.suptitle(f'Step {frame}: Initial/Final State', fontsize=22, y=0.98)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save each frame as PNG
    frame_filename = os.path.join(OUTPUT_FOLDER, f"frame_{frame:03d}.png")
    fig.savefig(frame_filename)

# --- 6. Create Animation ---
num_frames = len(dfs_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1500, repeat=False)

# Show animation
print("Displaying DFS animation...")
plt.show()

# Save GIF inside folder
gif_path = os.path.join(OUTPUT_FOLDER, "animation.gif")
try:
    print(f"Saving GIF to '{gif_path}'... This may take a moment.")
    ani.save(gif_path, writer='pillow', fps=0.5)
    print("Animation saved successfully.")
except Exception as e:
    print(f"Error saving GIF: {e}")
