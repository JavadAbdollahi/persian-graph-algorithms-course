# -----------------------------------------------------------------------------
# File:         eulerian_animation.py
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

# Define the graph
edges = [('A', 'B'), ('A', 'C'), ('B', 'C'), ('B', 'D'), 
         ('B', 'E'), ('C', 'D'), ('C', 'E')]
G = nx.Graph()
G.add_edges_from(edges)
start_node = 'A'

# --- 2. Hierholzer's Algorithm with History Tracking ---
def hierholzer_with_history(graph, start):
    g = graph.copy()
    stack = [start]
    tour = []
    history = []
    history.append({'stack': list(stack), 'tour': list(tour), 'graph_edges': list(g.edges())})

    while stack:
        u = stack[-1]
        if g.degree(u) > 0:
            v = list(g.neighbors(u))[0]
            stack.append(v)
            g.remove_edge(u, v)
        else:
            tour.insert(0, stack.pop())

        history.append({'stack': list(stack), 'tour': list(tour), 'graph_edges': list(g.edges()), 'current_node': u})
        
    return history

euler_history = hierholzer_with_history(G.copy(), start_node)

# --- 3. Animation Setup ---
pos = nx.circular_layout(G)
fig = plt.figure(figsize=(18, 9))
ax_graph = fig.add_subplot(1, 2, 1)
ax_data = fig.add_subplot(1, 2, 2)
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Update Function ---
def update(frame):
    state = euler_history[frame]
    stack_state = state['stack']
    tour_state = state['tour']
    remaining_edges = state['graph_edges']
    current_node = state.get('current_node')

    ax_graph.clear()
    ax_graph.set_title("Graph State", fontsize=20)
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_color='lightgray', node_size=2000)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=18, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=remaining_edges, edge_color='gray', width=1.5, style='dashed')
    tour_edges = list(zip(tour_state, tour_state[1:]))
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=tour_edges, edge_color='green', width=2.5)
    if stack_state:
        nx.draw_networkx_nodes(G, pos, nodelist=[stack_state[-1]], ax=ax_graph, node_color='red', node_size=2200)

    ax_data.clear()
    ax_data.set_title("Algorithm State", fontsize=20)
    ax_data.axis('off')
    ax_data.text(0.1, 0.9, "Stack (S):", fontsize=16, weight='bold')
    stack_text = " ".join(reversed(stack_state)) if stack_state else "[]"
    ax_data.text(0.5, 0.75, f"Top -> {stack_text}", fontsize=14, ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightgreen"))
    ax_data.text(0.1, 0.5, "Tour (T):", fontsize=16, weight='bold')
    tour_text = " -> ".join(tour_state) if tour_state else "[]"
    ax_data.text(0.5, 0.35, tour_text, fontsize=14, ha='center',
                 bbox=dict(boxstyle="round,pad=0.5", fc="lightblue"))

    if current_node:
        fig.suptitle(f'Step {frame}: Processing Node {current_node}', fontsize=22, y=0.98)
    else:
        fig.suptitle(f'Step {frame}: Initial State', fontsize=22, y=0.98)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save each frame
    frame_filename = os.path.join(output_folder, f'frame_{frame:03d}.png')
    fig.savefig(frame_filename)

# --- 5. Prepare Output Folder ---
gif_filename = 'animation.gif'
output_folder = os.path.splitext(gif_filename)[0]  # folder same as gif name without extension
os.makedirs(output_folder, exist_ok=True)

# --- 6. Create and Save Animation ---
num_frames = len(euler_history)
ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=1200, repeat=False)

# Save GIF inside the folder
ani.save(os.path.join(output_folder, 'animation.gif'), writer='pillow', fps=1)
print(f"Animation and frames saved in folder '{output_folder}'.")
