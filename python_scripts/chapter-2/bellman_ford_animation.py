# -----------------------------------------------------------------------------
# File:         bellman_ford_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         July 2025
# -----------------------------------------------------------------------------
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
OUTPUT_DIR = "bellman_ford_animation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

G = nx.DiGraph()
edges_with_weights = [
    ('S', 'A', 6), ('S', 'B', 7), ('A', 'C', 8),
    ('A', 'B', -4), ('B', 'C', -2), ('C', 'D', 5),
    ('C', 'A', -3), ('D', 'S', 1)
]
G.add_weighted_edges_from(edges_with_weights)
start_node = 'S'

# --- 2. Bellman-Ford Algorithm with History Tracking ---
def bellman_ford_with_history(graph, source):
    """
    Perform Bellman-Ford algorithm and collect state history for animation.

    Returns a list of dictionaries representing the algorithm state at each step.
    Each state includes:
      - current iteration number
      - currently processed edge
      - distances and predecessors
      - message for explanation
      - edge_queue to visualize traversal order
    """
    history = []
    distances = {node: float('infinity') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    distances[source] = 0

    edges = list(graph.edges(data=True))
    num_vertices = len(graph.nodes())
    edge_list_order = [(u, v) for u, v, _ in edges]

    # Initial state before iterations
    history.append({
        'iteration': 0,
        'current_edge': None,
        'distances': distances.copy(),
        'predecessors': predecessors.copy(),
        'message': 'Initialization',
        'edge_queue': edge_list_order
    })

    # Main Bellman-Ford relaxation loop
    for i in range(1, num_vertices):
        relaxed = False
        for u, v, data in edges:
            # Before relaxing each edge
            history.append({
                'iteration': i,
                'current_edge': (u, v),
                'distances': distances.copy(),
                'predecessors': predecessors.copy(),
                'message': f'Iter {i}: Relaxing ({u}->{v})',
                'edge_queue': edge_list_order
            })

            # Relaxation condition
            if distances[u] + data['weight'] < distances[v]:
                distances[v] = distances[u] + data['weight']
                predecessors[v] = u
                relaxed = True

                # After updating the distance
                history.append({
                    'iteration': i,
                    'current_edge': (u, v),
                    'distances': distances.copy(),
                    'predecessors': predecessors.copy(),
                    'message': f'Iter {i}: Dist({v}) updated',
                    'edge_queue': edge_list_order
                })
        if not relaxed:
            break

    # Final pass to check for negative-weight cycles
    for u, v, data in edges:
        history.append({
            'iteration': num_vertices,
            'current_edge': (u, v),
            'distances': distances.copy(),
            'predecessors': predecessors.copy(),
            'message': 'Checking for negative cycles...',
            'edge_queue': edge_list_order
        })
        if distances[u] + data['weight'] < distances[v]:
            history.append({
                'iteration': num_vertices,
                'current_edge': (u, v),
                'distances': distances.copy(),
                'predecessors': predecessors.copy(),
                'message': 'Negative Cycle Detected!',
                'edge_queue': edge_list_order
            })
            return history

    # Final state after successful execution
    history.append({
        'iteration': num_vertices,
        'current_edge': None,
        'distances': distances.copy(),
        'predecessors': predecessors.copy(),
        'message': 'Algorithm Finished',
        'edge_queue': edge_list_order
    })

    return history

bellman_ford_history = bellman_ford_with_history(G, start_node)

# --- 3. Animation Setup ---
pos = nx.circular_layout(G)
fig, (ax_graph, ax_data) = plt.subplots(1, 2, figsize=(18, 9))
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)
NODE_SIZE = 2000

straight_edges = [edge for edge in G.edges() if not G.has_edge(edge[1], edge[0])]
curved_edges = [edge for edge in G.edges() if G.has_edge(edge[1], edge[0])]
all_labels = nx.get_edge_attributes(G, 'weight')
straight_labels = {e: all_labels[e] for e in straight_edges}
curved_labels = {e: all_labels[e] for e in curved_edges}

# --- 4. Animation Update Function ---
def update(frame):
    ax_graph.clear()
    ax_data.clear()
    state = bellman_ford_history[frame]

    iteration = state['iteration']
    current_edge = state['current_edge']
    distances = state['distances']
    predecessors = state['predecessors']
    message = state['message']
    edge_queue = state['edge_queue']

    # --- Draw the graph ---
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=NODE_SIZE, node_color='#add8e6')
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=16, font_color='black')
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=straight_edges, edge_color='black', width=1.5,
                           arrows=True, arrowstyle='->', arrowsize=20, node_size=NODE_SIZE)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=curved_edges, connectionstyle='arc3, rad=0.2',
                           edge_color='black', width=1.5, arrows=True, arrowstyle='->', arrowsize=20, node_size=NODE_SIZE)
    if current_edge:
        rad = 0.2 if current_edge in curved_edges else 0
        nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=[current_edge], connectionstyle=f'arc3, rad={rad}',
                               edge_color='red', width=2.5, arrows=True, arrowstyle='->', arrowsize=20, node_size=NODE_SIZE)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=straight_labels, ax=ax_graph, font_size=14, label_pos=0.5,
                                 bbox=dict(facecolor='white', edgecolor='none', pad=1))
    for (u, v), weight in curved_labels.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    
        dx_line = x2 - x1
        dy_line = y2 - y1

        length = (dx_line**2 + dy_line**2)**0.5
        if length == 0:
            length = 1  

        nx_offset = -dy_line / length
        ny_offset = dx_line / length
        offset_factor = -0.22
    
        label_x = mx + nx_offset * offset_factor
        label_y = my + ny_offset * offset_factor
    
        ax_graph.text(label_x, label_y, f"{weight}",
                      fontsize=14,
                      bbox=dict(facecolor='white', edgecolor='none', pad=1))
    ax_graph.set_title("Graph Traversal", fontsize=18)

    # --- Draw the data state ---
    ax_data.axis('off')
    ax_data.set_title("Data Structures State", fontsize=18)
    ax_data.text(0.1, 0.95, f"Iteration: {iteration}", fontsize=14, weight='bold')

    # Display distances
    ax_data.text(0.1, 0.88, "Distances (d)", fontsize=14, weight='bold')
    sorted_nodes = sorted(distances.keys())
    for i, node in enumerate(sorted_nodes):
        dist_str = f"{distances[node]:.1f}" if distances[node] != float('inf') else 'inf'
        ax_data.text(0.1, 0.82 - i * 0.05, f"'{node}': {dist_str}", fontsize=12)

    # Display predecessors
    ax_data.text(0.6, 0.88, "Predecessors (\u03c0)", fontsize=14, weight='bold')
    for i, node in enumerate(sorted_nodes):
        pred = predecessors[node]
        pred_str = f"'{pred}'" if pred is not None else 'NIL'
        ax_data.text(0.6, 0.82 - i * 0.05, f"'{node}': {pred_str}", fontsize=12)

    # Display edge queue with highlight
    ax_data.text(0.1, 0.25, "Edge Queue (Edge Traversal Order)", fontsize=14, weight='bold')
    for j, (u, v) in enumerate(edge_queue):
        if current_edge is not None:
            cur_index = edge_queue.index(current_edge)
        else:
            cur_index = -1
        if j < cur_index:
            marker = '✓'
        elif j == cur_index:
            marker = '➤'
        else:
            marker = ' '
        ax_data.text(0.1, 0.2 - j * 0.035, f"{marker} ({u}→{v})", fontsize=11)

    fig.suptitle(f'Step {frame}: {message}', fontsize=20, y=0.98)
    frame_filename = os.path.join(OUTPUT_DIR, f'frame_{frame:03d}.png')
    plt.savefig(frame_filename)

# --- 5. Create and Save the Animation ---
ani = animation.FuncAnimation(fig, update, frames=len(bellman_ford_history), interval=800, repeat=False)
try:
    gif_filename = os.path.join(OUTPUT_DIR, 'bellman_ford_animation.gif')
    print(f"Saving animation to '{gif_filename}'...")
    ani.save(gif_filename, writer='pillow', fps=1.25)
    print("Animation and frames saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")
