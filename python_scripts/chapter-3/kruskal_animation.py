# -----------------------------------------------------------------------------
# File:         kruskal_animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         August 2025
# -----------------------------------------------------------------------------

import os
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# -----------------------------------------------------------------------------
# Description:  This script animates Kruskal's algorithm for finding a Minimum
#               Spanning Tree (forest). It demonstrates sorting edges,
#               selecting safe edges, and displays the state of the
#               Disjoint-Set (Union-Find) structure, stopping exactly when
#               |V| - k edges have been selected (k = # of connected components).
# -----------------------------------------------------------------------------

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
OUTPUT_DIR = "kruskal_animation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Graph Definition: designed to produce multiple rank increments ---
G = nx.Graph()
edges_with_weights = [
    # initial unions produce rank 1
    ('A', 'B', 1), ('C', 'D', 2),
    # merging two trees of rank 1 -> rank 2
    ('A', 'C', 3),
    # another initial union
    ('E', 'F', 4), ('G', 'H', 5),
    # merging those rank 1 trees -> rank 2
    ('E', 'G', 6),
    # finally merge two rank-2 trees -> rank 3
    ('A', 'E', 7),
    # extra edges to skip cycles
    ('B', 'C', 8), ('F', 'G', 9)
]
G.add_weighted_edges_from(edges_with_weights)

# --- Compute stopping condition: select |V| - k edges ---
n = G.number_of_nodes()
k = nx.number_connected_components(G)
max_edges = n - k

# --- 2. Disjoint-Set (Union-Find) Implementation with History ---
class UnionFind:
    def __init__(self, nodes):
        self.parent = {v: v for v in nodes}
        self.rank   = {v: 0 for v in nodes}
        self.history = []
        self.record('init')

    def find(self, v):
        if self.parent[v] != v:
            self.parent[v] = self.find(self.parent[v])
        return self.parent[v]

    def union(self, u, v):
        ru, rv = self.find(u), self.find(v)
        if ru == rv:
            self.record(f'skip union {u},{v}')
            return False
        if self.rank[ru] < self.rank[rv]:
            self.parent[ru] = rv
        elif self.rank[ru] > self.rank[rv]:
            self.parent[rv] = ru
        else:
            self.parent[rv] = ru
            self.rank[ru] += 1
        self.record(f'union {u},{v}')
        return True

    def record(self, message):
        # snapshot of DSU state: (message, parent dict, rank dict)
        self.history.append((message, dict(self.parent), dict(self.rank)))


def kruskal_with_history(graph):
    history = []
    edges = sorted(graph.edges(data='weight'), key=lambda x: x[2])
    uf = UnionFind(graph.nodes())
    mst_edges = []

    # initial state
    history.append({
        'step': 'sorted_edges',
        'edges': edges.copy(),
        'mst': mst_edges.copy(),
        'uf_state': uf.history[-1]
    })

    # main Kruskal loop with stopping condition
    for u, v, w in edges:
        if uf.union(u, v):
            mst_edges.append((u, v))
            history.append({
                'step': f'select {(u,v)}',
                'edges': edges.copy(),
                'mst': mst_edges.copy(),
                'uf_state': uf.history[-1]
            })
            if len(mst_edges) >= max_edges:
                break
        else:
            history.append({
                'step': f'skip {(u,v)}',
                'edges': edges.copy(),
                'mst': mst_edges.copy(),
                'uf_state': uf.history[-1]
            })

    # final state
    history.append({
        'step': 'finished',
        'edges': edges.copy(),
        'mst': mst_edges.copy(),
        'uf_state': uf.history[-1]
    })

    return history

kruskal_history = kruskal_with_history(G)

# --- 3. Animation Setup ---
pos = nx.spring_layout(G, seed=42)
fig, (ax_graph, ax_data) = plt.subplots(1, 2, figsize=(18, 9))
NODE_SIZE = 2000

# --- 4. Update Function for Animation ---
def update(frame):
    ax_graph.clear()
    ax_data.clear()

    state      = kruskal_history[frame]
    step       = state['step']
    edges      = state['edges']
    mst_edges  = state['mst']
    msg, parent, rank = state['uf_state']

    # draw graph nodes
    used_nodes = set(sum(([u, v] for u, v in mst_edges), []))
    colors = ['lightblue' if v in used_nodes else 'lightgray' for v in G.nodes()]
    nx.draw_networkx_nodes(G, pos, ax=ax_graph, node_size=NODE_SIZE, node_color=colors)
    nx.draw_networkx_labels(G, pos, ax=ax_graph, font_size=16)

    # draw edges
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edge_color='lightgray', width=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax_graph, edgelist=mst_edges, edge_color='blue', width=3)
    nx.draw_networkx_edge_labels(G, pos,
        edge_labels=nx.get_edge_attributes(G,'weight'),
        ax=ax_graph, font_size=12)

    ax_graph.set_title("Kruskal's Algorithm", fontsize=18)

    # draw DSU state
    ax_data.axis('off')
    ax_data.set_title('Disjoint-Set (UF) State', fontsize=18)
    ax_data.text(0.05, 0.90, f'Step: {step}', fontsize=14, weight='bold')
    ax_data.text(0.05, 0.85, 'Parent pointers:', fontsize=12)
    for i, v in enumerate(sorted(parent)):
        ax_data.text(0.10, 0.80 - i*0.05, f'{v} → {parent[v]}', fontsize=12)
    ax_data.text(0.55, 0.85, 'Ranks:', fontsize=12)
    for i, v in enumerate(sorted(rank)):
        ax_data.text(0.58, 0.80 - i*0.05, f'{v}: {rank[v]}', fontsize=12)

    fig.suptitle(f'{step}: {msg}', fontsize=20, y=0.98)
    plt.savefig(os.path.join(OUTPUT_DIR, f'frame_{frame:03d}.png'))


# --- 5. Create and Save Animation ---
ani = animation.FuncAnimation(fig, update,
                              frames=len(kruskal_history),
                              interval=1000, repeat=False)

try:
    ani.save(os.path.join(OUTPUT_DIR, 'kruskal_animation.gif'),
             writer='pillow', fps=1)
    print('Animation saved successfully.')
except Exception as e:
    print(f'Error saving animation: {e}')
