# -----------------------------------------------------------------------------
# File:         Ford_Fulkerson_Animation.py
# Project:      Graph Algorithms Course Notes
# Author:       Mohammad Javad Abdolahi
# GitHub:       https://github.com/JavadAbdollahi
# Supervisor:   Dr. Behnaz Omoomi
# Date:         August 2025
# -----------------------------------------------------------------------------

import networkx as nx
import matplotlib.pyplot as plt
import imageio
import os
from collections import deque
from PIL import Image, ImageDraw, ImageFont

def add_footer_to_gif(gif_path, footer_text):
    """Adds a persistent footer to each frame of a GIF."""
    with Image.open(gif_path) as im:
        frames = []
        duration = im.info.get('duration', 100)
        
        for frame_num in range(im.n_frames):
            im.seek(frame_num)
            new_frame = im.copy().convert("RGBA")
            
            draw = ImageDraw.Draw(new_frame)
            font = ImageFont.load_default()
            
            bbox = draw.textbbox((0, 0), footer_text, font=font)
            textwidth = bbox[2] - bbox[0]
            textheight = bbox[3] - bbox[1]
            
            width, height = new_frame.size
            x = width - textwidth - 10
            y = height - textheight - 5
            
            draw.text((x, y), footer_text, font=font, fill=(128, 128, 128, 255))
            frames.append(new_frame)
        
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)


def ford_fulkerson_animation(graph, s, t):
    """
    Generates an animation for the Ford-Fulkerson algorithm using BFS (Edmonds-Karp).
    """
    folder_name = "Ford_Fulkerson_Animation"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    flow = {u: {v: 0 for v in graph.neighbors(u)} for u in graph.nodes()}
    frame_number = 0
    max_flow = 0
    
    # --- Manual layout for the graph ---
    pos = {
        's': [0, 0], 'a': [1, 1], 'b': [1, -1],
        'c': [2, 1], 'd': [2, -1], 't': [3, 0]
    }
    
    # --- Initial Frame ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Initial State: Flow = 0', fontsize=16)
    
    ax1.set_title('Main Graph (Flow / Capacity)')
    edge_labels_flow = {(u, v): f"0/{data['capacity']}" for u, v, data in graph.edges(data=True)}
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, ax=ax1, arrows=True)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_flow, label_pos=0.5, ax=ax1)
    
    ax2.set_title('Initial Residual Graph')
    residual_labels = {(u, v): data['capacity'] for u, v, data in graph.edges(data=True)}
    nx.draw(graph, pos, with_labels=True, node_color='lightgreen', node_size=700, ax=ax2, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=residual_labels, ax=ax2)  # بدون label_pos ثابت

    plt.savefig(os.path.join(folder_name, f'frame_{frame_number:03d}.png'))
    plt.close()
    frame_number += 1
    
    # --- Main Loop ---
    while True:
        residual_graph = nx.DiGraph()
        for u, v, data in graph.edges(data=True):
            if data['capacity'] - flow[u].get(v, 0) > 0:
                residual_graph.add_edge(u, v, capacity=data['capacity'] - flow[u].get(v, 0), is_forward=True)
            if flow[u].get(v, 0) > 0:
                residual_graph.add_edge(v, u, capacity=flow[u].get(v, 0), is_forward=False)

        parent = {node: None for node in graph.nodes()}
        queue = deque([s])
        visited = {s}
        path_found = False
        while queue:
            u = queue.popleft()
            if u == t:
                path_found = True
                break
            for v in sorted(list(residual_graph.neighbors(u))):
                if v not in visited and residual_graph.get_edge_data(u, v, {}).get('capacity', 0) > 0:
                    parent[v] = u
                    visited.add(v)
                    queue.append(v)
        
        if not path_found:
            break

        path, curr = [], t
        while curr is not None:
            path.append(curr)
            curr = parent[curr]
        path.reverse()
        
        bottleneck = min(residual_graph[u][v]['capacity'] for u, v in zip(path, path[1:]))
        max_flow += bottleneck
        
        for u, v in zip(path, path[1:]):
            flow[u][v] = flow[u].get(v, 0) + bottleneck
            flow.setdefault(v, {})
            flow[v][u] = flow[v].get(u, 0) - bottleneck
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(f'Iteration {frame_number}: Path Found with Bottleneck = {bottleneck}, Total Flow = {max_flow}', fontsize=16)
        
        # --- Main Graph Labels in center ---
        edge_labels_flow = {(u, v): f"{flow[u].get(v, 0)}/{data['capacity']}" for u, v, data in graph.edges(data=True)}
        nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=700, ax=ax1)
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels_flow, label_pos=0.5, ax=ax1)
        
        # Residual Graph
        edge_labels_res = {(u, v): data['capacity'] for u, v, data in residual_graph.edges(data=True)}
        forward_edges = [(u, v) for u, v, d in residual_graph.edges(data=True) if d.get('is_forward', False)]
        backward_edges = [(u, v) for u, v, d in residual_graph.edges(data=True) if not d.get('is_forward', True)]
        path_edges = list(zip(path, path[1:]))

        nx.draw_networkx_nodes(residual_graph, pos, node_color='lightgreen', node_size=700, ax=ax2)
        nx.draw_networkx_labels(residual_graph, pos, ax=ax2)
        nx.draw_networkx_edges(residual_graph, pos, edgelist=forward_edges, ax=ax2, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(residual_graph, pos, edgelist=backward_edges, style='dashed', ax=ax2, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edges(residual_graph, pos, edgelist=path_edges, edge_color='red', width=2.5, ax=ax2, connectionstyle='arc3,rad=0.1')
        nx.draw_networkx_edge_labels(residual_graph, pos, edge_labels=edge_labels_res, label_pos=0.3, ax=ax2)
        
        plt.savefig(os.path.join(folder_name, f'frame_{frame_number:03d}.png'))
        plt.close()
        frame_number += 1

    gif_path = os.path.join(folder_name, 'Ford_Fulkerson_Animation.gif')
    images = [imageio.imread(os.path.join(folder_name, f'frame_{i:03d}.png')) for i in range(frame_number)]
    imageio.mimsave(gif_path, images, duration=2.5)
    
    add_footer_to_gif(gif_path, 'github.com/JavadAbdollahi')

    print(f"Animation saved as '{gif_path}'")
    print(f"Maximum Flow: {max_flow}")
    return max_flow

if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edge('s', 'a', capacity=10)
    G.add_edge('s', 'b', capacity=12)
    G.add_edge('a', 'c', capacity=4)
    G.add_edge('a', 'd', capacity=8)
    G.add_edge('b', 'a', capacity=5)
    G.add_edge('b', 'd', capacity=9)
    G.add_edge('c', 't', capacity=10)
    G.add_edge('d', 'c', capacity=6)
    G.add_edge('d', 't', capacity=10)

    ford_fulkerson_animation(G, 's', 't')
