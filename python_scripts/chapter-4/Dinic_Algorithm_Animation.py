# -----------------------------------------------------------------------------
# File:         Dinic_Algorithm_Animation.py
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
    with Image.open(gif_path) as im:
        frames = []
        duration = im.info.get('duration', 2000)
        for frame_num in range(im.n_frames):
            im.seek(frame_num)
            new_frame = im.copy().convert("RGBA")
            draw = ImageDraw.Draw(new_frame)
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), footer_text, font=font)
            textwidth, textheight = bbox[2] - bbox[0], bbox[3] - bbox[1]
            width, height = new_frame.size
            x, y = width - textwidth - 10, height - textheight - 5
            draw.text((x, y), footer_text, font=font, fill=(128, 128, 128, 255))
            frames.append(new_frame)
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=duration, loop=0)

def dinic_animation(graph, s, t):
    folder_name = "Dinic_Algorithm_Animation"
    os.makedirs(folder_name, exist_ok=True)

    residual_graph = nx.DiGraph()
    for u, v, data in graph.edges(data=True):
        residual_graph.add_edge(u, v, capacity=data['capacity'])
        if not residual_graph.has_edge(v, u):
            residual_graph.add_edge(v, u, capacity=0)

    frame_number = 0
    max_flow = 0

    # --- STABLE LAYOUT ---
    initial_level = {node: -1 for node in graph.nodes()}
    initial_level[s] = 0
    q_layout = deque([s])
    max_level = 0
    while q_layout:
        u = q_layout.popleft()
        for v in graph.neighbors(u):
            if initial_level[v] == -1:
                initial_level[v] = initial_level[u] + 1
                max_level = max(max_level, initial_level[v])
                q_layout.append(v)

    stable_pos = {}
    nodes_in_level = {i: [] for i in range(max_level + 1)}
    for node in sorted(graph.nodes()):
        lvl = initial_level[node]
        if lvl != -1:
            nodes_in_level[lvl].append(node)

    for lvl, nodes in nodes_in_level.items():
        y_start = (len(nodes) - 1) / 2.0
        for i, node in enumerate(nodes):
            stable_pos[node] = (lvl, y_start - i)

    # --- Visualization Helper ---
    def draw_state(phase_title, frame_title, current_level_dict, level_graph, current_path=None):
        nonlocal frame_number
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
        fig.suptitle(f'{phase_title}\n{frame_title} | Total Flow = {max_flow}', fontsize=18)

        # --- Main Graph ---
        ax1.set_title('Main Graph (Flow / Capacity)')
        flow_labels = {(u,v): f"{graph[u][v]['capacity'] - residual_graph[u][v]['capacity']}/{graph[u][v]['capacity']}" 
                       for u,v in graph.edges()}
        nx.draw(graph, stable_pos, with_labels=True, node_color='lightblue', node_size=1200, ax=ax1,
                arrowsize=20, arrowstyle='-|>', min_source_margin=15, min_target_margin=15)
        nx.draw_networkx_edge_labels(graph, stable_pos, edge_labels=flow_labels, font_size=10, label_pos=0.5, ax=ax1)

        # --- Level Graph ---
        ax2.set_title('Level Graph')
        nodes_to_draw = list(level_graph.nodes()) if level_graph else list(graph.nodes())
        node_labels = {}
        for n in nodes_to_draw:
            lvl = current_level_dict.get(n, '-')
            node_labels[n] = f"{n}"

        nx.draw_networkx_nodes(nodes_to_draw, stable_pos, node_color='lightgreen', node_size=1000, ax=ax2)
        nx.draw_networkx_labels(level_graph if level_graph else graph, stable_pos, labels=node_labels, font_size=12, ax=ax2)

        # Draw levels next to nodes
        for n in nodes_to_draw:
            lvl = current_level_dict.get(n, '-')
            x, y = stable_pos[n]
            ax2.text(x + 0.1, y, f"L:{lvl}", color='darkgreen', fontsize=12, fontweight='bold')

        if level_graph:
            edge_labels_level = {(u,v): d['capacity'] for u,v,d in level_graph.edges(data=True) if d['capacity']>0}
            nx.draw_networkx_edges(
                level_graph, stable_pos, edge_color='black', width=1.5, ax=ax2,
                arrowsize=20, arrowstyle='-|>',
                min_source_margin=15, min_target_margin=15
            )
            nx.draw_networkx_edge_labels(level_graph, stable_pos, edge_labels=edge_labels_level, label_pos=0.5, font_size=10, ax=ax2)

            if current_path:
                path_edges = list(zip(current_path, current_path[1:]))
                nx.draw_networkx_edges(
                    level_graph, stable_pos, edgelist=path_edges, edge_color='red', width=3.0, ax=ax2,
                    arrowsize=25, arrowstyle='-|>',
                    min_source_margin=15, min_target_margin=15
                )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(os.path.join(folder_name, f'frame_{frame_number:03d}.png'))
        plt.close()
        frame_number += 1

    # --- Dinic's Algorithm ---
    phase_count = 1
    while True:
        level = {node: -1 for node in graph.nodes()}
        level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v, data in residual_graph[u].items():
                if level[v] < 0 and data['capacity'] > 0:
                    level[v] = level[u] + 1
                    q.append(v)

        if level[t] < 0:
            break

        level_graph = nx.DiGraph()
        for u, v, data in residual_graph.edges(data=True):
            if level.get(u, -1) != -1 and level.get(v, -1) == level[u] + 1 and data['capacity'] > 0:
                level_graph.add_edge(u, v, capacity=data['capacity'])

        draw_state(f'Phase {phase_count}', 'Building Level Graph', level, level_graph)

        ptr = {node: 0 for node in graph.nodes()}

        def find_path_dfs(u, pushed, path_so_far):
            if pushed == 0: return 0, []
            if u == t: return pushed, path_so_far + [u]
            neighbors = list(level_graph.neighbors(u))
            while ptr[u] < len(neighbors):
                v = neighbors[ptr[u]]
                if level_graph.has_edge(u,v) and level_graph[u][v]['capacity'] > 0:
                    tr, final_path = find_path_dfs(v, min(pushed, level_graph[u][v]['capacity']), path_so_far + [u])
                    if tr > 0: return tr, final_path
                ptr[u] += 1
            return 0, []

        path_in_phase_count = 1
        while True:
            pushed, path = find_path_dfs(s, float('inf'), [])
            if pushed == 0: break
            max_flow += pushed
            for u, v in zip(path, path[1:]):
                residual_graph[u][v]['capacity'] -= pushed
                residual_graph[v][u]['capacity'] += pushed
                if level_graph.has_edge(u, v):
                    level_graph[u][v]['capacity'] -= pushed
            draw_state(f'Phase {phase_count}', f'Finding Blocking Flow (Path {path_in_phase_count}, Pushed={pushed})', level, level_graph, current_path=path)
            path_in_phase_count += 1
        phase_count += 1

    draw_state("Final", "Algorithm Terminated", {n:'' for n in graph.nodes()}, None)

    gif_path = os.path.join(folder_name, 'Dinic_Algorithm_Animation.gif')
    images = [imageio.imread(os.path.join(folder_name, f'frame_{i:03d}.png')) for i in range(frame_number)]
    imageio.mimsave(gif_path, images, duration=2.5)
    add_footer_to_gif(gif_path, 'github.com/JavadAbdollahi')

    print(f"\nAnimation saved as '{gif_path}'\n\n with total flow = {max_flow}")
    return max_flow

# ------------------ TEST EXAMPLE ------------------
if __name__ == '__main__':
    G = nx.DiGraph()
    G.add_edge('S', 'A', capacity=10)
    G.add_edge('A', 'D', capacity=8)
    G.add_edge('S', 'C', capacity=10)
    G.add_edge('A', 'B', capacity=4)
    G.add_edge('A', 'C', capacity=2)
    G.add_edge('C', 'D', capacity=9)
    G.add_edge('B', 'T', capacity=10)
    G.add_edge('D', 'B', capacity=6)
    G.add_edge('D', 'T', capacity=10)

    maxflow = dinic_animation(G, 'S', 'T')
