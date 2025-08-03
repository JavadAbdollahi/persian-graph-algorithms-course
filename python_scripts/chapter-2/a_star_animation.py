# -*- coding: utf-8 -*-
"""
Created on Sat Aug  2 17:52:59 2025

@author: Javad
"""

# -----------------------------------------------------------------------------
# File:         a_star_animation.py
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
import heapq

"""
Description:  This script animates the A* search algorithm on a grid.
              It shows the OPEN and CLOSED lists and the f, g, h scores.
"""

# --- 1. Constants and Setup ---
GITHUB_URL = "https://github.com/JavadAbdollahi"
OUTPUT_DIR = "a_star_animation_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Grid world definition
# 0 = traversable, 1 = obstacle
grid = [
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]
start_node = (0, 0)
goal_node = (4, 4)

# Heuristic function (Octile distance for grid with diagonal movement)
def heuristic(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return 1.4 * min(dx, dy) + 1 * (max(dx, dy) - min(dx, dy))

# --- 2. A* Algorithm with History Tracking ---
def a_star_with_history(grid_map, start, goal):
    history = []
    
    open_list = []
    heapq.heappush(open_list, (0, start)) # (f_score, node)
    
    came_from = {}
    g_score = { (r, c): float('inf') for r, row in enumerate(grid_map) for c, val in enumerate(row) }
    g_score[start] = 0
    
    f_score = { (r, c): float('inf') for r, row in enumerate(grid_map) for c, val in enumerate(row) }
    f_score[start] = heuristic(start, goal)
    
    closed_set = set()

    # Initial state
    history.append({
        'current_node': None,
        'open_list': list(open_list),
        'closed_set': closed_set.copy(),
        'g_scores': g_score.copy(),
        'path': [],
        'message': 'Initialization'
    })

    while open_list:
        _, current = heapq.heappop(open_list)
        
        if current == goal:
            path = []
            temp = current
            while temp in came_from:
                path.append(temp)
                temp = came_from[temp]
            path.append(start)
            path.reverse()
            history.append({'current_node': current, 'open_list': [], 'closed_set': closed_set, 'g_scores': g_score, 'path': path, 'message': 'Goal Reached!'})
            return history
            
        closed_set.add(current)

        # State after choosing a node
        history.append({
            'current_node': current,
            'open_list': list(open_list),
            'closed_set': closed_set.copy(),
            'g_scores': g_score.copy(),
            'path': [],
            'message': f'Expanding node {current}'
        })

        # Explore neighbors
        for dr, dc, cost in [(0,1,1), (0,-1,1), (1,0,1), (-1,0,1), (1,1,1.4), (1,-1,1.4), (-1,1,1.4), (-1,-1,1.4)]:
            neighbor = (current[0] + dr, current[1] + dc)
            
            if not (0 <= neighbor[0] < len(grid_map) and 0 <= neighbor[1] < len(grid_map[0]) and grid_map[neighbor[0]][neighbor[1]] == 0):
                continue

            if neighbor in closed_set:
                continue
            
            tentative_g_score = g_score[current] + cost
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                
                if not any(item[1] == neighbor for item in open_list):
                    heapq.heappush(open_list, (f_score[neighbor], neighbor))
                
                history.append({
                    'current_node': current,
                    'open_list': list(open_list),
                    'closed_set': closed_set.copy(),
                    'g_scores': g_score.copy(),
                    'path': [],
                    'message': f'Updating neighbor {neighbor}'
                })
    
    history.append({'current_node': None, 'open_list': [], 'closed_set': closed_set, 'g_scores': g_score, 'path': [], 'message': 'Failed to find a path'})
    return history


a_star_history = a_star_with_history(grid, start_node, goal_node)

# --- 3. Animation Setup ---
fig, (ax_grid, ax_data) = plt.subplots(1, 2, figsize=(18, 9))
fig.text(0.99, 0.01, GITHUB_URL, fontsize=10, color='gray', ha='right', va='bottom', alpha=0.7)

# --- 4. Animation Update Function (REVISED) ---
def update(frame):
    ax_grid.clear()
    state = a_star_history[frame]
    current, open_list_tuples, closed_set, g_scores, path, message = state.values()
    open_list_nodes = [node for _, node in open_list_tuples]

    # (Grid and Node Drawing - Unchanged)
    ax_grid.set_xticks(range(len(grid[0]) + 1))
    ax_grid.set_yticks(range(len(grid) + 1))
    ax_grid.grid(True)
    for r, row in enumerate(grid):
        for c, val in enumerate(row):
            color = 'black' if val == 1 else 'white'
            ax_grid.fill_between([c, c+1], r, r+1, color=color, alpha=0.3 if color=='black' else 1.0)
    for r, c in closed_set: ax_grid.fill_between([c, c+1], r, r+1, color='lightblue')
    for r, c in open_list_nodes: ax_grid.fill_between([c, c+1], r, r+1, color='lightgreen')
    if current: ax_grid.fill_between([current[1], current[1]+1], current[0], current[0]+1, color='orange')
    ax_grid.fill_between([start_node[1], start_node[1]+1], start_node[0], start_node[0]+1, color='green')
    ax_grid.fill_between([goal_node[1], goal_node[1]+1], goal_node[0], goal_node[0]+1, color='red')
    if path:
        path_x = [p[1] + 0.5 for p in path]
        path_y = [p[0] + 0.5 for p in path]
        ax_grid.plot(path_x, path_y, color='blue', linewidth=3)
    ax_grid.set_xlim(0, len(grid[0]))
    ax_grid.set_ylim(0, len(grid))
    ax_grid.invert_yaxis()
    ax_grid.set_aspect('equal', adjustable='box')
    ax_grid.set_title("A* Search on Grid", fontsize=18)
    
    # --- Update Data Subplot (REVISED) ---
    ax_data.clear()
    ax_data.axis('off')
    ax_data.set_title("Data Structures State", fontsize=18)
    
    # --- Column 1: OPEN List ---
    ax_data.text(0.05, 0.95, "OPEN List (f=g+h)", fontsize=14, weight='bold', va='top')
    
    open_display = []
    temp_open_list = [(g_scores.get(n, float('inf')) + heuristic(n, goal_node), g_scores.get(n, float('inf')), heuristic(n, goal_node), n) for n in open_list_nodes]
    temp_open_list.sort()
    
    for i, (f, g, h, node) in enumerate(temp_open_list[:15]): # Display up to 15 items
        text = f"{node}: {f:.1f} = {g:.1f} + {h:.1f}"
        ax_data.text(0.05, 0.85 - i * 0.05, text, fontsize=11)

    # --- Column 2: CLOSED Set ---
    ax_data.text(0.6, 0.95, "CLOSED Set", fontsize=14, weight='bold', va='top')
    sorted_closed = sorted(list(closed_set)) # Sort for consistent display
    for i, node in enumerate(sorted_closed[:15]): # Display up to 15 items
        ax_data.text(0.6, 0.85 - i * 0.05, str(node), fontsize=11)
        
    fig.suptitle(f'Step {frame}: {message}', fontsize=20, y=0.98)
    
    frame_filename = os.path.join(OUTPUT_DIR, f'a_star_animation_{frame:03d}.png')
    plt.savefig(frame_filename)

# --- 5. Create and Save the Animation ---
ani = animation.FuncAnimation(fig, update, frames=len(a_star_history), interval=300, repeat=False)
try:
    gif_filename = os.path.join(OUTPUT_DIR, 'a_star_animation.gif')
    print(f"Saving animation to '{gif_filename}'...")
    ani.save(gif_filename, writer='pillow', fps=3)
    print("Animation and frames saved successfully.")
except Exception as e:
    print(f"Error saving animation: {e}")