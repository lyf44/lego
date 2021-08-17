import numpy as np
import networkx as nx
from itertools import chain
try:
    from . import helper
    from . import astar
except:
    import helper
    import astar
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import math

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def visualize_nodes(curr_occ_grid, curr_node_posns, start_pos, goal_pos):
    fig1 = plt.figure(figsize=(10,6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    # occ_g = curr_occ_grid.reshape(10,10)
    occ_g = curr_occ_grid.reshape(50, 50)

    for i in range(50):
        for j in range(50):
            if occ_g[i,j] == 255:
                # ax1.add_patch(patches.Rectangle(
                #     (i/10.0 - 2.5, j/10.0 - 2.5),   # (x,y)
                #     0.1,          # width
                #     0.1,          # height
                #     alpha=0.6
                #     ))
                plt.scatter(j/10.0 - 2.55, 2.45 - i/10.0, color="black", s=100, alpha=1) # init
    plt.scatter(start_pos[0], start_pos[1], color="red", s=100, edgecolors='black', alpha=1, zorder=10) # init
    plt.scatter(goal_pos[0], goal_pos[1], color="blue", s=100, edgecolors='black', alpha=1, zorder=10) # goal

    curr_node_posns = np.array(curr_node_posns)
    if len(curr_node_posns)>0:
        plt.scatter(curr_node_posns[:,0], curr_node_posns[:,1], s = 50, color = 'green')
    plt.title("Visualization")
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.show()

def get_bottleneck_nodes(dense_G, sparse_G, path_nodes, occ_grid, dis, row, col, inc):
    assert len(path_nodes)>0
    THRESH = 0.1
    P_EPS = 10
    new_edges = []
    start_n = 'p0'
    goal_n = 'p'+str(len(path_nodes)-1)
    for count in range(len(path_nodes)):
        s1 = dense_G.nodes[path_nodes[count]]['coords']
        sparse_G.add_node('p'+str(count), coords = s1)
        if count>0:
            n1 = 'p'+str(count-1)
            n2 = 'p'+str(count)
            sparse_G.add_edge(n1, n2)
            sparse_G[n1][n2]['weight'] = helper.calc_weight_states(sparse_G.nodes[n1]['coords'], sparse_G.nodes[n2]['coords'])
            new_edges.append((n1, n2))
        for node in sparse_G.nodes():
            s2 = sparse_G.nodes[node]['coords']
            # s2 = node
            if 'p' in node:
                continue

            if helper.calc_weight_states(s1, s2) < THRESH:
                n1 = node
                n2 = 'p'+str(count)
                sparse_G.add_edge(node, 'p'+str(count))
                sparse_G.add_edge('p'+str(count), node)
                sparse_G[n1][n2]['weight'] = helper.calc_weight_states(sparse_G.nodes[n1]['coords'], sparse_G.nodes[n2]['coords'])
                sparse_G[n2][n1]['weight'] = helper.calc_weight_states(sparse_G.nodes[n1]['coords'], sparse_G.nodes[n2]['coords'])
                new_edges.append((node, 'p'+str(count)))
                new_edges.append(('p'+str(count), node))

    curr_path_nodes, curr_dis = astar.astar(sparse_G, start_n, goal_n, occ_grid, row, col, inc)
    while curr_dis < P_EPS*dis:
        for edge in new_edges:
            sparse_G[edge[0]][edge[1]]['weight'] *= 1.2
        last_dis = curr_dis
        curr_path_nodes, curr_dis = astar.astar(sparse_G, start_n, goal_n, occ_grid, row, col, inc)
        if math.fabs(last_dis - curr_dis) < 1e-4:
            break

    common_nodes = []
    for n in curr_path_nodes:
        if 'p' in n:
            common_nodes.append(n)

    return common_nodes, curr_path_nodes

def test(occ_grid, dense_G, sparse_G):
    INC = 0
    row = None
    col = None
    path_nodes = []
    while len(path_nodes) == 0:
        start_n, goal_n = helper.get_valid_start_goal(dense_G, occ_grid, row, col, inc = INC)
        start_pos = helper.state_to_numpy(dense_G.nodes[start_n]['coords'])
        goal_pos = helper.state_to_numpy(dense_G.nodes[goal_n]['coords'])
        path_nodes, dis = astar.astar(dense_G, start_n, goal_n, occ_grid, row, col, inc = INC)

    bottleneck_nodes, curr_path_nodes = get_bottleneck_nodes(dense_G, sparse_G, path_nodes, occ_grid, dis, row, col, inc = INC)
    print("bottleneck_nodes = ", bottleneck_nodes)
    points_x = []
    points_y = []
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in path_nodes:
        s = helper.state_to_numpy(dense_G.nodes[node]['coords'])
        points_x.append(s[0])
        points_y.append(s[1])

    # visualize_nodes(occ_grid,np.array(list(zip(points_x,points_y))), start_pos, goal_pos)

    points_x = []
    points_y = []
    ss = []
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in bottleneck_nodes:
        s = helper.state_to_numpy(sparse_G.nodes[node]['coords'])
        points_x.append(s[0])
        points_y.append(s[1])
        ss.append(s)

    # visualize_nodes(occ_grid,np.array(list(zip(points_x,points_y))), start_pos, goal_pos)
    return start_pos, goal_pos, ss

def main():
    INC = 0.03
    dense_G = nx.read_graphml(os.path.join(CUR_DIR, "../graphs/dense_graph.graphml"))
    sparse_G = nx.read_graphml(os.path.join(CUR_DIR, "../graphs/sparse_graph.graphml"))

    occ_grid, row, col = helper.get_random_occ_grid()
    print(row, col)
    start_n, goal_n = helper.get_valid_start_goal(dense_G, occ_grid, row, col, inc = INC)
    start_pos = helper.state_to_numpy(dense_G.nodes[start_n]['coords'])
    goal_pos = helper.state_to_numpy(dense_G.nodes[goal_n]['coords'])

    path_nodes, dis = astar.astar(dense_G, start_n, goal_n, occ_grid, row, col, inc = INC)
    bottleneck_nodes, curr_path_nodes = get_bottleneck_nodes(dense_G, sparse_G, path_nodes, occ_grid, dis, row, col, inc = INC)
    print("bottleneck_nodes = ", bottleneck_nodes)
    points_x = []
    points_y = []
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in path_nodes:
        s = helper.state_to_numpy(dense_G.nodes[node]['coords'])
        points_x.append(s[0])
        points_y.append(s[1])

    visualize_nodes(occ_grid,np.array(list(zip(points_x,points_y))), start_pos, goal_pos)

    points_x = []
    points_y = []
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in bottleneck_nodes:
        s = helper.state_to_numpy(sparse_G.node[node]['state'])
        points_x.append(s[0])
        points_y.append(s[1])

    visualize_nodes(occ_grid,np.array(list(zip(points_x,points_y))), start_pos, goal_pos)

if __name__ == '__main__':
    main()