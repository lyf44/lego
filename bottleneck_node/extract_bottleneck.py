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
import copy
import math

import lego.utils as utils

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

def get_bottleneck_nodes(maze, dense_G, sparse_G, path_nodes, occ_grid, dis, row, col, inc):
    assert len(path_nodes)>0
    THRESH = 0.1
    P_EPS = 2
    new_edges = []
    start_n = 'p0'
    goal_n = 'p'+str(len(path_nodes)-1)
    for count in range(len(path_nodes)):
        path_node = 'p' + str(count)
        s1 = dense_G.nodes[path_nodes[count]]['coords']
        sparse_G.add_node(path_node, coords = s1)
        if count > 0:
            n1 = 'p' + str(count-1)
            n2 = path_node
            sparse_G.add_edge(n1, n2)
            sparse_G.add_edge(n2, n1)
            sparse_G[n1][n2]['weight'] = helper.calc_weight_states(sparse_G.nodes[n1]['coords'], sparse_G.nodes[n2]['coords'])
            sparse_G[n2][n1]['weight'] = helper.calc_weight_states(sparse_G.nodes[n1]['coords'], sparse_G.nodes[n2]['coords'])
            new_edges.append((n1, n2))
            new_edges.append((n2, n1))
        for node in sparse_G.nodes():
            s2 = sparse_G.nodes[node]['coords']
            # s2 = node
            if 'p' in node:
                continue

            # if helper.calc_weight_states(s1, s2) < THRESH:
            if utils.is_edge_free(maze, s1, s2):
                # n1 = node
                # n2 = 'p'+str(count)
                sparse_G.add_edge(node, path_node)
                sparse_G.add_edge(path_node, node)
                sparse_G[node][path_node]['weight'] = helper.calc_weight_states(s1, s2)
                sparse_G[path_node][node]['weight'] = helper.calc_weight_states(s1, s2)
                new_edges.append((node, path_node))
                new_edges.append((path_node, node))

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

def test(maze, occ_grid, dense_G, sparse_G):
    INC = 0
    row = None
    col = None
    path_nodes = []
    sparse_G = copy.deepcopy(sparse_G)

    while len(path_nodes) == 0:
        start_n, goal_n = helper.get_valid_start_goal(dense_G, occ_grid, row, col, inc = INC)
        start_pos = helper.state_to_numpy(dense_G.nodes[start_n]['coords'])
        goal_pos = helper.state_to_numpy(dense_G.nodes[goal_n]['coords'])
        path_nodes, dis = astar.astar(dense_G, start_n, goal_n, occ_grid, row, col, inc = INC)

    bottleneck_nodes, curr_path_nodes = get_bottleneck_nodes(maze, dense_G, sparse_G, path_nodes, occ_grid, dis, row, col, inc = INC)
    print("bottleneck_nodes = ", bottleneck_nodes)
    points_x = []
    points_y = []
    print("path_nodes = ", path_nodes, curr_path_nodes)
    for node in path_nodes:
        s = helper.state_to_numpy(dense_G.nodes[node]['coords'])
        points_x.append(s[0])
        points_y.append(s[1])

    # visualize_nodes(occ_grid, np.array(list(zip(points_x,points_y))), start_pos, goal_pos)

    points_x = []
    points_y = []
    ss = []
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in bottleneck_nodes:
        s = helper.state_to_numpy(sparse_G.nodes[node]['coords'])
        points_x.append(s[0])
        points_y.append(s[1])
        ss.append(s)

    # visualize_nodes(occ_grid, np.array(list(zip(points_x,points_y))), start_pos, goal_pos)
    return start_pos, goal_pos, ss

def test2(maze, occ_grid, dense_G, sparse_G, start_n):
    INC = 0
    row = None
    col = None
    path_nodes = []
    sparse_G = copy.deepcopy(sparse_G)

    while len(path_nodes) == 0:
        _, goal_n = helper.get_valid_start_goal(dense_G, occ_grid, row, col, inc = INC)
        start_pos = helper.state_to_numpy(dense_G.nodes[start_n]['coords'])
        goal_pos = helper.state_to_numpy(dense_G.nodes[goal_n]['coords'])
        if helper.calc_weight_states(dense_G.nodes[start_n]['coords'], dense_G.nodes[goal_n]['coords']) < 0.1:
            continue
        path_nodes, dis = astar.astar(dense_G, start_n, goal_n, occ_grid, row, col, inc = INC)

    bottleneck_nodes, curr_path_nodes = get_bottleneck_nodes(maze, dense_G, sparse_G, path_nodes, occ_grid, dis, row, col, inc = INC)
    print("bottleneck_nodes = ", bottleneck_nodes)
    states = []
    print("path_nodes = ", path_nodes, curr_path_nodes)
    for node in curr_path_nodes:
        s = helper.state_to_numpy(sparse_G.nodes[node]['coords'])
        states.append(s)

    # utils.visualize_nodes(occ_grid, states, start_pos, goal_pos)

    points_x = []
    points_y = []
    ss = []

    # remove start and goal
    s1 = sparse_G.nodes[curr_path_nodes[0]]['coords']
    s2 = sparse_G.nodes[curr_path_nodes[1]]['coords']
    if helper.calc_weight_states(s1, s2) < 0.1:
        curr_path_nodes = curr_path_nodes[2:]
    else:
        curr_path_nodes = curr_path_nodes[1:]
    if len(curr_path_nodes) > 1:
        s1 = sparse_G.nodes[curr_path_nodes[-1]]['coords']
        s2 = sparse_G.nodes[curr_path_nodes[-2]]['coords']
        if helper.calc_weight_states(s1, s2) < 0.1:
            curr_path_nodes = curr_path_nodes[:-2]
        else:
            curr_path_nodes = curr_path_nodes[:-1]
    else:
        curr_path_nodes = curr_path_nodes[:-1]

    if len(curr_path_nodes) == 0:
        # bottleneck_nodes.append(curr_path_nodes[1]) # append second node in curr_path
        return goal_pos, ss

    # # handle the corner case where the start node is in the sparse graph and hence the first two nodes in path are the same
    # s1 = sparse_G.nodes[curr_path_nodes[0]]['coords']
    # s2 = sparse_G.nodes[curr_path_nodes[1]]['coords']
    # if helper.calc_weight_states(s1, s2) < 0.1:
    #     bottleneck_nodes  = [curr_path_nodes[2]]

    #     # sanity check
    #     # if len(curr_path_nodes) > 3:
    #     #     assert not sparse_G.has_edge(curr_path_nodes[1], curr_path_nodes[3])
    #     #     s1 = sparse_G.nodes[curr_path_nodes[1]]['coords']
    #     #     s2 = sparse_G.nodes[curr_path_nodes[3]]['coords']
    #     #     assert not utils.is_edge_free(maze, s1, s2)
    # else:
    #     bottleneck_nodes  = [curr_path_nodes[1]]

    #     # sanity check
    #     # if len(curr_path_nodes) > 2:
    #     #     assert not sparse_G.has_edge(curr_path_nodes[0], curr_path_nodes[2])
    #     #     s1 = sparse_G.nodes[curr_path_nodes[0]]['coords']
    #     #     s2 = sparse_G.nodes[curr_path_nodes[2]]['coords']
    #     #     assert not utils.is_edge_free(maze, s1, s2)

    # get bottleneck nodes
    bottleneck_nodes = [curr_path_nodes[0]]
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in bottleneck_nodes:
        if 'p' in node:
            s = helper.state_to_numpy(sparse_G.nodes[node]['coords'])
            points_x.append(s[0])
            points_y.append(s[1])
            ss.append(s)

    print("final bottleneck nodes: {}".format(ss))
    # visualize_nodes(occ_grid, np.array(list(zip(points_x,points_y))), start_pos, goal_pos)
    return goal_pos, ss

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

    # visualize_nodes(occ_grid,np.array(list(zip(points_x,points_y))), start_pos, goal_pos)

    points_x = []
    points_y = []
    print("path_nodes = ", path_nodes, bottleneck_nodes)
    for node in bottleneck_nodes:
        s = helper.state_to_numpy(sparse_G.node[node]['state'])
        points_x.append(s[0])
        points_y.append(s[1])

    # visualize_nodes(occ_grid,np.array(list(zip(points_x,points_y))), start_pos, goal_pos)

if __name__ == '__main__':
    main()