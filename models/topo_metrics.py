import numpy as np
from scipy.spatial import cKDTree
import networkx as nx
import torch

def match_nodes(pred_nodes, gt_nodes, tolerance=0.02):
    tree = cKDTree(gt_nodes)
    matches = {}
    for i, pt in enumerate(pred_nodes):
        dist, j = tree.query(pt)
        if dist <= tolerance:
            matches[i] = j
    return matches

def remap_edges(edges, match_dict):
    new_edges = []
    for u, v in edges:
        if u in match_dict and v in match_dict:
            a, b = match_dict[u], match_dict[v]
            if a != b:
                new_edges.append(tuple(sorted((a, b))))
    return new_edges

def build_graph(nodes, edges):
    G = nx.Graph()
    for i, (x, y) in enumerate(nodes):
        G.add_node(i, pos=(x, y))
    for u, v in edges:
        p1, p2 = nodes[u], nodes[v]
        dist = np.linalg.norm(p1 - p2)
        G.add_edge(u, v, weight=dist)
    return G

def correctness(gt_edges, pred_edges):
    gt_set = set(map(tuple, map(sorted, gt_edges)))
    pred_set = set(map(tuple, map(sorted, pred_edges)))
    return len(gt_set & pred_set) / len(pred_set) if pred_set else 0

def completeness(gt_edges, pred_edges):
    gt_set = set(map(tuple, map(sorted, gt_edges)))
    pred_set = set(map(tuple, map(sorted, pred_edges)))
    return len(gt_set & pred_set) / len(gt_set) if gt_set else 0

def compute_apls(G_gt, G_pred, epsilon=1e-5):
    total_diff = 0
    valid_paths = 0
    for u in G_gt.nodes:
        for v in G_gt.nodes:
            if u >= v: continue
            try:
                d_gt = nx.shortest_path_length(G_gt, u, v, weight='weight')
            except:
                continue
            try:
                d_pred = nx.shortest_path_length(G_pred, u, v, weight='weight')
            except:
                d_pred = float('inf')
            if d_gt == float('inf') and d_pred == float('inf'):
                continue
            total_diff += abs(d_pred - d_gt) / (d_gt + epsilon)
            valid_paths += 1
    return 1 - total_diff / valid_paths if valid_paths > 0 else 0


def build_graph(nodes, edges):
    G = nx.Graph()
    for i, (x, y) in enumerate(nodes):
        G.add_node(i, pos=(x, y))
    for u, v in edges:
        # 加入邊的長度作為 weight（距離）
        p1, p2 = np.array(nodes[u]), np.array(nodes[v])
        dist = np.linalg.norm(p1 - p2)
        G.add_edge(u, v, weight=dist)
    return G

def edge_set(edges):
    return set(tuple(sorted(edge)) for edge in edges)

def correctness(gt_edges, pred_edges):
    pred_set = edge_set(pred_edges)
    gt_set = edge_set(gt_edges)
    return len(pred_set & gt_set) / len(pred_set) if pred_set else 0

def completeness(gt_edges, pred_edges):
    pred_set = edge_set(pred_edges)
    gt_set = edge_set(gt_edges)
    return len(pred_set & gt_set) / len(gt_set) if gt_set else 0

def compute_apls(G_gt, G_pred, epsilon=1e-5):
    """
    G_gt, G_pred: NetworkX graphs with edge weights as distances
    """
    total_diff = 0
    valid_paths = 0

    nodes_gt = list(G_gt.nodes)
    for i in range(len(nodes_gt)):
        for j in range(i + 1, len(nodes_gt)):
            u, v = nodes_gt[i], nodes_gt[j]
            try:
                d_gt = nx.shortest_path_length(G_gt, u, v, weight='weight')
            except nx.NetworkXNoPath:
                continue
            try:
                d_pred = nx.shortest_path_length(G_pred, u, v, weight='weight')
            except nx.NetworkXNoPath:
                d_pred = float('inf')
            # Skip if both are disconnected
            if d_gt == float('inf') and d_pred == float('inf'):
                continue
            ratio = abs(d_pred - d_gt) / (d_gt + epsilon)
            total_diff += ratio
            valid_paths += 1

    if valid_paths == 0:
        return 0
    return 1 - total_diff / valid_paths
