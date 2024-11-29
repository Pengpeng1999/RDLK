import scipy.sparse as ssp
import random
import math
import numpy as np
import tqdm
import torch
import networkx as nx
import time
from TRPCA_torch import TRPCA
import numpy as np
import pandas as pd
from scipy.linalg import svd
from node2vec import Node2Vec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def sample_neg(net, test_ratio, train_pos=None, test_pos=None, max_train_num=None,
               all_unknown_as_negative=False):
    # get upper triangular matrix
    net_triu = ssp.triu(net, k=1)
    # sample positive links for train/test
    row, col, _ = ssp.find(net_triu)
    # sample positive links if not specified
    if train_pos is None and test_pos is None:
        perm = random.sample(range(len(row)), len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    # if max_train_num is set, randomly sample train links
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num] # 打乱索引，随机抽训练边
        train_pos = (train_pos[0][perm], train_pos[1][perm])
        perm = np.random.permutation(len(test_pos[0]))  # 打乱索引，随机抽训练边
        test_pos = (test_pos[0][perm], test_pos[1][perm])
    train_pos = (train_pos[0].tolist(), train_pos[1].tolist())
    test_pos = (test_pos[0].tolist(), test_pos[1].tolist())
    # sample negative links for train/test
    train_num = len(train_pos[0])
    test_num = len(test_pos[0])
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        # sample a portion unknown links as train_negs and test_negs (no overlap)
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        # regard all unknown links as test_negs, sample a portion from them as train_negs
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net == 0, k=1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    return train_pos, train_neg, test_pos, test_neg

def node_label(subgraph):
    # an implementation of the proposed double-radius node labeling (DRNL)
    K = subgraph.shape[0]
    subgraph_wo0 = subgraph[1:, 1:] # 去除0节点的A
    nodeslist = [0]+list(range(2, K))
    subgraph_wo1 = subgraph[nodeslist, :][:, nodeslist] # 去除1节点的A
    dist_to_0 = ssp.csgraph.shortest_path(subgraph_wo0, directed=False, unweighted=True)
    dist_to_0 = dist_to_0[1:, 0]
    dist_to_0[dist_to_0 == np.inf] = K ** 2
    dist_to_1 = ssp.csgraph.shortest_path(subgraph_wo1, directed=False, unweighted=True)
    dist_to_1 = dist_to_1[1:, 0]
    dist_to_1[dist_to_1 == np.inf] = K ** 2
    d = (dist_to_0 + dist_to_1).astype(int)
    d_over_2, d_mod_2 = np.divmod(d, 2)
    labels = 1 + np.minimum(dist_to_0, dist_to_1).astype(int) + d_over_2 * (d_over_2 + d_mod_2 - 1)
    labels = np.concatenate((np.array([1, 1]), labels))
    labels[np.isinf(labels)] = 0
    labels[labels > 1e6] = 0  # set inf labels to 0
    labels[labels < -1e6] = 0  # set -inf labels to 0
    return labels

def neighbors(fringe, A):
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def subgraph_extraction(ind, A, h=1, max_nodes_per_hop=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    while len(nodes) < 20:
    # for dist in range(1, 4):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    return nodes

def construct_denois_subgraph(sub, p):  # 可以是数量也可以是概率0.5
    # 1.数量
    # flattened = sub.flatten()
    # if e_num > len(flattened):
    #     e_num = len(flattened) - sub.shape[0]
    # top_e_num_indices = np.argpartition(flattened, -e_num)[-e_num:]
    # binary_matrix = np.zeros_like(flattened)
    # binary_matrix[top_e_num_indices] = int(1)
    # g = binary_matrix.reshape(sub.shape[0], sub.shape[0])
    # 2.概率
    g = np.where(sub > p, 1, 0)

    g = ssp.csr_matrix(np.maximum(g, g.T))
    sparse_lil = g.tolil()
    for i in range(min(sparse_lil.shape)):
        sparse_lil[i, i] = 0
    g = sparse_lil.tocsr()
    return g

def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None):
    # extract the h-hop enclosing subgraph around link 'ind'
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    dist = 1
    while len(nodes) < 20:
    # for dist in range(1, h+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
        dist = dist + 1
    # move target nodes to top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    # # apply node-labeling
    rpca_subgraph_float, _ = robust_pca(subgraph.A)
    rpca_subgraph = construct_denois_subgraph(rpca_subgraph_float, p = 0.2)
    subgraph = rpca_subgraph
    labels = node_label(rpca_subgraph)
    # # get node features
    # features = None
    # # construct nx graph
    # g = nx.from_scipy_sparse_array(subgraph)
    # # remove link between target nodes
    # if g.has_edge(0, 1):
    #     g.remove_edge(0, 1)
    return labels.tolist(), nodes, subgraph


def helper(G, links, h, max_nodes_per_hop, max_n_label, attributes, attr_sim_matirx,spectral_distances_matrix):
    list_g = []
    ego_fea = []
    avg_fea = []
    sum_fea = []
    for i, j in tqdm.tqdm(tuple(zip(links[0], links[1]))):
        attr_sim_links_list = att_similar_list(i, j, attr_sim_matirx, 3, 2)         # k表示与每个节点相似的topk，p表示与取多少个相似子
        topo_sim_links_list = topo_similar_list(i, j, attr_sim_matirx, 3, 2)
        p_g_attr_fea = []
        p_g_topo_fea = []
        for k, l in attr_sim_links_list:
            nodes = subgraph_extraction((k, l), G, h, max_nodes_per_hop)
            g_attributes = attributes[nodes]
            if torch.cuda.is_available() == True:
                g_attributes = torch.from_numpy(g_attributes).to('cuda')
                trpca = TRPCA()
                L, E = trpca.ADMM(g_attributes)
                L = L.cpu().numpy()
            else:
                L, E = robust_pca(g_attributes)
            p_g_attr_fea.append(L[:2, :].reshape(1, -1)[0].tolist())
            # p_g_attr_fea.append([L[:1, :].reshape(1, -1)[0].tolist()[i] + L[1:2, :].reshape(1, -1)[0].tolist()[i] for i in range(L.shape[1])])
        for k, l in topo_sim_links_list:
            n_labels, nodes, rpca_subgraph = subgraph_extraction_labeling((k, l), G, h, max_nodes_per_hop)
            node_idx = {node: idx for idx, node in enumerate(nodes)}
            # sorted for nodes_label
            top_k_nodes_label = dict(zip(nodes, n_labels))
            top_k_nodes_label = dict(sorted(top_k_nodes_label.items(), key=lambda item: item[1]))
            new_nodes = list(top_k_nodes_label.keys())
            idx1 = new_nodes.index(k)
            idx2 = new_nodes.index(l)
            idx = min(idx1, idx2)
            new_nodes = new_nodes[idx:] + new_nodes[:idx]
            new_nodes = [k, l] + new_nodes[2:]
            if len(new_nodes) >= 20:
                new_nodes = new_nodes[:20]
                new_idx = [node_idx[node] for node in new_nodes]
                g = rpca_subgraph[new_idx, :][:, new_idx]
                g_topo = ssp.csr_matrix(g)
                labels = np.array([top_k_nodes_label[node] for node in new_nodes])
            if len(new_nodes) < 20:
                new_idx = [node_idx[node] for node in new_nodes]
                g = rpca_subgraph[new_idx, :][:, new_idx]
                g_topo_padded = np.pad(g.toarray(), ((0, 20 - len(new_nodes)), (0, 20 - len(new_nodes))), mode='constant', constant_values=0)
                g_topo = ssp.csr_matrix(g_topo_padded)
                labels = np.array([top_k_nodes_label[node] for node in new_nodes] + [0] * (20-len(new_nodes))).reshape(-1, 1)
            g_topo[0, 1] = 0
            g_topo[1, 0] = 0
            g_topo_triu = ssp.triu(g_topo, k=1).toarray()
            # g_topo_triu = np.column_stack((g_topo_triu, labels))
            topo_vec = g_topo_triu[np.triu_indices(g_topo_triu.shape[1], k=1)] # 提取上三角元素
            p_g_topo_fea.append(topo_vec.tolist())
            max_n_label['value'] = max(max(n_labels), max_n_label['value'])
        ego_fea.append(p_g_topo_fea[0])
        avg_fea.append([(1 / 2) * (x + y) for x, y in zip(p_g_topo_fea[0], p_g_topo_fea[1])])
        # avg_fea.append([(1 / 3) * (x + y + z) for x, y, z in zip(p_g_topo_fea[0], p_g_topo_fea[1], p_g_topo_fea[2])])


        # ego_fea.append(p_g_attr_fea[0])
        # avg_fea.append([(1 / 3) * (x + y + z) for x, y, z in zip(p_g_attr_fea[0], p_g_attr_fea[1], p_g_attr_fea[2])])

        # ego_fea.append(p_g_attr_fea[0] + p_g_topo_fea[0])
        # avg_fea.append(p_g_topo_fea[0] +
        #                [(1 / 3) * (x + y + z) for x, y, z in zip(p_g_attr_fea[0], p_g_attr_fea[1], p_g_attr_fea[2])])
        # avg_fea.append([(1 / 2) * (x + y) for x, y in zip(p_g_topo_fea[0], p_g_topo_fea[1])] +
        #                [(1 / 3) * (x + y + z) for x, y, z in zip(p_g_attr_fea[0], p_g_attr_fea[1], p_g_attr_fea[2])])


        sum_fea.append([(x + y + z) for x, y, z in zip(p_g_attr_fea[0], p_g_attr_fea[1], p_g_attr_fea[2])] + topo_vec.tolist())
    return [list_g, ego_fea, avg_fea, sum_fea]  # finally features

def att_similar_list(node1, node2, matrix, k, p):
    targrt_vector1 = matrix[node1]
    targrt_vector2 = matrix[node2]
    s_ij = matrix[node1][node2]
    sim_value1, index1 = torch.topk(targrt_vector1, k+1)
    index1 =index1.cpu().numpy().tolist()
    sim_value2, index2 = torch.topk(targrt_vector2, k+1)
    index2 = index2.cpu().numpy().tolist()
    if node1 in index1:
        index1.remove(node1)
    if node2 in index2:
        index2.remove(node2)
    if node2 in index1:
        index1.remove(node2)
    if node1 in index2:
        index2.remove(node1)
    if index1 == index2:
        '1'
    nodepair_values = {}
    import itertools as it
    nodepair = it.product(index1, index2)
    for l, m in nodepair:
        if l != m:
            nodepair_values[(l, m)] = abs(matrix[l][m]-s_ij)
    sorted_items = sorted(nodepair_values.items(), key=lambda x: x[1])
    topp_keys = [list(item[0]) for item in sorted_items[:p]]
    topp_keys.insert(0, [node1, node2])
    return topp_keys

def topo_similar_list(node1, node2, matrix, k, p):
    targrt_vector1 = matrix[node1]
    targrt_vector2 = matrix[node2]
    d_ij = matrix[node1][node2]
    neg_target_vector1 = -targrt_vector1
    sim_value1, index1 = torch.topk(neg_target_vector1, k+1)
    sim_value1 = -sim_value1
    index1 =index1.cpu().numpy().tolist()
    neg_target_vector2 = -targrt_vector2
    sim_value2, index2 = torch.topk(neg_target_vector2, k+1)
    sim_value2 = -sim_value2
    index2 = index2.cpu().numpy().tolist()
    if node1 in index1:
        index1.remove(node1)
    if node2 in index2:
        index2.remove(node2)
    if node2 in index1:
        index1.remove(node2)
    if node1 in index2:
        index2.remove(node1)
    nodepair_values = {}
    import itertools as it
    nodepair = it.product(index1, index2)
    for l, m in nodepair:
        if l != m:
            nodepair_values[(l, m)] = abs(matrix[l][m]-d_ij)
    sorted_items = sorted(nodepair_values.items(), key=lambda x: x[1])
    topp_keys = [list(item[0]) for item in sorted_items[:p]]
    # topp_keys = [topp_keys[0]]
    topp_keys.insert(0, [node1, node2])
    return topp_keys


def robust_pca(M, lambda_val=1.0):
    """
    Perform Robust PCA using a simplified method.
    :param M: Input matrix to decompose.
    :param lambda_val: Regularization parameter.
    :return: Low-rank matrix L and sparse matrix S.
    """
    try:
        U, S, Vt = svd(M, full_matrices=False)
    except np.linalg.LinAlgError as e:
        M = M.astype(float)
        M[np.isinf(M)] = np.nan
        M = np.nan_to_num(M)
        U, S, Vt = ssp.linalg.svds(ssp.csr_matrix(M))
    # Thresholding for low-rank approximation
    S_thresholded = np.maximum(S - lambda_val, 0)
    # Reconstruct low-rank matrix L
    L = np.dot(U, np.dot(np.diag(S_thresholded), Vt))
    # Sparse matrix S
    S = M - L
    return L, S

def prime(x):
    if x < 2:
        return False
    if x == 2 or x == 3:
        return True
    for i in range(2, x):
        if x % i == 0:
            return False
    return True

def compute_geometric_mean_distance(subgraph, link):
    u = link[0]
    v = link[1]
    if [u, v] in subgraph.edges:
        subgraph.remove_edge(u, v)
    n_nodes = subgraph.number_of_nodes()
    u_reachable = nx.descendants(subgraph, source=u)
    v_reachable = nx.descendants(subgraph, source=v)
    for node in subgraph.nodes:
        distance_to_u = 0
        distance_to_v = 0
        if node != u:
            distance_to_u = nx.shortest_path_length(subgraph, source=node,
                                                    target=u) if node in u_reachable else 2 ** n_nodes
        if node != v:
            distance_to_v = nx.shortest_path_length(subgraph, source=node,
                                                    target=v) if node in v_reachable else 2 ** n_nodes
        subgraph.nodes[node]['avg_dist'] = math.sqrt(distance_to_u * distance_to_v)
    subgraph.add_edge(u, v, distance=0)
    return subgraph

def palette_wl(subgraph, link, prime_numbers):
    tmp_subgraph = subgraph.copy()
    if tmp_subgraph.has_edge(link[0], link[1]):
        tmp_subgraph.remove_edge(link[0], link[1])
    avg_dist = nx.get_node_attributes(tmp_subgraph, 'avg_dist')
    df = pd.DataFrame.from_dict(avg_dist, orient='index', columns=['hash_value'])
    df = df.sort_index()
    df['order'] = df['hash_value'].rank(axis=0, method='min').astype(int)
    df['previous_order'] = np.zeros(df.shape[0], dtype=int)
    adj_matrix = nx.adjacency_matrix(tmp_subgraph, nodelist=sorted(tmp_subgraph.nodes)).todense()
    while any(df.order != df.previous_order):
        df['log_prime'] = np.log(prime_numbers[df['order'].values])
        total_log_primes = np.ceil(np.sum(df.log_prime.values))
        df['hash_value'] = adj_matrix * df.log_prime.values.reshape(-1, 1) / total_log_primes + df.order.values.reshape(
            -1, 1)
        df.previous_order = df.order
        df.order = df.hash_value.rank(axis=0, method='min').astype(int)
    nodelist = df.order.sort_values().index.values
    return nodelist

def special_vec(G):
    special_res = []
    vec_dim = 0
    print('Calculate the topological vector of the subgraph corresponding to each node')
    for node in tqdm.tqdm(range(G.shape[0])):
        nodes = neighbors({node}, G)
        nodes = nodes.union({node})
        num = 0
        while len(nodes) < 20:
            nodes = neighbors(nodes, G)
            num = num + 1
            if num > 4:
                break
        nodes = list(nodes)
        vec_dim = max(len(nodes), vec_dim)
        sub_adj = G[nodes, :][:, nodes]
        sub_adj = sub_adj.A
        D = np.diag(np.sum(sub_adj, axis=1))
        D_inv_sqrt = np.diag(1 / np.sqrt(np.diag(D)))
        L_sym = np.eye(sub_adj.shape[0]) - D_inv_sqrt @ sub_adj @ D_inv_sqrt
        eigvals = np.linalg.eigvals(L_sym)
        sorted_eigvals = np.sort(eigvals)[::-1]
        special_res.append(sorted_eigvals.tolist())
    special_res = [node + [0] * (vec_dim - len(node)) for node in special_res]
    print('Finish the topological vector of the subgraph corresponding to each node')
    return np.array(special_res)

def generate_node2vec_embeddings(G, dim):
    node2vec = Node2Vec(G, dimensions=dim, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    node_embeddings = {node: model.wv[str(node)].tolist() for node in G.nodes()}
    return node_embeddings