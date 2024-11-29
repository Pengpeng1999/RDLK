import torch
import pickle
import time
import argparse
import os
import random
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import sample_neg, helper, generate_node2vec_embeddings, special_vec
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import scipy.sparse as ssp
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import networkx as nx

train_rio = [round(0.1 + 0.1 * i, 1) for i in range(8)]
for radio in train_rio:
    print('训练比率:', radio)
    parser = argparse.ArgumentParser(description='Link Prediction with RDLK')
    # general settings
    parser.add_argument('--data-name', default='2.cornell', help='network name')
    parser.add_argument('--train-name', default=None, help='train name')
    parser.add_argument('--test-name', default=None, help='test name')
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--max-train-num', type=int, default=20000,
                        help='set maximum number of train links (to fit into memory)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--test-ratio', type=float, default=1-radio,
                        help='ratio of test links')
    parser.add_argument('--no-parallel', action='store_true', default=True,
                        help='if True, use single thread for subgraph extraction; \
                        by default use all cpu cores to extract subgraphs in parallel')
    parser.add_argument('--all-unknown-as-negative', action='store_true', default=False,
                        help='if True, regard all unknown links as negative test data; \
                        sample a portion from them as negative training data. Otherwise,\
                        train negative and test negative data are both sampled from \
                        unknown links without overlap.')
    # model settings
    parser.add_argument('--hop', default=1, metavar='S',
                        help='enclosing subgraph hop number, \
                        options: 1, 2,...')
    parser.add_argument('--max-nodes-per-hop', default=500,
                        help='if > 0, upper bound the # nodes per hop by subsampling')
    parser.add_argument('--use-embedding', action='store_true', default=True,
                        help='whether to use node2vec node embeddings')
    parser.add_argument('--use-similar', action='store_true', default=True,
                        help='whether to use node2vec node embeddings')
    parser.add_argument('--use-simlar-methods', action='store_true', default=True,
                        help='whether to use CN and Jaccard Adamic-Adar')
    parser.add_argument('--use-attribute', action='store_true', default=True,
                        help='whether to use node attributes')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save the final model')
    parser.add_argument('--init-attri', action='store_true', default=False,
                        help='no rpca exam')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.max_nodes_per_hop is not None:
        args.max_nodes_per_hop = int(args.max_nodes_per_hop)
    print(args.data_name)
    '''Prepare data'''
    # build observed network
    args.file_dir = os.path.dirname(os.path.realpath('__file__'))
    args.data_dir = os.path.join(args.file_dir, 'data/{}.pkl'.format(args.data_name))
    with open(args.data_dir, 'rb') as file:
        data = pickle.load(file)
    net = data['topo']
    net.setdiag(0) # remove self loops
    net.eliminate_zeros()
    if 'attr' in data:
        attributes = data['attr'].toarray().astype('float32')
    if attributes.shape[1] > 500:
        print('执行PCA')
        pca = PCA(n_components=100)
        attributes = pca.fit_transform(attributes)

    # sample both positive and negative train/test links from net
    if args.train_name is None and args.test_name is None:
        train_pos, train_neg, test_pos, test_neg = sample_neg(
                net, args.test_ratio, max_train_num=args.max_train_num
            )
    else:
        # use provided train/test positive links, sample negative from net
        train_pos, train_neg, test_pos, test_neg = sample_neg(
            net,
            train_pos=train_pos,
            test_pos=test_pos,
            max_train_num=args.max_train_num,
            all_unknown_as_negative=args.all_unknown_as_negative
        )
    # 构建节点对拓扑特征，属性特征，Embedding特征
    # constructing the observed network
    A = net.copy()
    A[test_pos[0], test_pos[1]] = 0  # mask test links
    A[test_pos[1], test_pos[0]] = 0  # mask test links
    if np.array_equal(A.A ,A.A.T) == False:
        A = A.maximum(A.T)
        A.setdiag(0)
    A.eliminate_zeros()  # make sure the links are masked when using the sparse matrix in scipy-1.3.x
    # construct nx graph
    train_net = nx.from_scipy_sparse_array(A)

    # 使用初始特征实验
    if args.init_attri:
        train_pos_init_fea = []
        train_neg_init_fea = []
        test_pos_init_fea = []
        test_neg_init_fea = []
        for i in range(len(train_pos[0])):
            train_pos_init_fea.append((attributes[train_pos[0][i], :] + attributes[train_pos[1][i], :]).tolist())
        for i in range(len(train_neg[0])):
            train_neg_init_fea.append((attributes[train_neg[0][i], :] + attributes[train_neg[1][i], :]).tolist())
        for i in range(len(test_pos[0])):
            test_pos_init_fea.append((attributes[test_pos[0][i], :] + attributes[test_pos[1][i], :]).tolist())
        for i in range(len(test_neg[0])):
            test_neg_init_fea.append((attributes[test_neg[0][i], :] + attributes[test_neg[1][i], :]).tolist())
        X_train_init_fea = np.vstack((np.array(train_pos_init_fea), np.array(train_neg_init_fea)))
        y_train = np.array([1 for i in range(len(train_pos_init_fea))] + [0 for i in range(len(train_neg_init_fea))])
        X_test_init_fea = np.vstack((np.array(test_pos_init_fea), np.array(test_neg_init_fea)))
        y_test = np.array([1 for i in range(len(test_pos_init_fea))] + [0 for i in range(len(test_neg_init_fea))])

    # embeddings methods
    if args.use_embedding:
        # embeddings = generate_node2vec_embeddings(train_net, 64)
        # embeddings = np.array([embeddings[i] for i in range(attributes.shape[0])])
        # np.save(r'F:\实验\文章中的代码\WLRPCA\SEALRPCA\n2v\\' + args.data_name + '.npy', embeddings)
        embeddings = np.load(r'F:\实验\文章中的代码\WLRPCA\SEALRPCA\n2v\\' + args.data_name + '.npy')
        pca = PCA(n_components=3)
        embeddings = pca.fit_transform(embeddings)
        train_pos_embedding = []
        train_neg_embedding = []
        test_pos_embedding = []
        test_neg_embedding = []
        for i in range(len(train_pos[0])):
            train_pos_embedding.append((embeddings[train_pos[0][i]] + embeddings[train_pos[1][i]]))
        for i in range(len(train_neg[0])):
            train_neg_embedding.append((embeddings[train_neg[0][i]] + embeddings[train_neg[1][i]]))
        for i in range(len(test_pos[0])):
            test_pos_embedding.append((embeddings[test_pos[0][i]] + embeddings[test_pos[1][i]]))
        for i in range(len(test_neg[0])):
            test_neg_embedding.append((embeddings[test_neg[0][i]] + embeddings[test_neg[1][i]]))
        X_train_embedding_fea = np.vstack((np.array(train_pos_embedding), np.array(train_neg_embedding)))
        X_test_embedding_fea = np.vstack((np.array(test_pos_embedding), np.array(test_neg_embedding)))

    # similar methods
    if args.use_similar:
        train_pos_cnjaaa = []
        train_neg_cnjaaa = []
        test_pos_cnjaaa = []
        test_neg_cnjaaa = []
        cnjaaa = [[], [], []]
        cn = nx.common_neighbor_centrality(train_net, [(train_pos[0][i], train_pos[1][i]) for i in range(len(train_pos[0]))], alpha=1)
        ja = nx.jaccard_coefficient(train_net, [(train_pos[0][i], train_pos[1][i]) for i in range(len(train_pos[0]))])
        aa = nx.adamic_adar_index(train_net, [(train_pos[0][i], train_pos[1][i]) for i in range(len(train_pos[0]))])
        [cnjaaa[0].append(p) for _, __, p in cn]
        [cnjaaa[1].append(p) for _, __, p in ja]
        [cnjaaa[2].append(p) for _, __, p in aa]
        train_pos_cnjaaa = cnjaaa

        cnjaaa = [[], [], []]
        cn = nx.common_neighbor_centrality(train_net, [(train_neg[0][i], train_neg[1][i]) for i in range(len(train_neg[0]))], alpha=1)
        ja = nx.jaccard_coefficient(train_net, [(train_neg[0][i], train_neg[1][i]) for i in range(len(train_neg[0]))])
        aa = nx.adamic_adar_index(train_net, [(train_neg[0][i], train_neg[1][i]) for i in range(len(train_neg[0]))])
        [cnjaaa[0].append(p) for _, __, p in cn]
        [cnjaaa[1].append(p) for _, __, p in ja]
        [cnjaaa[2].append(p) for _, __, p in aa]
        train_neg_cnjaaa = cnjaaa

        cnjaaa = [[], [], []]
        cn = nx.common_neighbor_centrality(train_net, [(test_pos[0][i], test_pos[1][i]) for i in range(len(test_pos[0]))], alpha=1)
        ja = nx.jaccard_coefficient(train_net, [(test_pos[0][i], test_pos[1][i]) for i in range(len(test_pos[0]))])
        aa = nx.adamic_adar_index(train_net, [(test_pos[0][i], test_pos[1][i]) for i in range(len(test_pos[0]))])
        [cnjaaa[0].append(p) for _, __, p in cn]
        [cnjaaa[1].append(p) for _, __, p in ja]
        [cnjaaa[2].append(p) for _, __, p in aa]
        test_pos_cnjaaa = cnjaaa

        cnjaaa = [[],[],[]]
        cn = nx.common_neighbor_centrality(train_net, [(test_neg[0][i], test_neg[1][i]) for i in range(len(test_neg[0]))], alpha=1)
        ja = nx.jaccard_coefficient(train_net, [(test_neg[0][i], test_neg[1][i]) for i in range(len(test_neg[0]))])
        aa = nx.adamic_adar_index(train_net, [(test_neg[0][i], test_neg[1][i]) for i in range(len(test_neg[0]))])
        [cnjaaa[0].append(p) for _, __, p in cn]
        [cnjaaa[1].append(p) for _, __, p in ja]
        [cnjaaa[2].append(p) for _, __, p in aa]
        test_neg_cnjaaa = cnjaaa

        # X_train_cnjaaa_fea = np.vstack((np.array(train_pos_cnjaaa + train_neg_cnjaaa).T))
        # X_test_cnjaaa_fea = np.vstack((np.array(test_pos_cnjaaa + test_neg_cnjaaa).T))
        X_train_cnjaaa_fea = np.vstack((np.array(train_pos_cnjaaa).T, np.array(train_neg_cnjaaa).T))
        X_test_cnjaaa_fea = np.vstack((np.array(test_pos_cnjaaa).T, np.array(test_neg_cnjaaa).T))

        # model = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs', random_state=40)
        # model = MLPClassifier(hidden_layer_sizes=(32, 32, 2), activation='relu', max_iter=500, random_state=40)
        # model.fit(X_train_init_fea, y_train)
        # y_pred = model.predict(X_test_init_fea)
        # print("similar methods:", roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred),
        #       precision_score(y_test, y_pred), recall_score(y_test, y_pred))

    # extract enclosing subgraphs
    print('Enclosing subgraph extraction begins...')
    max_n_label = {'value': 0} # using for one hot label
    start = time.time()
    train_graphs, test_graphs = None, None
    attributes = torch.from_numpy(attributes).float().to('cuda' if args.cuda else 'cpu')
    attr_sim_matrix = cosine_similarity(attributes)
    # attr_sim_matrix = torch.from_numpy(attr_sim_matrix).to(args.cuda).float()
    attr_sim_matrix = torch.from_numpy(attr_sim_matrix).float().to('cuda' if args.cuda else 'cpu')

    node_num = A.A.shape[0]
    # Calculate the topological vector of the subgraph corresponding to each node
    nodes_special = np.real(special_vec(A))
    nodes_special_ssp = ssp.csr_matrix(nodes_special)
    distance_vector = pdist(nodes_special_ssp.toarray(), metric='euclidean')
    distance_matrix = squareform(distance_vector)
    spectral_distances_matrix = torch.from_numpy(distance_matrix).float().to('cuda' if args.cuda else 'cpu')

    train_pos_subgraphs, train_pos_ego_fea, train_pos_avg_fea, train_pos_sum_fea = helper(A,
                                                    train_pos, args.hop, args.max_nodes_per_hop, max_n_label, attributes, attr_sim_matrix,spectral_distances_matrix)
    train_neg_subgraphs, train_neg_ego_fea, train_neg_avg_fea, train_neg_sum_fea = helper(A,
                                                    train_neg, args.hop, args.max_nodes_per_hop, max_n_label, attributes, attr_sim_matrix,spectral_distances_matrix)
    test_pos_subgraphs, test_pos_ego_fea, test_pos_avg_fea, test_pos_sum_fea = helper(A,
                                                    test_pos, args.hop, args.max_nodes_per_hop, max_n_label, attributes, attr_sim_matrix,spectral_distances_matrix)
    test_neg_subgraphs, test_neg_ego_fea, test_neg_avg_fea, test_neg_sum_fea = helper(A,
                                                    test_neg, args.hop, args.max_nodes_per_hop, max_n_label, attributes, attr_sim_matrix,spectral_distances_matrix)
    end = time.time()
    print("Time eplased for subgraph extraction: {}s".format(end - start))



    X_train = np.vstack((np.array(train_pos_ego_fea), np.array(train_neg_ego_fea)))
    # X_train = np.hstack((X_train, X_train_embedding_fea))
    X_train = np.hstack((X_train, X_train_cnjaaa_fea))
    # X_train = np.hstack((X_train, X_train_init_fea))
    # X_train = np.hstack((X_train, X_train_cnjaaa_fea, X_train_embedding_fea))
    # X_train = np.hstack((X_train, X_train_cnjaaa_fea, X_train_embedding_fea, X_train_init_fea))
    y_train = np.array([1 for i in range(len(train_pos_ego_fea))] + [0 for i in range(len(train_neg_ego_fea))])

    X_test = np.vstack((np.array(test_pos_ego_fea), np.array(test_neg_ego_fea)))
    # X_test = np.hstack((X_test, X_test_embedding_fea))
    X_test = np.hstack((X_test, X_test_cnjaaa_fea))
    # X_test = np.hstack((X_test, X_test_init_fea))
    # X_test = np.hstack((X_test, X_test_cnjaaa_fea, X_test_embedding_fea))
    # X_test = np.hstack((X_test, X_test_cnjaaa_fea, X_test_embedding_fea, X_train_init_fea))
    y_test = np.array([1 for i in range(len(test_pos_ego_fea))] + [0 for i in range(len(test_neg_ego_fea))])

    model = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs', random_state=40)
    # model = MLPClassifier(hidden_layer_sizes=(64, 32, 32, 2), activation='relu', max_iter=500, random_state=40)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred))


    X_train = np.vstack((np.array(train_pos_avg_fea), np.array(train_neg_avg_fea)))
    # X_train = np.hstack((X_train, X_train_embedding_fea))
    X_train = np.hstack((X_train, X_train_cnjaaa_fea))
    # X_train = np.hstack((X_train, X_train_init_fea))
    # X_train = np.hstack((X_train, X_train_cnjaaa_fea, X_train_embedding_fea))
    # X_train = np.hstack((X_train, X_train_cnjaaa_fea, X_train_embedding_fea, X_train_init_fea))
    y_train = np.array([1 for i in range(len(train_pos_avg_fea))] + [0 for i in range(len(train_neg_avg_fea))])

    X_test = np.vstack((np.array(test_pos_avg_fea), np.array(test_neg_avg_fea)))
    # X_test = np.hstack((X_test, X_test_embedding_fea))
    X_test = np.hstack((X_test, X_test_cnjaaa_fea))
    # X_test = np.hstack((X_test, X_test_init_fea))
    # X_test = np.hstack((X_test, X_test_cnjaaa_fea, X_test_embedding_fea))
    # X_test = np.hstack((X_test, X_test_cnjaaa_fea, X_test_embedding_fea, X_train_init_fea))
    y_test = np.array([1 for i in range(len(test_pos_avg_fea))] + [0 for i in range(len(test_neg_avg_fea))])

    model = LogisticRegression(max_iter=500, multi_class='multinomial', solver='lbfgs', random_state=40)
    # model = MLPClassifier(hidden_layer_sizes=(64, 32, 32, 2), activation='relu', max_iter=500, random_state=40)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(roc_auc_score(y_test, y_pred), accuracy_score(y_test, y_pred), f1_score(y_test, y_pred), precision_score(y_test, y_pred), recall_score(y_test, y_pred))
