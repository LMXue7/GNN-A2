from typing import List, Tuple, Dict

import torch
import torch_geometric
from scipy.sparse import csr_matrix
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import to_undirected
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import numpy as np
import pickle
import pandas as pd
import os.path as osp
import itertools
import os
from icecream import ic


class Dataset(InMemoryDataset):
    def __init__(self, root, dataset, rating_file, sep, args, pred_edges=1, transform=None, pre_transform=None):

        self.path = root
        self.dataset = dataset
        self.rating_file = rating_file
        self.sep = sep
        self.store_backup = True
        self.args = args
        self.pred_edges = pred_edges

        super(Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.stat_info = torch.load(self.processed_paths[1])
        self.data_num = self.stat_info['data_num']
        self.feature_num = self.stat_info['feature_num']

    @property
    def raw_file_names(self):
        return ['{}{}/user_dict.pkl'.format(self.path, self.dataset),
                '{}{}/item_dict.pkl'.format(self.path, self.dataset),
                '{}{}/feature_dict.pkl'.format(self.path, self.dataset),
                '{}{}/{}'.format(self.path, self.dataset, self.rating_file), \
                '{}{}/{}.edge'.format(self.path, self.dataset, self.dataset)]

    @property
    def processed_file_names(self):
        if not self.pred_edges:
            return ['{}_edge/{}.dataset'.format(self.dataset, self.dataset), \
                    '{}_edge/{}.statinfo'.format(self.dataset, self.dataset)]

        else:
            return ['{}/{}.dataset'.format(self.dataset, self.dataset),
                    '{}/{}.statinfo'.format(self.dataset, self.dataset)]

    def download(self):
        # Download to `self.raw_dir`.
        pass


    def data_2_graphs(self, ratings_df, dataset='train'):
        graphs = []
        CenterUsers, CenterItems = [], []
        AttUsers, AttItems = [], []
        n = 0
        m = 0
        processed_graphs = 0
        error_num = 0
        num_graphs = ratings_df.shape[0]
        one_per = int(num_graphs/1000)
        percent = 0.0
        for i in range(len(ratings_df)):
            if processed_graphs % one_per == 0:
                print(f"Processing [{dataset}]: {percent/10.0}%, {processed_graphs}/{num_graphs}", end="\r")
                percent += 1
            processed_graphs += 1 
            line = ratings_df.iloc[i]
            user_index = self.user_key_type(line[0])
            item_index = self.item_key_type(line[1])
            rating = int(line[2])

            if item_index not in self.item_dict or user_index not in self.user_dict:
                error_num += 1
                continue

            user_id = self.user_dict[user_index]['name']
            item_id = self.item_dict[item_index]['title']

            user_attr_list = self.user_dict[user_index]['attribute']
            item_attr_list = self.item_dict[item_index]['attribute']

            user_list = [user_id] + user_attr_list
            item_list = [item_id] + item_attr_list
            n += len(user_list)
            m += len(item_list)

            graph, user_sr, item_sr, num_item_sr = self.construct_graphs(user_list, item_list, rating)
            user_sr_list = torch.LongTensor(user_sr)
            item_sr_list = torch.LongTensor(item_sr)

            setattr(graph, 'user_sr', user_sr_list)
            setattr(graph, 'item_sr', item_sr_list)
            setattr(graph, 'num_item_sr', num_item_sr)

            graphs.append(graph)

        print()

        return graphs



    def read_data(self):
        self.user_dict = pickle.load(open(self.userfile, 'rb'))
        self.item_dict = pickle.load(open(self.itemfile, 'rb'))
        self.user_key_type = type(list(self.user_dict.keys())[0])
        self.item_key_type = type(list(self.item_dict.keys())[0])
        feature_dict = pickle.load(open(self.featurefile, 'rb'))

        data = []
        error_num = 0

        ratings_df = pd.read_csv(self.ratingfile, sep=self.sep, header=None)
        train_df, test_df = train_test_split(ratings_df, test_size=0.4, random_state=self.args.random_seed, stratify=ratings_df[[0,2]])
        test_df, valid_df = train_test_split(test_df,  test_size=0.5, random_state=self.args.random_seed, stratify=test_df[[0,2]])

        # store a backup of train/valid/test dataframe
        if self.store_backup:
            backup_path = f"{self.path}{self.dataset}/split_data_backup/"
            if not os.path.exists(backup_path):
                os.mkdir(backup_path)

            train_df.to_csv(f'{backup_path}train_data.csv', index=False)
            valid_df.to_csv(f'{backup_path}valid_data.csv', index=False)
            test_df.to_csv(f'{backup_path}test_data.csv', index=False)

        print('(Only run at the first time training the dataset)')
        train_graphs = self.data_2_graphs(train_df, dataset='train')
        valid_graphs = self.data_2_graphs(valid_df, dataset='valid')
        test_graphs = self.data_2_graphs(test_df, dataset='test')

        graphs = train_graphs + valid_graphs + test_graphs

        stat_info = {}
        stat_info['data_num'] = len(graphs)
        stat_info['feature_num'] = len(feature_dict)
        stat_info['train_test_split_index'] = [len(train_graphs), len(train_graphs) + len(valid_graphs)]

        print('error number of data:', error_num)
        return graphs, stat_info


    def construct_graphs(self, user_list, item_list, rating):

        u_n = len(user_list)   # user node number
        i_n = len(item_list)   # item node number
        node_list = user_list + item_list
        # 创建索引映射
        user_index_map = {user: idx for idx, user in enumerate(user_list)}
        item_index_map = {item: idx for idx, item in enumerate(item_list)}

        out_CenterUser, out_CenterItem = [], []
        out_AttUser, out_AttItem = [], []
        # construct full inner edge
        inner_edge_index = [[], []]
        user_sender_receiver_list = []
        item_sender_receiver_list = []
        for i in range(u_n):
            out_CenterUser.extend([user_index_map[user_list[i]]] * i_n)
            out_AttUser.extend([item_index_map[item] for item in item_list])
            for j in range(i, u_n):
                inner_edge_index[0].append(i)
                inner_edge_index[1].append(j)
                #user_sender_receiver_list.append([node_list[i], node_list[j]])

        for i in range(u_n, u_n + i_n):
            out_CenterItem.extend([item_index_map[item_list[i - u_n]]] * u_n)
            out_AttItem.extend([user_index_map[user] for user in user_list])
            for j in range(i, u_n + i_n):
                inner_edge_index[0].append(i)
                inner_edge_index[1].append(j)
                #item_sender_receiver_list.append([node_list[i], node_list[j]])



        # construct outer edge
        outer_edge_index = [[], []]
        for i in range(u_n):
            for j in range(i_n):
                outer_edge_index[0].append(i)
                outer_edge_index[1].append(u_n + j)
                user_sender_receiver_list.append([node_list[i], node_list[u_n + j]])
        for i in range(i_n):
            for j in range(u_n):
                item_sender_receiver_list.append([node_list[u_n + i], node_list[j]])
        num_item_sr = len(item_sender_receiver_list)

        #construct graph
        inner_edge_index = torch.LongTensor(inner_edge_index)
        inner_edge_index = to_undirected(inner_edge_index)
        outer_edge_index = torch.LongTensor(outer_edge_index)
        outer_edge_index = to_undirected(outer_edge_index)

        out_CenterUser = np.array(out_CenterUser)
        out_CenterItem = np.array(out_CenterItem)
        out_AttUser = np.array(out_AttUser)
        out_AttItem = np.array(out_AttItem)

        out_UserAttNet = csr_matrix((np.ones(len(out_CenterUser)), (out_CenterUser, out_AttUser)), shape=(u_n, i_n))
        out_ItemAttNet = csr_matrix((np.ones(len(out_CenterItem)), (out_CenterItem, out_AttItem)), shape=(i_n, u_n))

        graph = self.construct_graph(node_list, inner_edge_index, outer_edge_index, rating, out_UserAttNet, out_ItemAttNet)

        return graph, user_sender_receiver_list, item_sender_receiver_list, num_item_sr

    def construct_graph(self, node_list, edge_index_inner, edge_index_outer, rating, out_UserAttNet, out_ItemAttNet):
        x = torch.LongTensor(node_list).unsqueeze(1)
        rating = torch.FloatTensor([rating])

        # 将稀疏矩阵转换为COO格式以便存储
        out_user_attnet_coo = out_UserAttNet.tocoo()
        out_item_attnet_coo = out_ItemAttNet.tocoo()

        out_user_attnet_indices = np.vstack((out_user_attnet_coo.row, out_user_attnet_coo.col))
        out_user_attnet_values = out_user_attnet_coo.data
        out_item_attnet_indices = np.vstack((out_item_attnet_coo.row, out_item_attnet_coo.col))
        out_item_attnet_values = out_item_attnet_coo.data

        out_user_attnet_indices = torch.tensor(out_user_attnet_indices, dtype=torch.long)
        out_item_attnet_indices = torch.tensor(out_item_attnet_indices, dtype=torch.long)

        return Data(x=x, edge_index=edge_index_inner, edge_attr=torch.transpose(edge_index_outer, 0, 1), y=rating,
                    Out_UserAttNet_indices=torch.transpose(out_user_attnet_indices, 0, 1),
                    Out_UserAttNet_values=torch.tensor(out_user_attnet_values, dtype=torch.float32),
                    Out_ItemAttNet_indices=torch.transpose(out_item_attnet_indices, 0, 1),
                    Out_ItemAttNet_values=torch.tensor(out_item_attnet_values, dtype=torch.float32))

    def process(self):
        self.userfile  = self.raw_file_names[0]
        self.itemfile  = self.raw_file_names[1]
        self.featurefile = self.raw_file_names[2]
        self.ratingfile  = self.raw_file_names[3]
        graphs, stat_info = self.read_data()
        #check whether foler path exist
        if not os.path.exists(f"{self.path}processed/{self.dataset}"):
            os.mkdir(f"{self.path}processed/{self.dataset}")

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])

        torch.save(stat_info, self.processed_paths[1])

    def feature_N(self):
        return self.feature_num

    def data_N(self):
        return self.data_num

