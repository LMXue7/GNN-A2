import torch
from torch import Tensor
import torch.nn as nn 
from torch.nn import Parameter 
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Linear
import numpy as np
import time
from icecream import ic
from torch_sparse import SparseTensor

from BLS import BLS


class inner_GNN(MessagePassing):
    def __init__(self, hidden_layer):
        super(inner_GNN, self).__init__(aggr='add')

        #construct pairwise modeling network
        self.hidden_layer = hidden_layer
        self.lin1 = nn.Linear(1, hidden_layer)
        self.lin2 = nn.Linear(hidden_layer, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.S = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight=None):
        # x has shape [N, dim]
        x = x.squeeze()
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        # x_i has shape [E, dim]
        # x_j has shape [E, dim]

        p = x_i * x_j
        pairwise_analysis = self.gap(p)

        pairwise_analysis = self.lin1(pairwise_analysis)
        pairwise_analysis = self.act(pairwise_analysis)
        pairwise_analysis = self.lin2(pairwise_analysis)

        pairwise_analysis = self.S(pairwise_analysis)
        interaction_analysis = pairwise_analysis * p
        interaction_analysis = interaction_analysis + p

        return interaction_analysis

    def update(self, aggr_out):
        # aggr_out has shape [N, dim]
        return aggr_out


class getAttEmb(nn.Module):
    def __init__(self, dim):
        super(getAttEmb, self).__init__()
        self.dim = dim
        # 延迟初始化节点嵌入
        self.user_embedding = None
        self.item_embedding = None
        self.num_initialized = False

    def forward(self, num_users, num_items, Out_UserAttNets, Out_ItemAttNets):

        # 延迟初始化物品嵌入
        if not self.num_initialized:
            self.initialize_embedding(num_users, num_items)
        users_emb = self.user_embedding.weight.cuda()  # 节点权重
        items_emb = self.item_embedding.weight.cuda()
        out_useratt_emb = self.getFusionAttribute(users_emb, items_emb, Out_UserAttNets)
        out_itematt_emb = self.getFusionAttribute(items_emb, users_emb, Out_ItemAttNets)
       
        return users_emb, items_emb, out_useratt_emb, out_itematt_emb

    def initialize_embedding(self, num_users, num_items):
        if not self.num_initialized:
            self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=self.dim)
            nn.init.normal_(self.user_embedding.weight, std=0.1)
            self.item_embedding = nn.Embedding(num_embeddings=num_items, embedding_dim=self.dim)
            nn.init.normal_(self.item_embedding.weight, std=0.1)

    def getFusionAttribute(self, embedding, att_embedding, adj_mat):
        att_attention = self.getAttention(embedding, att_embedding, adj_mat)
        f_att_emb = torch.mm(att_attention, att_embedding)
        return f_att_emb

    def getAttention(self, embedding, att_embedding, adj_mat):
        sim_matrix = self.cosine_distance(embedding, att_embedding)
        sim_matrix = torch.mul(sim_matrix, adj_mat)
        attention = self.normalization(sim_matrix)
        return attention

    def normalization(self, matrix):
        zero_vec = -9e15 * torch.ones_like(matrix)
        attention = torch.where(matrix != 0, matrix, zero_vec)
        softmax = torch.nn.Softmax(dim=1)
        return softmax(attention)

    def cosine_distance(self, matrix1, matrix2):
        matrix1 = F.normalize(matrix1, p=2, dim=1)
        matrix2 = F.normalize(matrix2, p=2, dim=1)
        return torch.mm(matrix1, matrix2.t())


class GNN_AA(nn.Module):
    """
    GNN-AA main model
    """
    def __init__(self, args, n_features, device):
        super(GNN_AA, self).__init__()

        self.n_features = n_features
        self.dim = args.dim
        self.hidden_layer = args.hidden_layer
        self.device = device
        self.batch_size = args.batch_size
        self.num_user_features = args.num_user_features
        self.l0_para = eval(args.l0_para)

        self.feature_embedding = nn.Embedding(self.n_features + 1, self.dim)
        #self.feature_embedding.weight.data.normal_(0.0,0.01)
        self.node_weight = nn.Embedding(self.n_features + 1, 1)
        self.node_weight.weight.data.normal_(0.0, 0.01)

        self.inner_gnn = inner_GNN(self.hidden_layer)
        self.getAttEmb = getAttEmb(self.dim)
        self.update_f = nn.GRU(input_size=self.dim, hidden_size=self.dim, dropout=0.5)
        self.g = nn.Linear(self.dim, 1, bias=False)
        self.BLS = BLS(self.dim)
        self.lin1 = nn.Linear(1, self.dim)

    def forward(self, data, is_training=True):
        # does not conduct link prediction, use all interactions

        node_id = data.x.to(self.device)
        batch = data.batch
        inner_edge_index = data.edge_index
        user_sr = data.user_sr
        item_sr = data.item_sr
        #记录每张图中item发送接收列表的长度
        num_item_sr = data.num_item_sr

        out_user_attnet_indices = torch.transpose(data.Out_UserAttNet_indices, 0, 1)
        out_user_attnet_values = data.Out_UserAttNet_values
        out_item_attnet_indices = torch.transpose(data.Out_ItemAttNet_indices, 0, 1)
        out_item_attnet_values = data.Out_ItemAttNet_values

        # 将COO格式转换为SparseTensor
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        num_users = self.num_user_features * len(nodes_per_graph)
        num_items = len(node_id) - num_users
        Out_UserAttNets = SparseTensor(row=out_user_attnet_indices[0], col=out_user_attnet_indices[1], value=out_user_attnet_values,
                                   sparse_sizes=(num_users, num_items)).to(device='cuda')
        Out_ItemAttNets = SparseTensor(row=out_item_attnet_indices[0], col=out_item_attnet_indices[1], value=out_item_attnet_values,
                                   sparse_sizes=(num_items, num_users)).to(device='cuda')
        Out_UserAtt_nets = Out_UserAttNets.to_dense().cuda()
        Out_ItemAtt_nets = Out_ItemAttNets.to_dense().cuda()

        # handle pointwise features
        node_w = torch.squeeze(self.node_weight(node_id))
        sum_weight = global_add_pool(node_w, batch)

        node_emb = self.feature_embedding(node_id)
        l0_penalty = 0

        inner_node_message = self.inner_gnn(node_emb, inner_edge_index)
        users_emb, items_emb, out_useratt_emb, out_itematt_emb = self.getAttEmb(
            num_users, num_items, Out_UserAtt_nets, Out_ItemAtt_nets)
        outer_node_message = self.connect(batch, self.num_user_features, users_emb, items_emb,
                                           out_useratt_emb, out_itematt_emb)

        new_batch = self.split_batch(batch, self.num_user_features)

        batch_size = len(sum_weight)
        # 根据 new_batch 中的标记直接分割用户节点和项目节点
        user_mask = (new_batch >= batch_size) & (new_batch <= batch_size * 2 - 1)
        item_mask = (new_batch >= 0) & (new_batch <= batch_size - 1)

        inner_user_node = inner_node_message[user_mask]
        inner_item_node = inner_node_message[item_mask]

        outer_user_node = outer_node_message[user_mask]
        outer_item_node = outer_node_message[item_mask]

        node_user = node_emb[user_mask]
        node_item = node_emb[item_mask]

        # Reshape user nodes
        re_inner_user_node = inner_user_node.view(batch_size, -1, self.dim)
        re_outer_user_node = outer_user_node.view(batch_size, -1, self.dim)

        # Zero padding for item nodes
        num_item_features_list = [(new_batch == i).sum().item() for i in range(batch_size)]
        max_num_item = max(num_item_features_list)
        inner_item_node_padded = torch.zeros((batch_size, max_num_item, self.dim)).to(self.device)
        outer_item_node_padded = torch.zeros((batch_size, max_num_item, self.dim)).to(self.device)
        item_masks = torch.ones((batch_size, max_num_item, self.dim)).to(self.device)
        current_index = 0
        for i in range(batch_size):
            num_item_nodes = (new_batch == i).sum().item()
            inner_item_node_padded[i, :num_item_nodes] = inner_item_node[
                                                           current_index:current_index + num_item_nodes]
            outer_item_node_padded[i, :num_item_nodes] = outer_item_node[
                                                            current_index:current_index + num_item_nodes]
            item_masks[i, num_item_nodes:] = 0  # 设置掩码
            current_index += num_item_nodes

        # Element-wise sum and BLS processing
        sum_user_features = re_inner_user_node + re_outer_user_node
        sum_item_features = inner_item_node_padded + outer_item_node_padded

        out_user = self.BLS.fit(sum_user_features, "user", torch.ones_like(sum_user_features))  # (batch_size*n_node, 1)
        out_user = self.lin1(torch.unsqueeze(out_user, 1))
        out_item = self.BLS.fit(sum_item_features, "item", item_masks)
        out_item = self.lin1(torch.unsqueeze(out_item, 1))

        # Matrix multiplication and fully connected layer
        out2_user = torch.matmul(inner_user_node, outer_user_node.transpose(0, 1))
        out2_user = torch.matmul(out2_user, torch.squeeze(node_user))    #(batch_size*n_node, dim)
        out2_item = torch.matmul(inner_item_node, outer_item_node.transpose(0, 1))
        out2_item = torch.matmul(out2_item, torch.squeeze(node_item))

        # Pointwise multiplication of out1, out2
        combined_user = out_user * out2_user   # (batch_size*n_node, 1)
        combined_item = out_item * out2_item
        combined_node = torch.zeros(len(node_id), self.dim).to(self.device)
        combined_node[user_mask] = combined_user
        combined_node[item_mask] = combined_item
        # aggregate all message
        if len(outer_node_message.size()) < len(node_emb.size()):
           combined_node = combined_node.unsqueeze(1)

        updated_node_input = torch.cat((node_emb, combined_node), 1)
        updated_node_input = torch.transpose(updated_node_input, 0, 1)

        gru_h0 = torch.normal(0, 0.01, (1, node_emb.size(0), self.dim)).to(self.device)
        gru_output, hn = self.update_f(updated_node_input, gru_h0)
        updated_node = gru_output[-1]                     #[batch_size*n_node, dim]

        updated_graph = torch.squeeze(global_mean_pool(updated_node, new_batch))
        item_graphs, user_graphs = torch.split(updated_graph, int(updated_graph.size(0)/2))

        y = torch.unsqueeze(torch.sum(user_graphs * item_graphs, 1) + sum_weight, 1)
        y = torch.sigmoid(y)

        return y

    def connect(self, batch, num_user_features, users_emb, items_emb, useratt_emb, itematt_emb):
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)  # 划分出每张图的节点数
        num_item_features = (nodes_per_graph - num_user_features).tolist()

        user_message = useratt_emb * users_emb
        item_message = itematt_emb * items_emb

        user_message_list = torch.split(user_message, num_user_features, dim=0)
        item_message_list = torch.split(item_message, num_item_features, dim=0)
        fusion_node_message = torch.empty(0, dtype=useratt_emb.dtype).to(self.device)
        for i in range(len(num_item_features)):

            messages = torch.cat((user_message_list[i], item_message_list[i]), 0)
            fusion_node_message = torch.cat((fusion_node_message, messages), 0)

        return fusion_node_message

    def split_batch(self, batch, user_node_num):
        """
        split batch id into user nodes and item nodes
        """
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        cum_num_list = [cum_num + i for i in range(user_node_num)]
        multi_hot = torch.cat(cum_num_list)
        test = torch.sum(F.one_hot(multi_hot, ones.size(0)), dim=0) * (torch.max(batch) + 1 )

        return batch + test

    def outer_offset(self, batch, user_node_num, outer_edge_index):
        ones = torch.ones_like(batch)
        nodes_per_graph = global_add_pool(ones, batch)
        inter_per_graph = (nodes_per_graph - user_node_num) * user_node_num * 2
        cum_num = torch.cat((torch.LongTensor([0]).to(self.device), torch.cumsum(nodes_per_graph, dim=0)[:-1]))
        offset_list = torch.repeat_interleave(cum_num, inter_per_graph, dim=0).repeat(2, 1)
        outer_edge_index_offset = outer_edge_index + offset_list
        return outer_edge_index_offset



