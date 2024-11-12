import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_

class BLS(nn.Module):
    """ Broad Learning System """

    def __init__(self, embedding_size):
        super(BLS, self).__init__()

        #self.F = Fields_size
        self.k = embedding_size
        self.NumEnhan = embedding_size
        self.MapNode = embedding_size

        self.initializer = nn.init.xavier_normal_
        self.mapping_weights = nn.ParameterList()
        self.mapping_biases = nn.ParameterList()
        self.enhance_weights = nn.ParameterList()
        self.enhance_biases = nn.ParameterList()

        # Initialize weights and biases for mapping and enhancing nodes
        for i in range(self.MapNode):
            weight = nn.Parameter(torch.Tensor(1, 1))
            bias = nn.Parameter(torch.Tensor(1, ))
            nn.init.xavier_normal_(weight)  # Initialize weights
            #nn.init.normal_(bias, 0, 0.01)
            nn.init.constant_(bias, 0)  # Initialize biases
            self.mapping_weights.append(weight)
            self.enhance_weights.append(weight)
            self.mapping_biases.append(bias)
            self.enhance_biases.append(bias)

    def mapping_node(self, data, name, mask):
        if name == 'user':
            mask = 1
        elif name == 'item':
            mask = mask
        weights = self.mapping_weights[0]
        out0 = torch.matmul(data, weights) + self.mapping_biases[0]*mask
        input = out0
        result = out0
        for i in range(1, self.MapNode):
            out_temp = F.leaky_relu(torch.matmul(input, self.mapping_weights[i]) + self.mapping_biases[i]*mask)
            input = out_temp
            result = torch.cat([result, out_temp], dim=-1)

        return result

    def enhance_node(self, data, name, mask):
        if name == 'user':
            mask = 1
        elif name == 'item':
            mask = mask
        weights = self.enhance_weights[0]
        out0 = torch.matmul(data, weights) + self.enhance_biases[0]*mask
        input = out0
        out0 = out0.mean(dim=-1, keepdim=True)
        result = out0
        for i in range(1, self.NumEnhan):
            out_temp = F.leaky_relu(torch.matmul(input, self.enhance_weights[i]) + self.enhance_biases[i]*mask)
            input = out_temp
            result = torch.cat([result, out_temp], dim=-1)

        return result

    def pinv_mat(self, data):
        r = torch.tensor(0.1, requires_grad=True)
        E = r * torch.eye(data.size(-1)).cuda()
        ATA = torch.matmul(data.transpose(1, 2), data)
        part1 = torch.inverse(ATA + E)
        part2 = data.transpose(1, 2)
        out = torch.matmul(part1, part2)
        out = out.transpose(1, 2)

        return out

    def fit(self, data, name, mask):
        data = data.mean(dim=-1, keepdim=True)
        mask = mask.mean(dim=-1, keepdim=True)
        mappingdata = self.mapping_node(data, name, mask)
        weights = nn.Parameter(torch.Tensor(self.MapNode, )).cuda()
        nn.init.normal_(weights, 0, 0.01)
        enhance_input = mappingdata * weights
        enhance_input = torch.mean(enhance_input, dim=-1, keepdim=True)
        enhancedata = self.enhance_node(enhance_input, name, mask)
        inputdata = torch.cat([mappingdata, enhancedata], dim=-1)
        pesuedoinverse = self.pinv_mat(inputdata)
        bls_out = pesuedoinverse.mean(dim=-1, keepdim=True)
        #bls_out = torch.sigmoid(bls_out)
        bls_out_flat = bls_out.view(-1)
        if name == 'user':
            bls_out_flat = bls_out_flat
        if name == 'item':
            bls_out_flat = bls_out_flat[bls_out_flat != 0]
        bls_out_flat = torch.sigmoid(bls_out_flat)

        return bls_out_flat