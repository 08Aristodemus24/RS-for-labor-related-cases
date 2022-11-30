import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb


####################### Basic Ops #############################

# # PGNN layer, only pick closest node for message passing
class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim, dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim * 2, output_dim)
        self.linear_out_position = nn.Linear(output_dim, 1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages)  # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data, feature_set_name):
        x = data.features[feature_set_name]
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GCN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)

    def forward(self, data, feature_set_name):
        x, edge_index = data.features[feature_set_name], data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class SAGE(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            # self.linear_pre = nn.Linear(input_dim, feature_dim) CHANGED FROM THAT!!!!!!!!!
            # self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)

            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_pre_sec = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.SAGEConv(hidden_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data, feature_set_name):
        x, edge_index = data.features[feature_set_name], data.edge_index
        if self.feature_pre:
            # x = self.linear_pre(x)
            x = self.linear_pre(x)
            x = self.linear_pre_sec(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

    def forward(self, data, feature_set_name):
        x, edge_index = data.features[feature_set_name], data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class GIN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

    def forward(self, data, feature_set_name):
        x, edge_index = data.features[feature_set_name], data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        return x


class PGNN(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num > 1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

    def forward(self, data, feature_set_name):
        x = data.features[feature_set_name]
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        return x_position


class PGNN_plus(torch.nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PGNN_plus, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        # self.out_aggregator = nn.Linear(hidden_dim * 2, hidden_dim)

        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_first = nn.Linear(input_dim, feature_dim)
            self.linear_second = nn.Linear(feature_dim, hidden_dim)

            self.conv_first = PGNN_layer(hidden_dim, hidden_dim)
            self.conv_second = tg.nn.SAGEConv(hidden_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
            self.conv_second = tg.nn.SAGEConv(input_dim, hidden_dim)
            # mix or not mix? not mix for now, but can discuss
        if layer_num > 1:
            self.conv_hidden_first = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out_first = PGNN_layer(hidden_dim, output_dim)
            self.conv_hidden_second = nn.ModuleList(
                [tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out_second = tg.nn.SAGEConv(hidden_dim, output_dim)

    def forward(self, data, feature_set_name):
        x, edge_index = data.features[feature_set_name], data.edge_index
        if self.feature_pre:
            x = self.linear_first(x)
            x = self.linear_second(x)

        x_position, x_p = self.conv_first(x, data.dists_max, data.dists_argmax)

        x_sage = self.conv_second(x, edge_index)
        # x = F.relu(x) # Note: optional!
        if self.layer_num == 1:
            if self.dropout:
                x_sage = F.dropout(x_sage, training=self.training)
                # x_position = F.dropout(x_position, training=self.training)
            x_sage = self.conv_out_second(x_sage, edge_index)
            x_sage = F.normalize(x_sage, p=2, dim=-1)
            x_two = torch.cat((x_position, x_sage), 0)
            # x_out = self.out_aggregator(x_two)
            return x_two

        for i in range(self.layer_num - 2):
            _, x_p = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            x_sage = self.conv_hidden[i](x, edge_index)
            if self.dropout:
                x_p = F.dropout(x_p, training=self.training)
                x_sage = F.dropout(x_sage, training=self.training)
        x_position, x_p = self.conv_out_first(x_p, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        x_sage = self.conv_out_second(x_sage, edge_index)
        x_sage = F.normalize(x_sage, p=2, dim=-1)

        x_two = torch.cat((x_position, x_sage), dim=-1)

        # x_out = self.out_aggregator(x_two)

        # dropout?
        return x_two


class Baseline(torch.nn.Module):
    def __init__(self, feature_shapes, hidden_dim, output_dim, device, **kwargs):
        super(Baseline, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        linear_first_layers = []
        linear_second_layers = []
        # first_counter = 0
        # second_counter = 0

        for i in range(len(feature_shapes)):
            if feature_shapes[i] >= hidden_dim:
                linear_first_layers.append(nn.Linear(feature_shapes[i], hidden_dim))
                linear_second_layers.append(nn.Linear(hidden_dim, output_dim))

                # first_counter += hidden_dim
                # second_counter += output_dim

            elif output_dim <= feature_shapes[i] < hidden_dim:
                linear_first_layers.append(nn.Linear(feature_shapes[i], output_dim))
                linear_second_layers.append(nn.Linear(output_dim, output_dim))

                # first_counter += output_dim
                # second_counter += output_dim

            elif feature_shapes[i] < output_dim:
                linear_first_layers.append(None)
                linear_second_layers.append(None)

                # first_counter += feature_shapes[i]
                # second_counter += feature_shapes[i]

        # print()
        # print(first_counter)
        # print(second_counter)
        # exit(0)

        self.linear_first = nn.ModuleList(linear_first_layers)
        self.linear_second = nn.ModuleList(linear_second_layers)
        self.dropout = nn.Dropout()

    def forward(self, feature_vectors):
        output_vectors = []

        for i in range(len(feature_vectors)):
            feature = feature_vectors[i].to(self.device)

            if feature.shape[1] < self.output_dim:
                output_vectors.append(feature.clone().detach().requires_grad_(True))
                continue

            first_out = self.linear_first[i](feature)
            second_out = self.linear_second[i](first_out)
            # second_out = self.dropout(second_out)

            output_vectors.append(second_out)

        return output_vectors


class Linear(torch.nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(Linear, self).__init__()

        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, prediction_vector):
        output = self.layer(prediction_vector)

        return output
