import torch
import torch.nn as nn
import numpy as np


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
