import torch.nn as nn

from mlp import MLP
from utils.utils import Init_random_seed


class FairMDC(nn.Module):
    def __init__(self, configs):
        super(FairMDC, self).__init__()
        self.rand_seed = configs['rand_seed']
        self.num_dim = configs['num_dim']

        self.adapters = nn.ModuleList([
            MLP(
            configs['num_feature'],
            configs['dim_z'],
            [configs['hidden_size']],
            batchNorm=False,dropout=False,nonlinearity='relu',
            with_output_nonlinearity=True
            )
            for j in range(self.num_dim)
        ])

        self.heads = nn.ModuleList([
            MLP(
                configs['dim_z'],
                configs['num_per_dim'][j],
                [],
                batchNorm=False,dropout=False,nonlinearity='relu',
                with_output_nonlinearity=False
            )
            for j in range(self.num_dim)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        Init_random_seed(self.rand_seed)

        for adapter in self.adapters:
            adapter.reset_parameters()
        for head in self.heads:
            head.reset_parameters()

    def forward(self, X):

        Z_list, logits = [], []

        for j in range(self.num_dim):
            Z_j = self.adapters[j](X)
            logit_j = self.heads[j](Z_j)

            Z_list.append(Z_j)
            logits.append(logit_j)

        return Z_list, logits
