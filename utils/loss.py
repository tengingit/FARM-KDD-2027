import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossDimPrototypeLoss(nn.Module):
    def __init__(self, num_dim, num_per_dim, dim_z, momentum=0.9, warmup_epochs=100):
        super().__init__()
        self.num_dim = num_dim
        self.max_c = max(num_per_dim)
        self.momentum = momentum
        self.warmup_epochs = warmup_epochs

        self.register_buffer("base_proto", torch.zeros(num_dim, self.max_c, dim_z))
        self.register_buffer("proto_init", torch.zeros(num_dim, self.max_c, dtype=torch.bool))

        self.register_buffer("alpha", torch.zeros(num_dim, self.max_c))
        self.register_buffer("cond_prob", torch.zeros(num_dim, self.max_c, num_dim, self.max_c))

        self.register_buffer("weight_gamma", torch.zeros(num_dim, self.max_c, num_dim))

    @torch.no_grad()
    def build_corrected_proto(self):

        init_mask = self.proto_init.unsqueeze(-1)
        safe_proto = self.base_proto * init_mask

        G_per_dim = torch.einsum('jckd, kdz -> jckz', self.cond_prob, safe_proto)

        gamma = self.weight_gamma.clone()

        eye_mask = torch.eye(self.num_dim, device=gamma.device).view(self.num_dim, 1, self.num_dim)
        gamma = gamma * (1 - eye_mask)

        sum_gamma = gamma.sum(dim=2, keepdim=True)

        tilde_gamma = gamma / (sum_gamma + 1e-8)

        G = (tilde_gamma.unsqueeze(-1) * G_per_dim).sum(dim=2)

        a = self.alpha.unsqueeze(-1)
        corrected_proto = (1 - a) * self.base_proto + a * G

        return corrected_proto

    def forward(self, z_list, labels, epoch):
        self.update_base_proto(z_list, labels)

        if epoch < self.warmup_epochs:
            return torch.tensor(0.0, device=labels.device, requires_grad=True)

        corrected_p = self.build_corrected_proto()

        dim_losses = []

        for j in range(self.num_dim):
            y = labels[:, j]
            z = z_list[j]

            valid_mask = (y >= 0) & self.proto_init[j, y]
            if not valid_mask.any(): continue

            z_valid = z[valid_mask]
            p_target = corrected_p[j, y[valid_mask]]

            sim = F.cosine_similarity(z_valid, p_target, dim=-1)

            loss_j = (1.0 - sim).mean()
            dim_losses.append(loss_j)

        return torch.stack(dim_losses)

    @torch.no_grad()
    def update_base_proto(self, z_list, labels):

        for j in range(self.num_dim):
            z = z_list[j]
            y = labels[:, j]
            unique_labels = torch.unique(y)
            for c in unique_labels:
                if c < 0: continue
                mask = (y == c)
                z_mean = z[mask].mean(dim=0)
                if not self.proto_init[j, c]:
                    self.base_proto[j, c] = z_mean
                    self.proto_init[j, c] = True
                else:
                    self.base_proto[j, c] = self.momentum * self.base_proto[j, c] + (1 - self.momentum) * z_mean
