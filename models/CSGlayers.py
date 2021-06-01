import typing as t

import torch
import torch.nn  as nn
import torch.nn.functional as F

FLOAT_EPS = torch.finfo(torch.float32).eps

def gumbel_softmax(prob, temp=1.0, dim=-2):
    sample = torch.rand_like(prob).clamp_min(FLOAT_EPS)
    gumbel_sample = -torch.log(-torch.log(sample) + FLOAT_EPS)
    if isinstance(temp, torch.Tensor):
        temp = temp.clamp(min = FLOAT_EPS) 

    return ((torch.log(prob + FLOAT_EPS) + gumbel_sample) / temp).softmax(dim=dim)

class CSG_layer(nn.Module):
    def __init__(self, num_shape_in, num_shape_out, threshold, latent_sz = 256):
        super().__init__()

        self.num_shape_in = num_shape_in
        self.num_shape_out = num_shape_out
        self.threshold = threshold
        self.latent_sz = latent_sz

        self.K_left = nn.Parameter(torch.Tensor(self.latent_sz, self.num_shape_in, 
                                    self.num_shape_out), requires_grad = True)
        self.K_right = nn.Parameter(torch.Tensor(self.latent_sz, self.num_shape_in, 
                                    self.num_shape_out), requires_grad = True)
        self.V_encode: t.Optional[torch.Tensor] = None
        nn.init.normal_(self.K_left, std=0.1)
        nn.init.normal_(self.K_right, std=0.1)
        
        self.temp = nn.Parameter(torch.Tensor(1), requires_grad = True)
        nn.init.constant_(self.temp, 2.)

        self.Vmask_encoder = nn.Sequential(nn.Linear(num_shape_in*2*num_shape_out, latent_sz),
                                    nn.LeakyReLU(negative_slope=0.01), nn.Linear(latent_sz, latent_sz))

    def forward(self, x, latent_vec):
        """
        input: [batch_sz, num_pt, num_shape]
        """
        batch_sz = x.shape[0]

        V_left = torch.sum(self.K_left.unsqueeze(dim = 0) * 
                        latent_vec[..., None, None], dim=1).softmax(dim=-2) #[batch, num_shape_in, num_shape_out]
        V_right = torch.sum(self.K_right.unsqueeze(dim = 0) * 
                        latent_vec[..., None, None], dim=1).softmax(dim=-2) #[batch, num_shape_in, num_shape_out]

        V_side = torch.stack([V_left, V_right], dim=1).unsqueeze(dim=2) #[batch, 2, 1, num_shape_in, num_shape_out]
        self.V_encode = self.Vmask_encoder(V_side.reshape(batch_sz, -1))

        V_mask = gumbel_softmax(V_side, self.temp.clamp(min=FLOAT_EPS, max=2), dim=-2) #[batch, 2, 1, num_shape_in, num_shape_out]
        shape_mask = V_mask.split(split_size=1, dim=1)
        y = x.unsqueeze(dim=-1) #[batch, num_pt, num_shape, 1]

        cand_shape = []
        for mask in shape_mask:
            mask = mask.squeeze(dim=1) #[batch, 1, num_shape_in, num_shape_out]
            cand_shape.append((mask * y).sum(dim=-2)) #[batch, num_pt, num_shape_out]
        
        y = torch.stack(cand_shape, dim=2) #[batch, num_pt, 2, num_shape_out]
        y = torch.cat([y[:, :, 0] + y[:, :, 1], y[:, :, 0] + y[:, :, 1] - 1,
                    y[:, :, 0] - y[:, :, 1], y[:, :, 1] - y[:, :, 0]],dim=-1)
        y = y.clamp(min=0, max=1) #[batch, num_pt, num_shape_out, 4]

        return y