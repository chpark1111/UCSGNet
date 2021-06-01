'''
Let's add more shapeevals and test them!!!
https://www.iquilezles.org/www/articles/distfunctions/distfunctions.htm
'''
import math
import typing as t

import torch
import torch.nn  as nn

from utils.util import quat_to_rot_matrix

class ShapeEval(nn.Module):
    def __init__(self, num_shape, num_param, num_dim, latent_sz=256):
        super(ShapeEval, self).__init__()
        
        self.num_shape = num_shape
        self.num_param = num_param
        self.num_dim = num_dim

        if self.num_dim == 2:
            self.num_rot = 1
        elif self.num_dim == 3:
            self.num_rot = 4
        else:
            print("Not supported dim")
            assert 0
        self.latent_sz = latent_sz
        
        self.fc = nn.Linear(self.latent_sz * 8, self.num_shape * (self.num_param + 
                                self.num_dim + self.num_rot))

        self.param: t.Optional[torch.Tensor] = None
        self.shift: t.Optional[torch.Tensor] = None
        self.rotation: t.Optional[torch.Tensor] = None
        
    def forward(self, x, pt):
        '''
        x: (batch_size, latent_sz * 8)
        pt: (num_pt, 2) y, x order
        '''
        batch_sz = x.shape[0]
        primitives = self.fc(x).reshape(batch_sz, self.num_shape, -1) #[batch_sz, num_shape, num_param + num_dim + self.num_rot]
        
        param = primitives[:, :, :self.num_param] #[batch_sz, num_shape, num_param]
        shift = primitives[:, :, self.num_param: (self.num_param + self.num_dim)] #[batch_sz, num_shape, num_dim]
        rotation = primitives[:, :, (self.num_param + self.num_dim):] #[batch_sz, num_shape, num_rot]

        self.param = param
        self.shift = shift
        self.rotation = rotation

        pt = self.transform(pt, shift, rotation)
        return self.evalpt(pt, param)
    
    def transform(self, pt, shift, rotation):
        rotation = rotation.unsqueeze(dim = -2) #[batch_sz, num_shape, 1, num_rot]

        if self.num_dim == 2:
            rotation_mat = rotation.new_zeros(rotation.shape[:-1] + (2, 2)) #[batch_sz, num_shape, 1, 2, 2]
            rotation = rotation[..., 0] #[batch_sz, num_shape, 1] 

            rotation_mat[..., 0, 0] = rotation.cos()
            rotation_mat[..., 0, 1] = rotation.sin()
            rotation_mat[..., 1, 0] = -rotation.sin()
            rotation_mat[..., 1, 1] = rotation.cos()

        else:
            rotation_mat = quat_to_rot_matrix(rotation)
            rotation_mat = rotation_mat.transpose(-2, -1)
            
        pt = pt - shift.unsqueeze(dim=-2) #[batch_sz, num_shape, num_pt, num_dim]
        pt = (rotation_mat * pt.unsqueeze(dim=-1)).sum(dim=-2) #[batch_sz, num_shape, num_pt, num_dim]

        return pt
    
    def shift_vector_prediction(self):
        return self.shift

    def rotation_params_prediction(self):
        return self.rotation

    def last_predicted_parameters_of_shape(self):
        return self.param

    def clear_shift_vector_prediction(self):
        self.shift = None

    def last_parameters(self):
        return self.param

    def evalpt(self, pt, param):
        return 1

class CircleSphereEval(ShapeEval):
    def __init__(self, num_shape, num_dim):
        super().__init__(num_shape, 1, num_dim)

    def evalpt(self, pt, param): 
        '''
        pt: [batch_sz, num_shape, num_pt, num_dim]
        param: [batch_sz, num_shape, num_param]
        '''
        dis = pt.norm(dim=-1)
        return dis - param

    def vol(self):
        if self.param is None:
            return 0.0
        if self.num_dim == 3:
            return 4 / 3 * self.param.pow(3).sum(dim=-1) * math.pi
        elif self.num_dim == 2:
            return self.param.pow(2).sum(dim=-1) * math.pi
        else:
            print("Not supported dim")
            assert 0

class SquareCubeEval(ShapeEval):
    def __init__(self, num_shape, num_dim):
        super().__init__(num_shape, num_dim, num_dim)

    def evalpt(self, pt, param):
        '''
        pt: [batch_sz, num_shape, num_pt, num_dim]
        param: [batch_sz, num_shape, num_param]
        '''
        q = pt.abs() - param.unsqueeze(dim=-2)
        dis = q.max(torch.zeros_like(q)).norm(dim=-1)

        q_x = q[..., 0]
        q_y = q[..., 1]
        if self.num_dim > 2:
            q_z = q[..., 2]
            ret = q_x.max(q_y).max(q_z).min(torch.zeros_like(dis))
        else:
            ret = q_x.max(q_y).min(torch.zeros_like(dis))
    
        return dis + ret

    def vol(self):  
        return self.param.prod(dim=-1)

'''
Below are almost copied from
https://github.com/kacperkan/ucsgnet/
'''
class PlanesEval(nn.Module):
    def __init__(self, num_plane, num_dim, latent_sz = 256):
        super().__init__()
        self.latent_sz = latent_sz
        self.num_plane = num_plane
        self.num_dim = num_dim

        self.fc = nn.Linear(self.latent_sz * 8, self.num_plane * (self.num_dim + 1))
        self.param: t.Optional[torch.Tensor] = None
    
    def forward(self, x, pt):
        pt_shape = tuple(pt.shape[:-1]) + (1,)

        additional_pt = pt.new_ones(pt_shape)
        extended_pt = torch.cat((pt, additional_pt), dim=-1)

        param = self.fc(x).reshape(-1, self.num_plane, self.num_dim + 1)
        dis = extended_pt.bmm(param.permute(0, 2, 1))

        self.param = param

        return dis

    def last_predicted_parameters(self):
        return self.param


class CompoundEval(nn.Module):
    def __init__(self, parts):
        super().__init__()
        self.parts = nn.ModuleList(parts)
    
    def forward(self, x, pt):  
        return torch.cat([part(x, pt) for part in self.parts], dim=1)
    
    def __len__(self):
        return len(self.parts)

    def get_all_shift_vectors(self):
        return torch.cat([part.shift_vector_prediction for part in self.parts], dim=1)

    def get_all_rotation_vectors(self):
        return torch.cat([part.rotation_params_prediction for part in self.parts], dim=1)

    def get_all_last_predicted_parameters_of_shapes(self):
        return [(part.__class__.__name__, part.last_predicted_parameters_of_shape)
            for part in self.parts]

    def clear_translation_vectors(self):
        for part in self.parts:
            part.clear_shift_vector_prediction()

    def __iter__(self) -> ShapeEval:
        for part in self.parts:
            yield part

    def enumerate_indices(self):
        offset = 0
        for part in self.parts:
            yield offset, offset + part.num_shape
            offset += part.num_shape

def CompoundShapeEval(planes: bool, num_shape, num_dim):
    if planes:
        return PlanesEval(num_shape, num_dim)
    return CompoundEval([CircleSphereEval(num_shape, num_dim),
                SquareCubeEval(num_shape, num_dim)])