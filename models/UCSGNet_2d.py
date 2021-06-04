import torch
import torch.nn  as nn
from models.model import Encoder, Decoder, Converter
from models.shapeeval import CompoundShapeEval
from models.CSGlayers import CSG_layer

class UCSGNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args.latent_sz)
        self.decoder = Decoder(args.latent_sz)
        self.evaluator = CompoundShapeEval(args.use_planes, args.num_shape_type, args.num_dim)
        self.converter = Converter()

        self.num_shape_type = args.num_shape_type
        self.out_shape_layer = args.out_shape_layer
        self.threshold = args.threshold
        self.num_csg_layer = args.num_csg_layer
        self.use_planes = args.use_planes

        self.num_evaluators = 1 if self.use_planes else len(self.evaluator)  
        self.out_shape_eval = self.num_shape_type * self.num_evaluators

        csg_layer = []

        in_shape = self.out_shape_eval
        out_shape = self.out_shape_layer

        for i in range(args.num_csg_layer):
            if i == args.num_csg_layer-1:
                out_shape = 1
            
            csg_layer.append(CSG_layer(in_shape, out_shape, self.threshold, args.latent_sz))
            in_shape = out_shape * 4 + self.out_shape_eval
        
        self.csg_layer = nn.ModuleList(csg_layer)
        self.gru = nn.GRUCell(input_size=args.latent_sz, 
                                        hidden_size=args.latent_sz, bias=True) #In: [batch, input_size], Out: [batch, hidden_size]

        self.gru_hidden_vec = nn.Parameter(torch.Tensor(1, args.latent_sz), requires_grad=True)
        nn.init.constant_(self.gru_hidden_vec, 0.01)


    def forward(self, img, pt, return_distances_to_base_shapes=False, 
                    return_intermediate_output_csg=False, return_scaled_distances_to_shapes=False):
        batch_sz = img.shape[0]
        latent_vec = self.encoder(img)
        
        shape_param = self.decoder(latent_vec) #[batch, latent_sz * 8]
        
        if self.use_planes:
            sdf_shape = self.evaluator(shape_param, pt) #[batch, num_pt, num_shape]
        else:
            pt = pt.unsqueeze(dim=1) #[batch, 1, num_pt, 2]
            sdf_shape = self.evaluator(shape_param, pt) #[batch, num_shape, num_pt]
            sdf_shape = sdf_shape.permute((0, 2, 1)) #[batch, num_pt, num_shape]
        
        init_sdf = 1 - self.converter(sdf_shape) #[batch, num_pt, num_shape]

        last_sdf = init_sdf
        pred_sdf = [last_sdf]
        
        latent_vec = self.gru(latent_vec, 
                    self.gru_hidden_vec.expand(batch_sz, self.gru_hidden_vec.shape[1]))
        
        for i, layer in enumerate(self.csg_layer):
            if i:
                last_sdf = torch.cat([last_sdf, init_sdf], dim=-1)

            last_sdf = layer(last_sdf, latent_vec) #[batch, num_pt, num_shape_out, 4]
            pred_sdf.append(last_sdf)
            assert layer.V_encode != None
            latent_vec = self.gru(layer.V_encode, latent_vec)
        
        last_sdf = last_sdf[..., 0]
        sdf = last_sdf.clamp(0, 1)
        outputs = [sdf]
        if return_distances_to_base_shapes:
            outputs.append(sdf_shape)
        if return_intermediate_output_csg:
            outputs.append(pred_sdf)
        if return_scaled_distances_to_shapes:
            outputs.append(init_sdf)
            
        return tuple(outputs) if len(outputs) > 1 else outputs[0]
            
    def binarize(self, x):
        return (x >= self.threshold).float()