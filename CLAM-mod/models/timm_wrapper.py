import torch
import timm
import timm_3d #Ahmad
import pdb


class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50.tv_in1k', dim = 2, Blacks=True,
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        self.dim=dim
        if dim == 2:
            self.model = timm.create_model(model_name, **kwargs)
            if not Blacks:
                self._adjust_first_conv_layer()
            if pool:
                self.pool = torch.nn.AdaptiveAvgPool2d(1)
            else:
                self.pool = None
        elif dim == 3:
            self.model = timm_3d.create_model(model_name, **kwargs)
            #self._adjust_padding_stride_of_first_layer() #I want to get rid of the depth-wise padding. Ahmad
            if not Blacks:
                self._adjust_first_conv_layer()
            if pool:
                self.pool = torch.nn.AdaptiveAvgPool3d(1)
            else:
                self.pool = None
        self.model_name = model_name    
    def forward(self, x):
        out = self.model(x)
        
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        #out = torch.flatten(out,start_dim=1) #for vit. Ahmad
        return out