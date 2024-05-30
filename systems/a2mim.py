import torch
import torch.nn as nn
from types import MethodType

from consts import WEIGHT_DIR

WEIGHTS = {
    'resnet50': f"{WEIGHT_DIR}/a2mim_r50_l3_sz224_init_8xb256_cos_ep300_ft_rsb_a2.pth"
}

def load_weights(model, weight_name='resnet50'):
    if weight_name not in WEIGHTS.keys():
        raise ValueError(f'invalid weight name {weight_name}')
    elif weight_name == 'resnet50':
        raw_ckpt = torch.load(WEIGHTS[weight_name])
        sd = {k[k.find('.')+1:] : v for k,v in raw_ckpt['state_dict'].items()}
        model.load_state_dict(sd)

class ResNetMIMMask(nn.Module):
    def __init__(self,
                 mask_mode='learnable',
                 mask_init=0,
                 replace=True,
                 detach=False,
                 mask_dims=1024):
        super().__init__()
        self.mask_mode = mask_mode
        self.replace = replace
        self.detach = detach
        assert self.mask_mode in [None, 'randn', 'zero', 'mean', 'instance_mean', 'learnable',]
        self.mask_dims = mask_dims
        if self.mask_mode not in [None, 'instance_mean',]:
            self.mask_token = nn.Parameter(torch.zeros(1, self.mask_dims, 1, 1))
        if mask_init > 0 and not replace:
            self.mask_gamma = nn.Parameter(
                mask_init * torch.ones((1, self.mask_dims, 1, 1)), requires_grad=True)
        else:
            self.mask_gamma = None
    
    def forward(self, x, mask=None):
        """ perform MIM with mask and mask_token """
        B, _, H, W = x.size()
        if self.mask_mode is None:
            return x
        elif self.mask_mode == 'instance_mean':
            mask_token = x.mean(dim=[2, 3], keepdim=True).expand(B, -1, H, W)
        else:
            if self.mask_mode == 'mean':
                self.mask_token.data = x.mean(dim=[0, 2, 3], keepdim=True)
            mask_token = self.mask_token.expand(B, -1, H, W)
        assert mask is not None
        mask = mask.view(B, 1, H, W).type_as(mask_token)
        if self.replace:
            x = x * (1. - mask) + mask_token * mask
        else:
            if self.detach:
                x = x * (1. - mask) + x.clone().detach() * mask
            if self.mask_gamma is not None:
                x = x * (1. - mask) + (x * mask) * self.mask_gamma
            x = x + mask_token * mask  # residual
        return x

def get_resnet_layer_input_dim(net, layer_id):
    layer = getattr(net, f'layer{layer_id}')
    inp_block = layer[0]
    return inp_block.conv1.in_channels

def inject_mask(net, mask_cfg, layer=4):
    mask_dims = get_resnet_layer_input_dim(net, layer)
    # add mask module to the resnet
    net.add_module('mim_mask', ResNetMIMMask(mask_dims=mask_dims, **mask_cfg))
    # define forward function with masking operation added at the correct layer
    def modified_forward(self, x, mask=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(1,5):
            if i == layer and self.training: # disable masking in eval mode
                self.mim_mask(x)
            x = getattr(self, f'layer{i}')(x)
        return x
    # inject the masked forward into the resnet
    net.forward = MethodType(modified_forward, net)
    return net

class A2MIMSystem:
    """
    - cfg: easydict
        - mask_cfg: nested dict
            - mask_mode: str
            - mask_init: float
            - replace: bool
            - detach: bool
        - mask_layer: int (1 to 4)
        - hparams: nested dict
            - n_epochs: int
            - lr: float
            - wd: float
    - model: unmodified resnet model
    - trn_dset, val_dset are pytorch datasts
    - OptimClass: class of optimizer to use (e.g. torch.optim.AdamW)
    """
    def __init__(self, cfg, model, trn_dset, val_dset=None, \
                 device='cpu'):
        self.model = inject_mask(model, cfg.mask_cfg, layer=cfg.mask_layer)
        self.trn_dset = trn_dset
        self.val_dset = val_dset

    def train_step(self, batch):
        pass
