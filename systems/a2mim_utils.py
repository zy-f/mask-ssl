import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from types import MethodType
from openmixup.models import builder as om_builder
from openmixup.core.optimizers import builder as oo_builder

## modifications to the resnet
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

def inject_mask(net, mask_cfg={}, layer=4):
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
                self.mim_mask(x, mask=mask)
            x = getattr(self, f'layer{i}')(x)
        return x
    # inject the masked forward into the resnet
    net.forward = MethodType(modified_forward, net)
    return net

def inject_multi_mask(net, mask_cfg=[]):
    for m_layer, cfg in mask_cfg:
        mask_dims = get_resnet_layer_input_dim(net, m_layer)
        net.add_module(f'mim_mask{m_layer}', ResNetMIMMask(mask_dims=mask_dims, **cfg))
    def modified_forward(self, x, mask=None, layer=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(1,5):
            if i == layer and self.training: # disable masking in eval mode
                getattr(self, f'mim_mask{i}')(x, mask=mask)
            x = getattr(self, f'layer{i}')(x)
        return x
    net.forward = MethodType(modified_forward, net)
    return net

## modifications to imagenet
class BlockwiseMaskGenerator(object):
    """Generate random block for the image.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(self,
                 input_size=224,
                 mask_patch_size=32,
                 model_patch_size=1,
                 mask_ratio=0.6,
                 mask_color='zero',
                 seed=None
                ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_color = mask_color
        assert self.mask_color in ['mean', 'zero']
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        assert isinstance(img, torch.Tensor)
        mask_idx = self.rng.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.from_numpy(mask)  # [H, W]

        if self.mask_color == 'mean':
            mask_img = mask.repeat_interleave(self.model_patch_size, 0).repeat_interleave(
                self.model_patch_size, 1).contiguous()
            img = img.clone()
            mean = img.mean(dim=[1,2])
            for i in range(img.size(0)):
                img[i, mask_img == 1] = mean[i]
        return img, mask

class MaskedDataset(Dataset):
    def __init__(self, base_dataset, mask_gen_cfg={}, mask_layer=4, input_name='x', seed=None):
        self.base_dataset = base_dataset
        mask_gen_cfg['model_patch_size'] = 2**mask_layer
        mask_gen_cfg['seed'] = seed
        self.mask_generator = BlockwiseMaskGenerator(**mask_gen_cfg)
        self.do_mask = True
        self.key = input_name

    def enable_mask(self, enable=True):
        self.do_mask = enable
    
    def disable_mask(self):
        self.enable_mask(enable=False)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, i):
        out = self.base_dataset[i]
        if self.do_mask:
            out[self.key+'_mask'], out.mask = self.mask_generator(out[self.key])
        return out
    
class MultiMaskedDataset(Dataset):
    def __init__(self, base_dataset, mask_gen_cfg=[], input_name='x', seed=None):
        self.base_dataset = base_dataset
        self.mask_layers = []
        self.mask_generators = []
        for mask_layer, cfg in mask_gen_cfg:
            cfg['model_patch_size'] = 2**mask_layer
            self.mask_layers.append(mask_layer)
            self.mask_generators.append(BlockwiseMaskGenerator(**cfg, seed=seed))
        self.do_mask = True
        self.key = input_name
        self.rng = np.random.default_rng(seed)

    def enable_mask(self, enable=True):
        self.do_mask = enable
    
    def disable_mask(self):
        self.enable_mask(enable=False)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, indices): # must requet entire batch at once
        mask_idx = self.rng.integers(len(self.mask_layers))
        batch = []
        for i in indices:
            out = self.base_dataset[i]
            if self.do_mask:
                # select a random mask to apply    
                out[self.key+'_mask'], out.mask = self.mask_generators[mask_idx](out[self.key])
            batch.append(out)
        out_batch = torch.utils.data.default_collate(batch)
        out_batch.layer = self.mask_layers[mask_idx]
        return out_batch

## ADDING DECODER (wrapper)
NECK_CFG = dict(
    type='NonLinearMIMNeck',
    decoder_cfg=None,
    kernel_size=1,
    in_channels=2048, in_chans=3, encoder_stride=32
)

def get_resnet_output_dim(net):
    return net.fc.in_features

class A2MIMModel(nn.Module):
    def __init__(self, backbone, cfg=NECK_CFG):
        super().__init__()
        self.backbone = backbone
        cfg['in_channels'] = get_resnet_output_dim(backbone)
        self.decoder = om_builder.build_neck(cfg)
        self.decoder.init_weights()

    def forward(self, x, mask=None, layer=None):
        if layer is None:
            x = self.backbone(x, mask=mask)
        else:
            x = self.backbone(x, mask=mask, layer=layer)
        return self.decoder([x])[0]


## LOSS (wrapper)
HEAD_CFG = dict(
    type='A2MIMHead',
    loss=dict(type='RegressionLoss', mode='focal_l1_loss',
        loss_weight=1.0, reduction='none',
        activate='sigmoid', alpha=0.2, gamma=1.0, residual=False),
    unmask_weight=0.,
    fft_weight=0.5,
    fft_focal=True,
    fft_unmask_weight=1e-3,  # unmask patches in the fft loss
    fft_unmask_replace='mixed',
    fft_reweight=False,
    encoder_in_channels=3
)

class A2MIMLoss(nn.Module):
    def __init__(self, cfg=HEAD_CFG):
        super().__init__()
        self.head = om_builder.build_head(cfg)
    def forward(self, img, img_rec, mask):
        return self.head(img, img_rec, mask)['loss']

## optimizer wrapper
def prepare_optimizer(model, hparams=None):
    opt_cfg = dict(
        type='AdamW',
        lr=3e-4, # modified from theirs
        betas=(0.9, 0.999), weight_decay=0.05, eps=1e-8,
        paramwise_options={
            '(bn|ln|gn)(\d+)?.(weight|bias)': dict(weight_decay=0.),
            'bias': dict(weight_decay=0.),
            'mask_token': dict(weight_decay=0., lr_mult=1e-1,),
    })
    if hparams is not None:
        for k in opt_cfg.keys():
            if hparams.get(k) is not None:
                opt_cfg[k] = hparams[k]
    return oo_builder.build_optimizer(model, opt_cfg)