import torch
from consts import WEIGHT_DIR
from ..models.resnet import ResNet

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

def modified_forward(model_obj, x):
    x = model_obj.conv1(x)
    x = model_obj.bn1(x)
    x = model_obj.relu(x)
    x = model_obj.maxpool(x)

    for n in range(1,5):
        x = getattr(model_obj, f'layer{n}')(x)


    x = model_obj.avgpool(x)
    x = torch.flatten(x, 1)
    x = model_obj.fc(x)

    return x


def inject_mask(resnet, layer=4):
    pass

# class MIMResNet(ResNet):
#     def __init__(self, *args, **kwargs):

#     def forward(self, x):
#         pass