import torch
from consts import WEIGHT_DIR
WEIGHTS = {
    'resnet50': f"{WEIGHT_DIR}/dino_resnet50_pretrain.pth"
}

def load_weights(model, weight_name='resnet50'):
    if weight_name not in WEIGHTS.keys():
        raise ValueError(f'invalid weight name {weight_name}')
    elif weight_name == 'resnet50':
        sd = torch.load(WEIGHTS[weight_name])
        model.load_state_dict(sd, strict=False)
        print("WARNING: fc layer/head has no loaded weights")