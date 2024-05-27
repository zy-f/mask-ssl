import torch
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