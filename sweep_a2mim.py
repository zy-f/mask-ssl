import argparse
import systems.a2mim as a2
from models import ARCHITECTURES
from dsets.imagenet import ImageNet, TRAIN_TRANSFORM
from consts import SEED, CLASS_SUBSET_100
from utils import load_config, seed_all

# one item: [layer, patch_size, mask_ratio]
SWEEP = [
    # [4, 32, 0.8], # <- need to rerun to get neck back
    [2, 4, 0.6],
    [4, 64, 0.4],
    [4, 64, 0.6],
    # [4, 32, 0.4],
    # [4, 64, 0.8],
]

def main(cfg):
    seed_all(SEED)
    cfg.mask_gen_cfg.seed = SEED
    for (layer, patch_size, mask_ratio) in SWEEP:
        cfg.exp_name = f"layer{layer}_patch{patch_size}_ratio{int(mask_ratio*100)}"
        print(f'<<< {cfg.exp_name} >>>')
        cfg.mask_layer = layer
        cfg.mask_gen_cfg.mask_patch_size = patch_size
        cfg.mask_gen_cfg.mask_ratio = mask_ratio
        # run single
        model = ARCHITECTURES[cfg.arch](num_classes=100)
        sys = a2.A2MIMSystem(
            cfg=cfg,
            model=model,
            trn_dset=ImageNet('train', subset_classes=CLASS_SUBSET_100, transform=TRAIN_TRANSFORM),
            val_dset=ImageNet('val', subset_classes=CLASS_SUBSET_100),
            device='cuda'
        )
        sys.run_training()

if __name__ == '__main__':
    base_cfg = load_config('gen_a2mim_res50')
    main(base_cfg)