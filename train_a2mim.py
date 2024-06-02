import argparse
import systems.a2mim as a2
from models import ARCHITECTURES
from dsets.imagenet import ImageNet, TRAIN_TRANSFORM
from consts import SEED, CLASS_SUBSET_100
from utils import load_config, seed_all

def main(cfg):
    seed_all(SEED)
    cfg.mask_gen_cfg.seed = SEED
    model = ARCHITECTURES[cfg.arch](num_classes=100)
    # del model.fc
    # del model.avgpool
    sys = a2.A2MIMSystem(
        cfg=cfg,
        model=model,
        trn_dset=ImageNet('train', subset_classes=CLASS_SUBSET_100, transform=TRAIN_TRANSFORM),
        val_dset=ImageNet('val', subset_classes=CLASS_SUBSET_100),
        device='cuda'
    )
    sys.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run a2mim pretraining')
    parser.add_argument('-c', '--config', type=str,
                        help=f'name of a2mim pre-training config')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get('exp_name'):
        cfg.exp_name = 'ftune_'+cfg.weight_name
    main(cfg)