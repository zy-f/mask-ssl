import argparse
import torch

from models import ARCHITECTURES
from dsets.imagenet import ImageNet, TRAIN_TRANSFORM
from consts import SEED, CLASS_SUBSET_100, WEIGHT_DIR
from utils import load_config, seed_all
import systems.finetune as ftune
import systems.a2mim as a2

def load_model(arch, weight_name, n_cls=100):
    model = ARCHITECTURES[arch](num_classes=n_cls)
    a2.load_weights(model, weight_name=weight_name)
    return model

def main(cfg):
    seed_all(SEED)
    model = load_model(cfg.arch, cfg.weight_name, n_cls=cfg.n_cls)
    sys = ftune.GenericFinetuneSystem(
        cfg = cfg,
        model = model,
        trn_dset = ImageNet('train', subset_classes=CLASS_SUBSET_100, transform=TRAIN_TRANSFORM),
        val_dset = ImageNet('val', subset_classes=CLASS_SUBSET_100),
        tst_dset = ImageNet('test', subset_classes=CLASS_SUBSET_100),
        device = 'cuda'
    )
    sys.run_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run a2mim finetuning')
    parser.add_argument('-c', '--config', type=str,
                        help=f'name of a2mim finetuning config')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not cfg.get('exp_name'):
        cfg.exp_name = 'ftune_'+cfg.weight_name
    main(cfg)
