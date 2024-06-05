import systems.a2mim as a2
from models import ARCHITECTURES
from dsets.imagenet import ImageNet, TRAIN_TRANSFORM
from consts import SEED, CLASS_SUBSET_100
from utils import load_config, seed_all
import systems.finetune as ftune

# one item: [layer, patch_size, mask_ratio]
SWEEP = [
    # [2, 4, 0.8] # other instance
    # [4, 32, 0.6], # other instance
    [4, 32, 0.8], # <- need to rerun to get neck back
    # [4, 32, 0.4], # no time
    [3, 56, 0.4],
    [3, 56, 0.6],
    # [3, 56, 0.8], # no time
]

SWEEP_FTUNE = [
    # [4, 32, 0.6], # other instance
    # [4, 32, 0.8], # not interesting/no time
    # [4, 32, 0.4], # no time/never run
    [3, 56, 0.4], # running
    [2, 4, 0.8], # moved over from other instance
    [3, 56, 0.6], # running
    # [3, 56, 0.8], # no time
]

def main_a2mim(cfg):
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

def load_model(arch, weight_name, n_cls=100):
    model = ARCHITECTURES[arch](num_classes=n_cls)
    a2.load_weights(model, weight_name=weight_name)
    return model

def main_ftune(cfg):
    seed_all(SEED)
    for (layer, patch_size, mask_ratio) in SWEEP_FTUNE:
        cfg.exp_name = f"ftune-layer{layer}_patch{patch_size}_ratio{int(mask_ratio*100)}"
        cfg.weight_name = f"layer{layer}_patch{patch_size}_ratio{int(mask_ratio*100)}_full-best"
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
    # base_cfg = load_config('gen_a2mim_res50')
    # main_a2mim(base_cfg)
    base_cfg = load_config('ftune_res50')
    main_ftune(base_cfg)