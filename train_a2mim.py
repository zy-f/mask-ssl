import systems.a2mim as a2
from models.resnet import resnet50, resnet18
from dsets.imagenet import ImageNet
import torchvision.transforms as tvt
from easydict import EasyDict
from consts import SEED, RESNET_INP_DIM, CLASS_SUBSET_100

cfg = EasyDict(
    exp_name = 'res18_layer4_patch32',
    mask_cfg = {},
    mask_gen_cfg = dict(
        mask_patch_size=32,
        mask_ratio=0.6,
        mask_color='mean',
        seed=SEED
    ),
    mask_layer = 4,
    dl_kwargs = dict(
        num_workers = 8,
        batch_size = 256
    ),
    hparams = dict(n_epochs = 20)
)

imagenet_normalize = tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = tvt.Compose([
    tvt.Resize(256),
    tvt.CenterCrop(RESNET_INP_DIM),
    tvt.ToTensor(),
    imagenet_normalize, 
    tvt.RandomResizedCrop(RESNET_INP_DIM, scale=(0.67, 1.0), ratio=(3/4, 4/3)),
    tvt.RandomHorizontalFlip()
])

sys = a2.A2MIMSystem(
    cfg=cfg,
    model=resnet18(),
    trn_dset=ImageNet('train', subset_classes=CLASS_SUBSET_100, transform=train_transform),
    val_dset=ImageNet('val', subset_classes=CLASS_SUBSET_100),
    device='cuda'
)

sys.run_training()