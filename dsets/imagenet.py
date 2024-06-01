from torch.utils.data import Dataset
import torchvision.transforms as tvt
from datasets import load_dataset
from datasets import Image as HfImage
from PIL import Image
from io import BytesIO
from easydict import EasyDict
import numpy as np

from consts import RESNET_INP_DIM

DSET_NAME = 'imagenet-1k'

imagenet_normalize = tvt.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
DEFAULT_TRANSFORM = tvt.Compose([
    tvt.Resize(256),
    tvt.CenterCrop(RESNET_INP_DIM),
    tvt.ToTensor(),
    imagenet_normalize
])

class ImageNet(Dataset):
    def __init__(self, split, transform=DEFAULT_TRANSFORM, subset_classes=None):
        hf_split = {
            'train': 'train',
            'val': 'validation',
            'test': 'test'
        }[split] # convert shortened split names to huggingface names
        self.base_ds = load_dataset(
            DSET_NAME, trust_remote_code=True, split=hf_split
        ).cast_column('image', HfImage(decode=False, mode=''))
        self.subset_classes = subset_classes
        if subset_classes is not None:
            self.class_mapping = {orig_class_id:idx for idx, orig_class_id in enumerate(subset_classes)}
            self.base_ds = self.base_ds.filter(lambda x: self.class_mapping.get(x['label']) is not None, 
                                               load_from_cache_file=True)
        self.transform = transform
    
    def __len__(self):
        return len(self.base_ds)
    
    def __getitem__(self, i):
        base_data = self.base_ds[i]
        img_dict, lbl = base_data['image'], base_data['label']
        if self.subset_classes is not None:
            lbl = self.class_mapping[lbl]
        img = Image.open(BytesIO(img_dict['bytes'])).convert('RGB')
        img.load()
        if self.transform is not None:
            img = self.transform(img)
        return EasyDict(x=img, y=lbl, path=img_dict['path'])
