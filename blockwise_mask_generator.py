import math
import random
import inspect
import warnings
from numbers import Number
from typing import Sequence, Tuple

import mmcv
import numpy as np
from PIL import Image, ImageFilter, ImageOps

import torch
from torchvision import transforms as _transforms
import torchvision.transforms.functional as F

class Registry(object):

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __repr__(self):
        format_str = self.__class__.__name__ + '(name={}, items={})'.format(
            self._name, list(self._module_dict.keys()))
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        return self._module_dict.get(key, None)

    def _register_module(self, module_class, force=False):
        """Register a module.

        Args:
            module (:obj:`nn.Module`): Module to be registered.
        """
        if not inspect.isclass(module_class):
            raise TypeError('module must be a class, but got {}'.format(
                type(module_class)))
        module_name = module_class.__name__
        if not force and module_name in self._module_dict:
            raise KeyError('{} is already registered in {}'.format(
                module_name, self.name))
        self._module_dict[module_name] = module_class

    def register_module(self, cls=None, force=False):
        if cls is None:
            return partial(self.register_module, force=force)
        self._register_module(cls, force=force)
        return cls


PIPELINES = Registry('pipeline')

def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, Image.Image):
        data = np.array(data, dtype=np.uint8)
        if data.ndim < 3:
            data = np.expand_dims(data, axis=-1)
        data = np.rollaxis(data, 2)  # HWC to CHW
        return data
    elif isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to numpy.')


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data).type(torch.float32)
    elif isinstance(data, Image.Image):
        return torchvision.transforms.functional.to_tensor(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


@PIPELINES.register_module()
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
                 input_size=192,
                 mask_patch_size=32,
                 model_patch_size=4,
                 mask_ratio=0.6,
                 mask_only=False,
                 mask_color='zero',
                ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_only = mask_only
        self.mask_color = mask_color
        assert self.mask_color in ['mean', 'zero', 'rand',]
        if self.mask_color != 'zero':
            assert mask_only == False

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img=None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.from_numpy(mask)  # [H, W]

        if self.mask_color == 'mean':
            if isinstance(img, Image.Image):
                img = np.array(img)
                mask_ = to_numpy(mask).reshape((self.rand_size * self.scale, -1, 1))
                mask_ = mask_.repeat(
                    self.model_patch_size, axis=0).repeat(self.model_patch_size, axis=1)
                mean = img.reshape(-1, img.shape[2]).mean(axis=0)
                img = np.where(mask_ == 1, img, mean)
                img = Image.fromarray(img.astype(np.uint8))
            elif isinstance(img, torch.Tensor):
                mask_ = to_tensor(mask)
                mask_ = mask_.repeat_interleave(self.model_patch_size, 0).repeat_interleave(
                    self.model_patch_size, 1).contiguous()
                img = img.clone()
                mean = img.mean(dim=[1,2])
                for i in range(img.size(0)):
                    img[i, mask_ == 1] = mean[i]

        if self.mask_only:
            return mask
        else:
            return img, mask