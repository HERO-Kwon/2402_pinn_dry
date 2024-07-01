# -*- encoding: utf-8 -*-
"""
Desc      :   Transforms.
"""
# File    :   np_transforms.py
# Time    :   2020/04/06 17:24:54
# Author  :   Zweien
# Contact :   278954153@qq.com

import cv2
import torch
from torchvision import transforms
import numpy as np

class ToTensor:
    """Transform np.array to torch.tensor
    Args:
        add_dim (bool, optional): add first dim. Defaults to True.
        type_ (torch.dtype, optional): dtype of the tensor. Defaults to tensor.torch.float32.
    Returns:
        torch.tensor: tensor
    """

    def __init__(self, add_dim=True, type_=torch.float32):
        self.add_dim = add_dim
        self.type = type_

    def __call__(self, x):
        if self.add_dim:
            return torch.tensor(x, dtype=self.type).unsqueeze(0)
        return torch.tensor(x, dtype=self.type)


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, x):
        x_tr = np.transpose(x,(1,2,0))
        x_res = cv2.resize(x_tr, self.size, interpolation=cv2.INTER_NEAREST)
        return np.transpose(x_res,(2,0,1))


class Lambda(transforms.Lambda):
    pass


class Compose(transforms.Compose):
    pass


class Normalize(transforms.Normalize):
    pass
