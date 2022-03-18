import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import skimage.io as io
import torchvision.transforms as transforms
import config
from tqdm import tqdm
from CAPS.caps_model_v3 import CAPSModel

if __name__ == '__main__':
    args = config.get_args()
    model = CAPSModel(args)
    super_net = model.model.net
    save_path = "/home/dm/work/02.workspace/caps/out/train_caps_magic_v3_2/super_point_200000.pth"
    torch.save(super_net.state_dict(), save_path)
