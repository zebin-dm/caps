# from scipy.fftpack import ss_diff
# import skimage.io as io
# import cv2
# import torchvision.transforms as transforms
# import numpy as np


# img_path = "/home/dm/work/04.dataset/youfang/back/580.png"
# # image = io.imread(img_path)
# # print("scipy shape: {}".format(image.shape))

# cv_img = cv2.imread(img_path, 0)
# print("cv shape: {}".format(cv_img.shape))

# trans = transforms.ToPILImage()
# c_trans = transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.4)
# pil_img = trans(cv_img)
# print("pil shape: {}".format(pil_img.size))

# tsr_trans = transforms.Compose([transforms.ToTensor()])
# tsr_img = tsr_trans(cv_img)
# print("tsr shape: {}".format(tsr_img.shape))
# print(tsr_img[:, :5, :5])
# print(cv_img[:5, :5])

# np_img = np.array(pil_img)
# print("np shape: {}".format(np_img.shape))


# # print(np_img[0: 10, 0: 10, :] - cv_img[0: 10, 0:10, :])


# import numpy as np
# data = np.random.rand(30, 440)
# print(data.shape)

# data = data[np.newaxis,]
# print(data.shape)

import torch

data = torch.randn(4, 4, 3)
point = torch.tensor([[0, 1], [0, 2]])
print(data)

px = point[:, 0]
py = point[:, 1]
print(px)
print(py)

feat = data[px, py]
print(feat)