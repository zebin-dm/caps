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
from imgaug import augmenters as iaa


def ratio_preserving_resize(img, target_size):
    '''
    :param img: raw img
    :param target_size: (h,w)
    :return:
    '''
    scales = np.array((target_size[0]/img.shape[0], target_size[1]/img.shape[1])) ##h_s,w_s

    new_size = np.round(np.array(img.shape)*np.max(scales)).astype(int)#
    temp_img = cv2.resize(img, tuple(new_size[::-1]))
    curr_h, curr_w = temp_img.shape
    target_h, target_w = target_size
    ##
    hp = (target_h-curr_h)//2
    wp = (target_w-curr_w)//2
    aug = iaa.Sequential([iaa.CropAndPad(px=(hp, wp, target_h-curr_h-hp, target_w-curr_w-wp),keep_size=False),])
    new_img = aug(images=temp_img)
    return new_img


def least_common_multiple(src_len, base_len):
    if src_len % base_len != 0:
        dst_len = (src_len // base_len + 1) * base_len
    else:
        dst_len = src_len
    return dst_len

def pic_rb_pading(img, base_len=16):
    sh, sw = img.shape
    dh = least_common_multiple(sh, base_len)
    dw = least_common_multiple(sw, base_len)

    if dh == sh and dw == sw:
        return img
    
    d_img = np.zeros((dh, dw), dtype=img.dtype)
    d_img[:sh, :sw] = img
    return d_img


class HPatchDataset(Dataset):
    def __init__(self, imdir):
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                      transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                                                           std=(0.229, 0.224, 0.225)),
        #                                      ])
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.imfs = []
        for f in os.listdir(imdir):
            scene_dir = os.path.join(imdir, f)
            self.imfs.extend([os.path.join(scene_dir, '{}.ppm').format(ind) for ind in range(1, 7)])

    def __getitem__(self, item):
        imf = self.imfs[item]
        # modify the loader from io.imread to cv2.imread
        # im = io.imread(imf)
        im = cv2.imread(imf, 0)
        src_shape = im.shape
        im = pic_rb_pading(im)
        # print(imf)
        # print("src_img")
        # print(im[100: 110, 100: 110])
        
        im_tensor = self.transform(im)
        # print("tesnor img shape: {}".format(im_tensor.shape))
        # print(im_tensor[0, 100: 110, 100: 110])
        # using sift keypoints
        # sift = cv2.SIFT_create()
        # gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        # kpts = sift.detect(im)
        # kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in kpts])
        # coord = torch.from_numpy(kpts).float()
        # print("ori coord shape: {}".format(coord.shape))
        # out = {'im': im_tensor, 'coord': coord, 'imf': imf}
        # return out
        return im_tensor, src_shape, imf

    def __len__(self):
        return len(self.imfs)


if __name__ == '__main__':
    # example code for extracting features for HPatches dataset, SIFT keypoint is used
    args = config.get_args()
    device = torch.device('cuda:0')

    dataset = HPatchDataset(args.extract_img_dir)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.workers)

    model = CAPSModel(args)

    outdir = args.extract_out_dir
    os.makedirs(outdir, exist_ok=True)

    img_save_path = "/home/dm/work/02.workspace/caps/out/output_img"
    number = 0
    with torch.no_grad():
        for (im_data, src_shape, img_path) in data_loader:
            im_data = im_data.to(device)
            coord, feats = model.extract_det_and_des(im_data, src_shape)
            desc = feats[0].squeeze(0).detach().cpu().numpy()
            kpt = coord[0].cpu().numpy()
 
            save_folder = os.path.join(outdir, os.path.basename(os.path.dirname(img_path[0])))
            os.makedirs(save_folder, exist_ok=True)
            save_file = os.path.join(save_folder, "{}.magicv3_2".format(os.path.basename(img_path[0])))
            print(kpt.shape)
            print(desc.shape)
            break
            with open(save_file, 'wb') as output_file:
                np.savez(
                    output_file,
                    keypoints=kpt,
                    scores=[],
                    descriptors=desc)