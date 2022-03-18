import torch
from torch.utils.data import Dataset
import os
import numpy as np
# import skimage.io as io
import cv2
import torchvision.transforms as transforms
import collections
from tqdm import tqdm
import dataloader.data_utils as data_utils
import utils.draw_utils as draw_utils


rand = np.random.RandomState(234)



def generate_gray_kpts(gray_img, mode, num_pts, h, w):
    # generate candidate query points
    if mode == 'random':
        kp1_x = np.random.rand(num_pts) * (w - 1)
        kp1_y = np.random.rand(num_pts) * (h - 1)
        coord = np.stack((kp1_x, kp1_y)).T

    elif mode == 'sift':
        sift = cv2.SIFT_create(nfeatures=num_pts)
        kp1 = sift.detect(gray_img)
        coord = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1])

    elif mode == 'mixed':
        kp1_x = np.random.rand(1 * int(0.1 * num_pts)) * (w - 1)
        kp1_y = np.random.rand(1 * int(0.1 * num_pts)) * (h - 1)
        kp1_rand = np.stack((kp1_x, kp1_y)).T

        sift = cv2.SIFT_create(nfeatures=int(0.9 * num_pts))
        kp1_sift = sift.detect(gray_img)
        kp1_sift = np.array([[kp.pt[0], kp.pt[1]] for kp in kp1_sift])
        if len(kp1_sift) == 0:
            coord = kp1_rand
        else:
            coord = np.concatenate((kp1_rand, kp1_sift), 0)

    else:
        raise Exception('unknown type of keypoints')

    return coord


class MegaDepthLoader(object):
    def __init__(self, args):
        self.args = args
        self.dataset = MegaDepth(args)
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=args.workers, collate_fn=self.my_collate)

    def my_collate(self, batch):
        ''' Puts each data field into a tensor with outer dimension batch size '''
        batch = list(filter(lambda b: b is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)

    def load_data(self):
        return self.data_loader

    def name(self):
        return 'MegaDepthLoader'

    def __len__(self):
        return len(self.dataset)



def load_image(img_path):
    return cv2.imread(img_path)

class MegaDepth(Dataset):
    def __init__(self, args):
        self.args = args
        if args.phase == 'train':

        #     self.transform = transforms.Compose([transforms.ToPILImage(),
        #                                          transforms.RandomChoice([
        #                                              transforms.RandomApply([transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.4),], p=0.7),
        #                                              transforms.RandomAutocontrast()]),
        #                                          transforms.RandomGrayscale(p=0.5),
        #                                          transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3)),], p=0.3),
        #                                          transforms.ToTensor(),
        #                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                                                               std=(0.229, 0.224, 0.225)),
        #                                          ])

        # else:
        #     self.transform = transforms.Compose([transforms.ToTensor(),
        #                                          transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #                                                               std=(0.229, 0.224, 0.225)),
        #                                          ])

            self.transform = transforms.Compose([transforms.ToPILImage(),
                                                 transforms.RandomChoice([
                                                     transforms.RandomApply([transforms.ColorJitter(brightness=1, contrast=1, saturation=1, hue=0.4),], p=0.7),
                                                     transforms.RandomAutocontrast()]),
                                                 transforms.RandomGrayscale(p=0.5),
                                                 transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 3)),], p=0.3),
                                                 transforms.ToTensor()
                                                 ])

        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
        self.phase = args.phase
        self.root = os.path.join(args.datadir, self.phase)
        self.images = self.read_img_cam()
        self.imf1s, self.imf2s = self.read_pairs()
        print('total number of image pairs loaded: {}'.format(len(self.imf1s)))
        # shuffle data
        index = np.arange(len(self.imf1s))
        rand.shuffle(index)
        self.imf1s = list(np.array(self.imf1s)[index])
        self.imf2s = list(np.array(self.imf2s)[index])

    def read_img_cam(self):
        images = {}
        Image = collections.namedtuple(
            "Image", ["name", "w", "h", "fx", "fy", "cx", "cy", "rvec", "tvec"])
        for scene_id in os.listdir(self.root):
            densefs = [f for f in os.listdir(os.path.join(self.root, scene_id))
                       if 'dense' in f and os.path.isdir(os.path.join(self.root, scene_id, f))]
            for densef in densefs:
                folder = os.path.join(self.root, scene_id, densef, 'aligned')
                img_cam_txt_path = os.path.join(folder, 'img_cam.txt')
                with open(img_cam_txt_path, "r") as fid:
                    while True:
                        line = fid.readline()
                        if not line:
                            break
                        line = line.strip()
                        if len(line) > 0 and line[0] != "#":
                            elems = line.split()
                            image_name = elems[0]
                            img_path = os.path.join(folder, 'images', image_name)
                            w, h = int(elems[1]), int(elems[2])
                            fx, fy = float(elems[3]), float(elems[4])
                            cx, cy = float(elems[5]), float(elems[6])
                            R = np.array(elems[7:16])
                            T = np.array(elems[16:19])
                            images[img_path] = Image(
                                name=image_name, w=w, h=h, fx=fx, fy=fy, cx=cx, cy=cy, rvec=R, tvec=T
                            )
        return images

    def read_pairs(self):
        imf1s, imf2s = [], []
        print('reading image pairs from {}...'.format(self.root))
        for scene_id in tqdm(os.listdir(self.root), desc='# loading data from scene folders'):
            densefs = [f for f in os.listdir(os.path.join(self.root, scene_id))
                       if 'dense' in f and os.path.isdir(os.path.join(self.root, scene_id, f))]
            for densef in densefs:
                imf1s_ = []
                imf2s_ = []
                folder = os.path.join(self.root, scene_id, densef, 'aligned')
                pairf = os.path.join(folder, 'pairs.txt')

                if os.path.exists(pairf):
                    f = open(pairf, 'r')
                    for line in f:
                        imf1, imf2 = line.strip().split(' ')
                        imf1s_.append(os.path.join(folder, 'images', imf1))
                        imf2s_.append(os.path.join(folder, 'images', imf2))

                # make # image pairs per scene more balanced
                if len(imf1s_) > 5000:
                    index = np.arange(len(imf1s_))
                    rand.shuffle(index)
                    imf1s_ = list(np.array(imf1s_)[index[:5000]])
                    imf2s_ = list(np.array(imf2s_)[index[:5000]])

                imf1s.extend(imf1s_)
                imf2s.extend(imf2s_)

        return imf1s, imf2s

    @staticmethod
    def get_intrinsics(im_meta):
        return np.array([[im_meta.fx, 0, im_meta.cx],
                         [0, im_meta.fy, im_meta.cy],
                         [0, 0, 1]])

    @staticmethod
    def get_extrinsics(im_meta):
        R = im_meta.rvec.reshape(3, 3)
        t = im_meta.tvec
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = R
        extrinsic[:3, 3] = t
        return extrinsic

    def __getitem__(self, item):
        imf1 = self.imf1s[item]
        imf2 = self.imf2s[item]
        im1_meta = self.images[imf1]
        im2_meta = self.images[imf2]
        # im1 = io.imread(imf1)
        # im2 = io.imread(imf2)
        # modify : change io to cv2. and read gray image 
        im1 = cv2.imread(imf1, 0)
        im2 = cv2.imread(imf2, 0)
        h, w = im1.shape
        # h, w = im1.shape[:2]

        intrinsic1 = self.get_intrinsics(im1_meta)
        intrinsic2 = self.get_intrinsics(im2_meta)

        extrinsic1 = self.get_extrinsics(im1_meta)
        extrinsic2 = self.get_extrinsics(im2_meta)

        relative = extrinsic2.dot(np.linalg.inv(extrinsic1))
        R = relative[:3, :3]
        # remove pairs that have a relative rotation angle larger than 80 degrees
        theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) * 180 / np.pi
        if theta > 80 and self.phase == 'train':
            return None

        T = relative[:3, 3]
        tx = data_utils.skew(T)
        E_gt = np.dot(tx, R)
        F_gt = np.linalg.inv(intrinsic2).T.dot(E_gt).dot(np.linalg.inv(intrinsic1))

        # modify
        # generate candidate query points
        # coord1 = data_utils.generate_query_kpts(im1, self.args.train_kp, 10*self.args.num_pts, h, w)
        coord1 = generate_gray_kpts(im1, self.args.train_kp, 10*self.args.num_pts, h, w)

        # if no keypoints are detected
        if len(coord1) == 0:
            return None

        # prune query keypoints that are not likely to have correspondence in the other image
        if self.args.prune_kp:
            ind_intersect = data_utils.prune_kpts(coord1, F_gt, im2.shape[:2], intrinsic1, intrinsic2,
                                                  relative, d_min=4, d_max=400)
            if np.sum(ind_intersect) == 0:
                return None
            coord1 = coord1[ind_intersect]

        coord1 = draw_utils.random_choice(coord1, self.args.num_pts)
        coord1 = torch.from_numpy(coord1).float()

        im1_ori, im2_ori = torch.from_numpy(im1), torch.from_numpy(im2)

        F_gt = torch.from_numpy(F_gt).float() / (F_gt[-1, -1] + 1e-10)
        intrinsic1 = torch.from_numpy(intrinsic1).float()
        intrinsic2 = torch.from_numpy(intrinsic2).float()
        pose = torch.from_numpy(relative[:3, :]).float()
        im1_tensor = self.transform(im1)
        im2_tensor = self.transform(im2)

        out = {'im1': im1_tensor,
               'im2': im2_tensor,
               'im1_ori': im1_ori,
               'im2_ori': im2_ori,
               'pose': pose,
               'F': F_gt,
               'intrinsic1': intrinsic1,
               'intrinsic2': intrinsic2,
               'coord1': coord1}

        return out

    def __len__(self):
        return len(self.imf1s)


if __name__ == "__main__":
    import config
    config = config.get_args()
    dataset = MegaDepth(args=config)

    for idx, data in enumerate(dataset):
        print("idx: {}".format(idx))
        print(data['im1'].shape)