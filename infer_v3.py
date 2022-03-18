import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
import skimage.io as io
import torch.nn.functional as F
import torchvision.transforms as transforms
from tqdm import tqdm
from typing import List
from loguru import logger
from CAPS.effiUnet_v3_2 import EfficientUNet
from imgaug import augmenters as iaa
from utils.draw_utils import visualize_point


def img_loader(img_file):
    return cv2.imread(img_file, 0)

def img_saver(save_file, img_data):
    cv2.imwrite(save_file, img_data)
    
def npy_saver(save_file, kpt, desc):
    """
    param: save_file: file saving path
    param: kpt      : N x 2
    param: desc     : N x 128
    """
    with open(save_file, 'wb') as output_file:
        np.savez(
            output_file,
            keypoints=kpt,
            scores=[],
            descriptors=desc)


def ratio_preserving_resize(img_data, target_size):
    """
    :param img_data   : (h, w), gray raw img
    :param target_size: (h, w)

    return sized_data, ratio
    :param sized_data: is equal to target_size
    :param ratio     : is the resize ratio
    """
    s_h, s_w = img_data.shape
    d_h, d_w = target_size
    r_h = -1
    r_w = -1
    ratio = -1

    h_ratio = float(d_h)/s_h
    w_ratio = float(d_w)/s_w
    sized_data = np.zeros((d_h, d_w), dtype=img_data.dtype)
    
    if h_ratio >= 1 and w_ratio >= 1:
        sized_data[:s_h, :s_w] = img_data
        ratio = 1
    else:
        if h_ratio > w_ratio:
            r_w = d_w
            r_h = round(s_h * w_ratio)
            ratio = w_ratio
        else:
            r_h = d_h
            r_w = round(s_w * h_ratio)
            ratio = h_ratio
        
        temp_img = cv2.resize(img_data, (r_w, r_h))
        sized_data[:r_h, :r_w] = temp_img
    return sized_data, ratio, (r_h, r_w)


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
    r_shape = [sh, sw]
    return d_img, 1.0, r_shape 
    

def preprocess(img_data, target_size):
    """
    :param img_data: (h, w), gray raw img, nparray
    
    return
        norm_img: torch tensor:(1, 1, h, w)
        ratio   : float
        r_shape : list, (h, w)
    """
    sized_img, ratio, r_shape = ratio_preserving_resize(img_data, target_size)
    norm_img = sized_img / 255.0
    tensor_img = torch.from_numpy(norm_img.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
    return tensor_img, ratio, r_shape


def hpatches_reader(imdir):
    imfs = list()
    for f in os.listdir(imdir):
        scene_dir = os.path.join(imdir, f)
        imfs.extend([os.path.join(scene_dir, '{}.ppm').format(ind) for ind in range(1, 7)])
    return imfs


class LFNetInfer(object):
    def __init__(self, pth_path, in_w=640, in_h=384, edge_filter_width=30, point_thresh=0.015) -> None:
        self.base_step = 16
        self.edge_filter_width = edge_filter_width
        self.point_thresh = point_thresh
        self.pth_path = pth_path
        self.in_w = in_w
        self.in_h = in_h
        # parameter checking
        self.parameter_check()
        # init model
        self.model = EfficientUNet()
        self.model.load_state_dict(torch.load(pth_path))
        self.model.cuda()
        self.model.eval()

    
    def parameter_check(self):
        assert self.in_w % self.base_step == 0, "in_w is not multiple of {}".format(self.base_step)
        assert self.in_h % self.base_step == 0, "in_h is not multiple of {}".format(self.base_step)
        assert os.path.exists(self.pth_path), "pth file: {} not exist".format(self.pth_path)


    def preprocess(self, img_data):
        """
        :param img_data: (h, w), gray raw img, nparray
        
        return
            norm_img: torch tensor:(1, 1, h, w)
            ratio   : float
            r_shape : list, (h, w)
        """
        sized_img, ratio, r_shape = ratio_preserving_resize(img_data, (self.in_h, self.in_w))
        norm_img = sized_img / 255.0
        tensor_img = torch.from_numpy(norm_img.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
        return tensor_img, ratio, r_shape
    
    def preprocess_v2(self, img_data):
        sized_img, ratio, r_shape = pic_rb_pading(img_data, )
        norm_img = sized_img / 255.0
        tensor_img = torch.from_numpy(norm_img.astype(np.float32)).unsqueeze(dim=0).unsqueeze(dim=0)
        return tensor_img, ratio, r_shape
    
    def post_process(self, prob_nms, featmap, tensor_img, ratio, r_shape):
        """
        :param prob_nms: tensor: B, 1, H, W
        :param featmap : tensor: B, 128, H/4, W/4
        
        return:
            point     : list[tensor[npt, 2]]
            descriptor: list[tenosr[1, npt, 128]]
        """
        prob_nms = prob_nms.squeeze(dim=1)   # B, H, W
        # print("prob_nms shape: {}".format(prob_nms.shape))
        # filter edget point
        filter_width = self.edge_filter_width
        prob_nms[:, :filter_width, :] = -1
        prob_nms[:, :, :filter_width] = -1
        prob_nms[:, r_shape[0] - filter_width:, :] = -1
        prob_nms[:, :, r_shape[1] - filter_width:] = -1
        # extract point
        points = [torch.stack(torch.where(pred > self.point_thresh)).T for pred in prob_nms] # list[npt, 2]
        points = [torch.flip(element, dims=[1]) for element in points]  # list[npt, 2]
        # extract descriptor
        descriptors = list()
        for batch_idx in range(len(points)):
            coord_n = self.normalize(points[batch_idx], self.in_h, self.in_w)
            descr_n = self.sample_feat_by_coord(featmap[batch_idx: batch_idx+1], coord_n.unsqueeze(dim=0))
            # descr_n: list[1, npt, 128] 
            descriptors.append(descr_n)
        # rescal point to source image scale
        points = [element / ratio for element in points]  # list[npt, 2]
        return points, descriptors     
    
    @staticmethod
    def normalize(coord, h, w):
        '''
        turn the coordinates from pixel indices to the range of [-1, 1]
        :param coord: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: the normalized coordinates [..., 2]
        '''
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord.device).float()
        coord_norm = (coord - c) / c
        return coord_norm
    
    @staticmethod
    def sample_feat_by_coord(x, coord_n, norm=False):
        '''
        sample from normalized coordinates
        :param x: feature map [batch_size, n_dim, h, w]
        :param coord_n: normalized coordinates, [batch_size, n_pts, 2]
        :param norm: if l2 normalize features
        :return: the extracted features, [batch_size, n_pts, n_dim]
        '''
        feat = F.grid_sample(x, coord_n.unsqueeze(2), align_corners=True).squeeze(-1)
        if norm:
            feat = F.normalize(feat)
        feat = feat.transpose(1, 2)
        return feat
    
    def vis_point(img, pts):
        pass
    
    def infer(self, img_file):
        raw_data =  img_loader(img_file=img_file)  # h, w
        tensor_img, ratio, r_shape = self.preprocess(raw_data)
        with torch.no_grad():
            # model infer
            input_tensor = tensor_img.cuda()
            prob_nms, featmap = self.model(input_tensor)
            points, descriptors = self.post_process(prob_nms, featmap, tensor_img, ratio, r_shape)
            return points, descriptors
        
        
    def run(self, data_path, save_path, ext):
        img_file_lst = hpatches_reader(data_path)
        
        for img_f in tqdm(img_file_lst):
            pts_t, des_t = self.infer(img_f)
            save_folder = os.path.join(save_path, os.path.basename(os.path.dirname(img_f)))
            os.makedirs(save_folder, exist_ok=True)
            save_file = os.path.join(save_folder, "{}.{}".format(os.path.basename(img_f), ext))
            kpt = pts_t[0].cpu().numpy()
            desc = des_t[0].squeeze(0).detach().cpu().numpy()
            npy_saver(save_file, kpt, desc)

   
   
class LFNet(nn.Module):
    def __init__(self, base_pth, in_w=640, in_h=384, edge_filter_width=30, point_thresh=0.015,
                 tork=2000) -> None:
        super(LFNet, self).__init__()
        self.edge_filter_width = edge_filter_width
        self.point_thresh = point_thresh
        self.in_w = in_w
        self.in_h = in_h
        self.scale_cood2feat = 4
        self.topk = tork
        self.base_net = EfficientUNet()
        self.base_net.load_state_dict(torch.load(base_pth))
        self.pts_pad = torch.zeros(size=(self.topk, 2), dtype=torch.int64, device=torch.device("cuda"))
        self.des_pad = torch.zeros(size=(self.topk, 128), dtype=torch.float, device=torch.device("cuda"))
    
    @staticmethod
    def bilinear_grid_sample(im, coord_pts, scale_cood2feat):
        """
        param: im     : [h, w, ndim]     
        param: coord_n: [n_pts, 2] 
        """
        coord_feat = torch.div(coord_pts, scale_cood2feat)
        x = coord_feat[:, 0]
        y = coord_feat[:, 1]
        
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        x1 = x0 + 1
        y1 = y0 + 1
        
        wa = ((x1 - x) * (y1 - y)).unsqueeze(1)
        wb = ((x1 - x) * (y - y0)).unsqueeze(1)
        wc = ((x - x0) * (y1 - y)).unsqueeze(1)
        wd = ((x - x0) * (y - y0)).unsqueeze(1)

        Ia = im[y0, x0]
        Ib = im[y1, x0]
        Ic = im[y0, x1]
        Id = im[y1, x1]

        feat = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return feat        
        
    def post_process(self, 
                     prob_nms: torch.tensor, 
                     featmap: torch.tensor,
                     ratio: torch.tensor, 
                     r_shape: torch.tensor):
        """
        :param prob_nms: tensor: B, 1, H, W
        :param featmap : tensor: B, 128, H/4, W/4
        :param ratios: tensor: B, 1
        :param r_shape: tensor: B, 2(h, w)
        
        return:
            point     : list[tensor[npt, 2]]
            descriptor: list[tenosr[1, npt, 128]]
        """
        # extract point
        prob_nms = prob_nms.squeeze(dim=1)   # B, H, W
        # filter edget point
        batch_size = prob_nms.shape[0]
        filter_width = self.edge_filter_width
           
        pts_lst = list()
        des_lst = list()
        featmap = featmap.permute(0, 2, 3, 1)  # B, H/4, W/4, 128
        for batch_idx in range(batch_size):
            # point extract
            cur_rshape = r_shape[batch_idx]
            prob_nms[batch_idx, :filter_width, :] = -1
            prob_nms[batch_idx, :, :filter_width] = -1
            prob_nms[batch_idx, cur_rshape[0] - filter_width:, :] = -1
            prob_nms[batch_idx, :, cur_rshape[1] - filter_width:] = -1
            pts = torch.stack(torch.where(prob_nms[batch_idx] > self.point_thresh)).transpose(0, 1) # list[npt, 2]
            pts = torch.flip(pts, dims=[1]) # list[npt, 2]
            # descriptor extract
            des = self.bilinear_grid_sample(featmap[batch_idx], pts, self.scale_cood2feat)
            # rescal point to source image scale
            pts = pts / ratio[batch_idx]
            # keep the tok result
            p_num = pts.shape[0]
            if p_num < self.topk:
                pad_num = self.topk - p_num
                pts = torch.cat([pts, self.pts_pad[0: pad_num]], dim=0)
                des = torch.cat([des, self.des_pad[0: pad_num]], dim=0)
            else:
                pts = pts[: self.topk]
                des = des[: self.topk]
            
            pts_lst.append(pts)
            des_lst.append(des)
        
        pts_t = torch.stack(pts_lst, dim=0)
        des_t = torch.stack(des_lst, dim=0)
        
        return pts_t, des_t
        
    def forward(self, x, ratio, r_shape):
        prob_nms, featmap = self.base_net(x)
        pts_t, des_t = self.post_process(prob_nms, featmap, ratio, r_shape)
        return pts_t, des_t     


class LFNetInferV2(object):
    def __init__(self, base_pth, in_w=640, in_h=384, edge_filter_width=30, point_thresh=0.015) -> None:
        self.lf_net = LFNet(base_pth=base_pth,
                            in_w=in_w,
                            in_h=in_h,
                            edge_filter_width=edge_filter_width,
                            point_thresh=point_thresh)
        self.lf_net.cuda()
        self.lf_net.eval()

    def run(self, data_path, save_path, ext):
        img_file_lst = hpatches_reader(data_path)
        
        for img_idx, img_f in tqdm(enumerate(img_file_lst)):
            # print(img_f)
            pts_t, des_t = self.infer(img_f)
            save_folder = os.path.join(save_path, os.path.basename(os.path.dirname(img_f)))
            os.makedirs(save_folder, exist_ok=True)
            save_file = os.path.join(save_folder, "{}.{}".format(os.path.basename(img_f), ext))
            kpt = pts_t.squeeze(0).detach().cpu().numpy()
            desc = des_t.squeeze(0).detach().cpu().numpy()
            # print(kpt[:10])
            slice_idx = 2000
            for idx, value in enumerate(kpt):
                if value[0] < 1:
                    slice_idx = idx
                    break
                
            kpt = kpt[:slice_idx]
            desc = desc[:slice_idx]
                
            # print(kpt.shape)
            # print(desc.shape)
            npy_saver(save_file, kpt, desc)
            
    
    def infer(self, img_file):
        with torch.no_grad():
            raw_data =  img_loader(img_file=img_file)  # h, w
            tensor_img, ratio, r_shape = preprocess(raw_data, (self.lf_net.in_h, self.lf_net.in_w))
            tensor_img = tensor_img.cuda()
            ratio = torch.tensor([ratio,]).cuda()
            r_shape = torch.tensor([r_shape,]).cuda()
            pts_t, des_t = self.lf_net(tensor_img, ratio, r_shape)
            return pts_t, des_t
        


if __name__ == '__main__':

    pth_path = "/home/dm/work/02.workspace/caps/out/train_caps_magic_v3_2/super_point_200000.pth"
    in_w = 960
    in_h = 960
    filter_width = 30
    point_thresh = 0.015
    img_file = "/home/dm/work/04.dataset/youfang/back/608.png"
    # save_path = "/home/dm/work/02.workspace/caps/out/debug"
    # save_file = os.path.join(save_path, os.path.basename(img_file))
    data_path = '/home/dm/work/04.dataset/hpatches-sequences/hpatches-sequences-release'
    save_path_v1 = "/home/dm/work/02.workspace/caps/out/extract_testv1"

    lfnet_ins = LFNetInfer(pth_path=pth_path,
                           in_w=in_w,
                           in_h=in_h)
    lfnet_ins.run(data_path=data_path, save_path=save_path_v1, ext="magic_testv1")


    
    # for batch_idx in range(len(points)):
    #     logger.info("batch idx: points num: {}, descrpoint num:{}".format(
    #         points[batch_idx].shape, descriptors[batch_idx].shape))
    #     img_data = cv2.imread(img_file)
    #     img_data = visualize_point(img_data, points[batch_idx])
    #     img_saver(save_file, img_data)


    # save_path_v2 = "/home/dm/work/02.workspace/caps/out/extract_testv2"
    # lfnetv2_ins = LFNetInferV2(base_pth=pth_path,
    #                            in_w=in_w,
    #                            in_h=in_h,
    #                            edge_filter_width=filter_width,
    #                            point_thresh=point_thresh)
    # lfnetv2_ins.run(data_path=data_path, save_path=save_path_v2, ext="magic_testv2")
    
        
    
    
    
    batch_size = 10
    device = torch.device("cuda:0")
    
    data = torch.randn([batch_size, 1, in_h, in_w], device=device)
    ratio = torch.tensor([0.6]*batch_size, dtype=torch.float, device=device)
    r_shape = torch.tensor([448, 450]*batch_size, dtype=torch.int, device=device)
    with torch.no_grad():
        lfnet(data, ratio, r_shape)
    onnx_model_path = "./lfnet.onnx"
    torch.onnx.export(lfnet,
                      args=(data, ratio, r_shape),
                      f=onnx_model_path,
                      input_names=["data", "ratio", "r_shape"],
                      output_names=["pts", "des"],
                      dynamic_axes=None,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=True)
    
 



