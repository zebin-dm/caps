import torch
import torch.nn as nn
import torch.nn.functional as F
# from CAPS.effiUnet_v3 import EfficientUNet
from loguru import logger
# from CAPS.effiUnet_v3_1 import EfficientUNet
from CAPS.effiUnet_v4 import EfficientUNet


class CAPSNet(nn.Module):
    def __init__(self, args, device):
        super(CAPSNet, self).__init__()
        self.args = args
        self.device = device
        self.net = EfficientUNet()
        if args.phase == "train":
            if not args.magic_pretrain:
                raise Exception("args.magic_pretrain should not be none in traing mode")
            magic_net_model_dict = torch.load(args.magic_pretrain)
            self.net.magic_net.load_state_dict(magic_net_model_dict)
        self.net.to(device)

        for param in self.net.magic_net.parameters():
            param.requires_grad = False

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
    def denormalize(coord_norm, h, w):
        '''
        turn the coordinates from normalized value ([-1, 1]) to actual pixel indices
        :param coord_norm: [..., 2]
        :param h: the image height
        :param w: the image width
        :return: actual pixel coordinates
        '''
        c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).to(coord_norm.device)
        coord = coord_norm * c + c
        return coord

    def ind2coord(self, ind, width):
        ind = ind.unsqueeze(-1)
        x = ind % width
        # y = ind // width
        y = torch.div(ind, width, rounding_mode='floor')
        coord = torch.cat((x, y), -1).float()
        return coord

    def gen_grid(self, h_min, h_max, w_min, w_max, len_h, len_w):
        x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w), torch.linspace(h_min, h_max, len_h)])
        grid = torch.stack((x, y), -1).transpose(0, 1).reshape(-1, 2).float().to(self.device)
        return grid

    def sample_feat_by_coord(self, x, coord_n, norm=False):
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

    def compute_prob(self, feat1, feat2):
        '''
        compute probability
        :param feat1: query features, [batch_size, m, n_dim]
        :param feat2: reference features, [batch_size, n, n_dim]
        :return: probability, [batch_size, m, n]
        '''
        assert self.args.prob_from in ['correlation', 'distance']
        if self.args.prob_from == 'correlation':
            sim = feat1.bmm(feat2.transpose(1, 2))
            prob = F.softmax(sim, dim=-1)  # Bxmxn
        else:
            dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
                   torch.sum(feat2**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * feat1.bmm(feat2.transpose(1, 2))
            prob = F.softmax(-dist, dim=-1)  # Bxmxn
        return prob

    def get_1nn_coord(self, feat1, featmap2):
        '''
        find the coordinates of nearest neighbor match
        :param feat1: query features, [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the other image
        :return: normalized correspondence locations [batch_size, n_pts, 2]
        '''
        batch_size, d, h, w = featmap2.shape
        feat2_flatten = featmap2.reshape(batch_size, d, h*w).transpose(1, 2)  # Bx(hw)xd

        assert self.args.prob_from in ['correlation', 'distance']
        if self.args.prob_from == 'correlation':
            sim = feat1.bmm(feat2_flatten.transpose(1, 2))
            ind2_1nn = torch.max(sim, dim=-1)[1]
        else:
            dist = torch.sum(feat1**2, dim=-1, keepdim=True) + \
                   torch.sum(feat2_flatten**2, dim=-1, keepdim=True).transpose(1, 2) - \
                   2 * feat1.bmm(feat2_flatten.transpose(1, 2))
            ind2_1nn = torch.min(dist, dim=-1)[1]

        coord2 = self.ind2coord(ind2_1nn, w)
        coord2_n = self.normalize(coord2, h, w)
        return coord2_n

    def get_expected_correspondence_locs(self, feat1, featmap2, with_std=False):
        '''
        compute the expected correspondence locations
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations [batch_size, n_pts, 2]
        '''
        B, d, h2, w2 = featmap2.size()
        grid_n = self.gen_grid(-1, 1, -1, 1, h2, w2)
        featmap2_flatten = featmap2.reshape(B, d, h2*w2).transpose(1, 2)  # BX(hw)xd
        prob = self.compute_prob(feat1, featmap2_flatten)  # Bxnx(hw)

        grid_n = grid_n.unsqueeze(0).unsqueeze(0)  # 1x1x(hw)x2
        expected_coord_n = torch.sum(grid_n * prob.unsqueeze(-1), dim=2)  # Bxnx2

        if with_std:
            # convert to normalized scale [-1, 1]
            var = torch.sum(grid_n**2 * prob.unsqueeze(-1), dim=2) - expected_coord_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord_n, std
        else:
            return expected_coord_n

    def get_expected_correspondence_within_window(self, feat1, featmap2, coord2_n, with_std=False):
        '''
        :param feat1: the feature vectors of query points [batch_size, n_pts, n_dim]
        :param featmap2: the feature maps of the reference image [batch_size, n_dim, h, w]
        :param coord2_n: normalized center locations [batch_size, n_pts, 2]
        :param with_std: if return the standard deviation
        :return: the normalized expected correspondence locations, [batch_size, n_pts, 2], optionally with std
        '''
        batch_size, n_dim, h2, w2 = featmap2.shape
        n_pts = coord2_n.shape[1]
        grid_n = self.gen_grid(h_min=-self.args.window_size, h_max=self.args.window_size,
                               w_min=-self.args.window_size, w_max=self.args.window_size,
                               len_h=int(self.args.window_size*h2), len_w=int(self.args.window_size*w2))

        grid_n_ = grid_n.repeat(batch_size, 1, 1, 1)  # Bx1xhwx2
        coord2_n_grid = coord2_n.unsqueeze(-2) + grid_n_  # Bxnxhwx2
        feat2_win = F.grid_sample(featmap2, coord2_n_grid, padding_mode='zeros', align_corners=True).permute(0, 2, 3, 1)  # Bxnxhwxd

        feat1 = feat1.unsqueeze(-2)

        prob = self.compute_prob(feat1.reshape(batch_size*n_pts, -1, n_dim),
                                 feat2_win.reshape(batch_size*n_pts, -1, n_dim)).reshape(batch_size, n_pts, -1)

        expected_coord2_n = torch.sum(coord2_n_grid * prob.unsqueeze(-1), dim=2)  # Bxnx2

        if with_std:
            var = torch.sum(coord2_n_grid**2 * prob.unsqueeze(-1), dim=2) - expected_coord2_n**2  # Bxnx2
            std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # Bxn
            return expected_coord2_n, std
        else:
            return expected_coord2_n

    def forward(self, im1, im2, coord1):
        # extract features for both images
        # modify the output
        # xf1 = self.net(im1)
        # xf2 = self.net(im2)
        prob_nms1, xf1 = self.net(im1)
        prob_nms2, xf2 = self.net(im2)

        # image width and height
        h1i, w1i = im1.size()[2:]
        h2i, w2i = im2.size()[2:]

        # normalize coordination
        coord1_n = self.normalize(coord1, h1i, w1i)

        # the center locations  of the local window for fine level computation
        feat1_fine = self.sample_feat_by_coord(xf1, coord1_n)  # Bxnxd
        coord2_ef_n, std_2f = self.get_expected_correspondence_locs(feat1_fine, xf2, with_std=True)

        feat2_fine = self.sample_feat_by_coord(xf2, coord2_ef_n)  # Bxnxd
        coord1_ef_n, std_1f = self.get_expected_correspondence_locs(feat2_fine, xf1, with_std=True)
  
        coord2_ef = self.denormalize(coord2_ef_n, h2i, w2i)
        coord1_ef = self.denormalize(coord1_ef_n, h1i, w1i)

        return {'coord2_ef': coord2_ef, 'coord1_ef': coord1_ef,
                'std_1f': std_1f, 'std_2f': std_2f}

    def extract_features(self, im, coord):
        '''
        extract coarse and fine level features given the input image and 2d locations
        :param im: [batch_size, 3, h, w]
        :param coord: [batch_size, n_pts, 2]
        :return: coarse features [batch_size, n_pts, coarse_feat_dim] and fine features [batch_size, n_pts, fine_feat_dim]
        '''
        xf = self.net(im)
        hi, wi = im.size()[2:]
        coord_n = self.normalize(coord, hi, wi)
        feat_f = self.sample_feat_by_coord(xf, coord_n)
        return feat_f

    def exetrct_det_and_des(self, im, src_shape):
        prob_nms, xf = self.net(im)
        # logger.info("im shape: {}".format(im.shape))
        # logger.info("prob_nms.shape: {}".format(prob_nms.shape))
        # logger.info("xf shape: {}".format(xf.shape))

        prob_nms = prob_nms.squeeze(dim=1)
        edge_size = 30
        prob_nms[:, :edge_size, :] = -1
        prob_nms[:, :, :edge_size] = -1
        prob_nms[:, src_shape[0] - edge_size:, :] = -1
        prob_nms[:, :, src_shape[1] - edge_size:] = -1
        # preds = [pred > 0.015 for pred in prob_nms]

        points = [torch.stack(torch.where(pred > 0.015)).T for pred in prob_nms]
        points = [torch.flip(element, dims=[1]) for element in points]
        # logger.info("prob_nms.shape: {}".format(prob_nms.shape))
        # logger.info("the first pred shape is : {}".format(preds[0].shape))
        # logger.info("len preds is: {}".format(len(preds)))
        # logger.info("points len: {}".format(len(points[0])))
        # print(points[0])
      
        # print(points[0])

        hi, wi = im.size()[2:]
        batch_size = im.size()[0]
        discriptor = list()
        for i in range(batch_size):
            coord_n = self.normalize(points[i], hi, wi)
            feat_f = self.sample_feat_by_coord(xf[i: i+1], coord_n.unsqueeze(dim=0))
            discriptor.append(feat_f)
        return points, discriptor         

    def test(self, im1, im2, coord1):
        '''
        given a pair of images im1, im2, compute the coorrespondences for query points coord1.
        We performa full image search at coarse level and local search at fine level
        :param im1: [batch_size, 3, h, w]
        :param im2: [batch_size, 3, h, w]
        :param coord1: [batch_size, n_pts, 2]
        :return: the fine level correspondence location [batch_size, n_pts, 2]
        '''

        xc1, xf1 = self.net(im1)
        xc2, xf2 = self.net(im2)

        h1i, w1i = im1.shape[2:]
        h2i, w2i = im2.shape[2:]

        coord1_n = self.normalize(coord1, h1i, w1i)
        feat1_c = self.sample_feat_by_coord(xc1, coord1_n)
        _, std_c = self.get_expected_correspondence_locs(feat1_c, xc2, with_std=True)

        coord2_ec_n = self.get_1nn_coord(feat1_c, xc2)
        feat1_f = self.sample_feat_by_coord(xf1, coord1_n)
        _, std_f = self.get_expected_correspondence_within_window(feat1_f, xf2, coord2_ec_n, with_std=True)

        coord2_ef_n = self.get_1nn_coord(feat1_f, xf2)
        coord2_ef = self.denormalize(coord2_ef_n, h2i, w2i)
        std = (std_c + std_f)/2

        return coord2_ef, std
