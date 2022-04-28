# -*-coding:utf8-*-
import math
import torch
import torch.nn as nn
from CAPS import efficient
from functools import partial
from CAPS.ops_misc import Conv2dNormActivation


def pixel_shuffle(tensor, scale_factor):
    """
    Implementation of pixel shuffle using numpy

    Parameters:
    -----------
    tensor: input tensor, shape is [N, C, H, W]
    scale_factor: scale factor to up-sample tensor

    Returns:
    --------
    tensor: tensor after pixel shuffle, shape is [N, C/(r*r), r*H, r*W],
        where r refers to scale factor
    """
    num, ch, height, width = tensor.shape
    assert ch % (scale_factor * scale_factor) == 0

    # new_ch = ch // (scale_factor * scale_factor)
    new_ch = torch.div(ch, scale_factor * scale_factor, rounding_mode='floor')
    new_height = height * scale_factor
    new_width = width * scale_factor

    tensor = tensor.reshape(
        [num, new_ch, scale_factor, scale_factor, height, width])
    # new axis: [num, new_ch, height, scale_factor, width, scale_factor]
    tensor = tensor.permute(0, 1, 4, 2, 5, 3)
    tensor = tensor.reshape(num, new_ch, new_height, new_width)
    return tensor


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2.0 / (fan_out // m.groups)))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data.fill_(1.0)
            m.bias.data.zero_()
        elif isinstance(m, torch.nn.Linear):
            m.weight.data.uniform_(-1.0 / math.sqrt(m.weight.size()[0]), 1.0 / math.sqrt(m.weight.size()[0]))
            m.bias.data.zero_()


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


class EfficientBB(nn.Module):
    def __init__(self):
        super().__init__()
        efficient_ins = efficient.efficientnet_v2_s(pretrained=True, progress=True)
        self.first_conv = Conv2dNormActivation(in_channels=1,
                                               out_channels=3, 
                                               kernel_size=3, 
                                               stride=1, 
                                               norm_layer=partial(nn.BatchNorm2d, eps=1e-03),
                                               activation_layer=nn.SiLU)

        self.pointfeat = efficient_ins.features[:4]   # H/8
        initialize_weights(self.first_conv)

    def forward(self, x):
        out = self.first_conv(x)
        out = self.pointfeat(out)
        return out


class DetectorHead(torch.nn.Module):
    def __init__(self, input_channel, grid_size):
        super().__init__()
        self.grid_size = grid_size
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)
        self.bnPa = torch.nn.BatchNorm2d(256)
        self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = self.bnPa(self.relu(self.convPa(x)))
        logits = self.bnPb(self.convPb(out))  #(B,65,H,W)

        prob = self.softmax(logits)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        prob = prob.squeeze(dim=1)#[B,H,W]
        return logits, prob


class MagicPoint(nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, nms, bb_name, grid_size=8):
        super(MagicPoint, self).__init__()
        self.nms = nms
        self.backbone = EfficientBB()
        self.bb_out_chs = 64
        self.detector_head = DetectorHead(input_channel=self.bb_out_chs, grid_size=grid_size)

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        """
        x_s8 = self.backbone(x)
        logits, prob = self.detector_head(x_s8)   # N x H x W
        if not self.training:
            prob_nms = simple_nms(prob, nms_radius=self.nms)
        else:
            prob_nms = None
        return x_s8, logits, prob, prob_nms




if __name__ == "__main__":
    import os
    import yaml
    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    config_file = "./config/magic_point_syn_train.yaml"
    with open(config_file, 'r') as fin:
        config = yaml.safe_load(fin)
    
    device=torch.device("cuda:0")
    nms = config['model']['nms']
    net = MagicPoint(nms=nms)
    net = net.to(device)
    net.eval()
    in_size=[1, 1, 608, 608]
    with torch.no_grad():
        data = torch.randn(*in_size, device=device)
        out = net(data)


    # model = EfficientBB()
    # model.eval()
    # in_size=[1, 1, 608, 608]
    # data = torch.randn(*in_size)
    # with torch.no_grad():
    #     print(data.shape) 
    #     out = model(data)
    #     print(out.shape)







