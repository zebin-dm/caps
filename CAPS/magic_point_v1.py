# -*-coding:utf8-*-
import math
import torch
import torch.nn as nn
from torchvision.models import efficientnet
from torchvision.ops.misc import ConvNormActivation
from utils.tensor_op import pixel_shuffle
from utils.debug_utils import AverageTimer


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
    def __init__(self, encoder='b0', pretrained=True, coarse_out_ch=128, fine_out_ch=128):
        super(EfficientBB, self).__init__()
        efficient_ins = efficientnet.efficientnet_b0(pretrained=pretrained)
        self.first_conv = ConvNormActivation(in_channels=1,
                                             out_channels=32, 
                                             kernel_size=3, 
                                             stride=2, 
                                             norm_layer=nn.BatchNorm2d,
                                             activation_layer=nn.SiLU)

        self.layer2_3 = efficient_ins.features[1:3]   # H/4
        self.layer4 = efficient_ins.features[3]      # H/8
        initialize_weights(self.first_conv)


    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        x = self.first_conv(x)
        x_s4 = self.layer2_3(x)
        # print("x_s4 shape: {}".format(x_s4.shape))
        x_s8 = self.layer4(x_s4)
        # print("x_s8 shape: {}".format(x_s8.shape))
        return x_s4, x_s8


class EfficientBBV2(nn.Module):
    def __init__(self, encoder='b0', pretrained=True, coarse_out_ch=128, fine_out_ch=128):
        super(EfficientBBV2, self).__init__()
        efficient_ins = efficientnet.efficientnet_b0(pretrained=pretrained)
        self.first_conv = ConvNormActivation(in_channels=1,
                                             out_channels=3, 
                                             kernel_size=3, 
                                             stride=1, 
                                             norm_layer=nn.BatchNorm2d,
                                             activation_layer=nn.SiLU)

        self.layer1_3 = efficient_ins.features[:3]   # H/4
        self.layer4 = efficient_ins.features[3]      # H/8
        initialize_weights(self.first_conv)


    def forward(self, x):
        # print("x shape: {}".format(x.shape))
        x = self.first_conv(x)
        x_s4 = self.layer1_3(x)
        # print("x_s4 shape: {}".format(x_s4.shape))
        x_s8 = self.layer4(x_s4)
        # print("x_s8 shape: {}".format(x_s8.shape))
        return x_s4, x_s8


class DetectorHead(torch.nn.Module):
    def __init__(self, input_channel, grid_size):
        super(DetectorHead, self).__init__()
        self.grid_size = grid_size
        self.convPa = torch.nn.Conv2d(input_channel, 256, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)

        self.convPb = torch.nn.Conv2d(256, pow(grid_size, 2)+1, kernel_size=1, stride=1, padding=0)
        self.bnPa = torch.nn.BatchNorm2d(256)
        self.bnPb = torch.nn.BatchNorm2d(pow(grid_size, 2)+1)

        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        out = None
        out = self.bnPa(self.relu(self.convPa(x)))
        logits = self.bnPb(self.convPb(out))  #(B,65,H,W)

        prob = self.softmax(logits)
        prob = prob[:, :-1, :, :]  # remove dustbin,[B,64,H,W]
        # Reshape to get full resolution heatmap.
        prob = pixel_shuffle(prob, self.grid_size)  # [B,1,H*8,W*8]
        return logits, prob


class MagicPoint(nn.Module):
    """ Pytorch definition of SuperPoint Network. """
    def __init__(self, nms, bb_name, input_channel=1, grid_size=8):
        super(MagicPoint, self).__init__()
        self.nms = nms
        self.bb_name = bb_name
        if bb_name == "EfficientBB":
            self.backbone = EfficientBB()
            out_chs = 40
        elif bb_name == "EfficientBBV2":
            self.backbone = EfficientBBV2()
            out_chs = 40
        else:
            raise ValueError("Backbone not support")

        self.detector_head = DetectorHead(input_channel=out_chs, grid_size=grid_size)
        self.average_time = AverageTimer()

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x H x W.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
        """
        self.average_time.reset()      # TODO remove 
        x_s4, x_s8 = self.backbone(x)
        self.average_time.update("backbone")   # TODO remove

        logits, prob = self.detector_head(x_s8)   # N x H x W
        self.average_time.update("detector_head")
        if not self.training:
            prob_nms = simple_nms(prob, nms_radius=self.nms)
            self.average_time.update("nms")
        else:
            prob_nms = None
        return x_s4, x_s8, logits, prob, prob_nms


if __name__ == "__main__":
    import time
    device=torch.device("cuda:0")
    net = MagicPoint(nms=4, bb_name="EfficientBBV2")
    net = net.to(device)
    net.eval()
    net.average_time.cuda = True
    in_size=[1, 1, 608, 608]
    with torch.no_grad():
        data = torch.randn(*in_size, device=device)
        net.average_time.add = False
        out = net(data)
        net.average_time.add = True

        run_time = 1000
        torch.cuda.synchronize()
        start_time = time.time()
        for idx in range(run_time):
            out = net(data)
        torch.cuda.synchronize()
        time_interval = time.time() - start_time
        print(time_interval)
        net.average_time.print()









