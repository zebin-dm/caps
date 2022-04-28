import torch
import torch.nn as nn
import torch.nn.functional as F
from CAPS.efficient import efficientnet_v2_s
from CAPS.magic_point_v4 import MagicPoint, initialize_weights


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(num_in_layers,
                              num_out_layers,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=(self.kernel_size - 1) // 2)
        self.bn = nn.BatchNorm2d(num_out_layers)

    def forward(self, x):
        return F.elu(self.bn(self.conv(x)), inplace=True)


class upconv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, scale):
        super().__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, pretrained=True, fine_out_ch=128):
        super().__init__()
        filters = [48, 64, 160]

        self.upconv3 = upconv(filters[2], filters[1], 3, 2)
        self.iconv3_x8 = conv(filters[1], filters[1], 3, 1)
        self.iconv3 = conv(filters[1] + filters[1], filters[1], 3, 1)
        self.upconv2 = upconv(filters[1], filters[0], 3, 2)
        self.iconv2 = conv(filters[0], filters[0], 3, 1)
        # fine-level conv
        self.conv_fine = conv(filters[0], fine_out_ch, 1, 1)
        initialize_weights(self)
        efficient_ins = efficientnet_v2_s(pretrained=pretrained)
        self.magic_net = MagicPoint(nms=4, bb_name="EfficientBBV2")
        self.layer5_6 = efficient_ins.features[4: 6]  # H/16

    def forward(self, x):
        x_s8, logits, prob, prob_nms = self.magic_net(x)
        x_16 = self.layer5_6(x_s8)
        x = self.upconv3(x_16)
        x_s8 = self.iconv3_x8(x_s8)
        x = torch.cat([x_s8, x], dim=1)
        x = self.iconv3(x)
        x = self.upconv2(x)
        x = self.iconv2(x)
        discriptor = self.conv_fine(x)
        return prob_nms, discriptor


def run():
    import time
    model = EfficientUNet()
    model.cuda()
    model.eval()

    run_time = 1000

    with torch.no_grad():
        in_size = [1, 1, 224, 224]
        dummy_input = torch.randn(*in_size, device='cuda')
        prob_nms, discriptor = model(dummy_input)
        print("discriptor.shape: {}".format(discriptor.shape))

        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(run_time):
            model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time() - start_time
    print("the time interval is :{}".format(end_time))


def m_test():
    model = EfficientUNet()
    model.cuda()
    model.eval()

    with torch.no_grad():
        in_size = [1, 1, 224, 224]
        dummy_input = torch.randn(*in_size, device='cuda')
        prob_nms, discriptor = model(dummy_input)
        print("prob_nms shape: {}".format(prob_nms.shape))
        print("discriptor shape: {}".format(discriptor.shape))


def to_onnx():
    efficient_unet = EfficientUNet()
    efficient_unet.cuda()
    efficient_unet.eval()
    in_name = ["data",]
    out_name = ["descriptor",]
    dummy_input = torch.randn([1, 1, 224, 224], device=torch.device("cuda:0"))
    onnx_model_path = "./efficient_unet.onnx"
    torch.onnx.export(efficient_unet,
                      dummy_input,
                      onnx_model_path,
                      input_names=in_name,
                      output_names=out_name,
                      dynamic_axes=None,
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      verbose=True)


if __name__ == "__main__":
    # run()
    # m_test()
    to_onnx()

