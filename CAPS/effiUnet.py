import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet


class conv(nn.Module):
    def __init__(self, num_in_layers, num_out_layers, kernel_size, stride):
        super(conv, self).__init__()
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
        super(upconv, self).__init__()
        self.scale = scale
        self.conv = conv(num_in_layers, num_out_layers, kernel_size, 1)

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.scale, align_corners=True, mode='bilinear')
        return self.conv(x)


class EfficientUNet(nn.Module):
    def __init__(self, encoder='b0', pretrained=True, coarse_out_ch=128, fine_out_ch=128):
        super(EfficientUNet, self).__init__()
        # assert encoder in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], "Incorrect encoder type"
        # if encoder in ['resnet18', 'resnet34']:
        #     filters = [64, 128, 256, 512]
        # else:
        #     filters = [256, 512, 1024, 2048]

        filters = [24, 40, 112]
        efficient_ins = efficientnet.efficientnet_b0(pretrained=True)
        self.layer1_3 = efficient_ins.features[:3]   # H/4
        self.layer4 = efficient_ins.features[3]      # H/8
        self.layer5_6 = efficient_ins.features[4:6]  # H/16

        self.upconv3 = upconv(filters[2], filters[1], 3, 2)
        self.iconv3 = conv(filters[1] + filters[1], filters[1], 3, 1)
        self.upconv2 = upconv(filters[1], filters[0], 3, 2)
        self.iconv2 = conv(filters[0] + filters[0], filters[0] + filters[0], 3, 1)
        # fine-level conv
        self.conv_fine = conv(48, fine_out_ch, 1, 1)

        # self.firstconv = resnet.conv1  # H/2
        # self.firstbn = resnet.bn1
        # self.firstrelu = resnet.relu
        # self.firstmaxpool = resnet.maxpool  # H/4
        #
        # # encoder
        # self.layer1 = resnet.layer1  # H/4
        # self.layer2 = resnet.layer2  # H/8
        # self.layer3 = resnet.layer3  # H/16
        #
        # # coarse-level conv
        # self.conv_coarse = conv(filters[2], coarse_out_ch, 1, 1)
        #
        # # decoder
        # self.upconv3 = upconv(filters[2], 512, 3, 2)
        # self.iconv3 = conv(filters[1] + 512, 512, 3, 1)
        # self.upconv2 = upconv(512, 256, 3, 2)
        # self.iconv2 = conv(filters[0] + 256, 256, 3, 1)
        #
        # # fine-level conv
        # self.conv_fine = conv(256, fine_out_ch, 1, 1)

    def skipconnect(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        return x

    def forward(self, x):
        # x = self.firstrelu(self.firstbn(self.firstconv(x)))
        # x = self.firstmaxpool(x)
        #
        # x1 = self.layer1(x)
        # x2 = self.layer2(x1)
        # x3 = self.layer3(x2)
        #
        # x_coarse = self.conv_coarse(x3)
        #
        # x = self.upconv3(x3)
        # x = self.skipconnect(x2, x)
        # x = self.iconv3(x)
        #
        # x = self.upconv2(x)
        # x = self.skipconnect(x1, x)
        # x = self.iconv2(x)
        #
        # x_fine = self.conv_fine(x)
        # return [x_coarse, x_fine]

        x_s4 = self.layer1_3(x)
        # print("x_s4 shape: {}".format(x_s4.shape))
        x_s8 = self.layer4(x_s4)
        # print("x_s8 shape: {}".format(x_s8.shape))
        x_16 = self.layer5_6(x_s8)
        # print("x_16 shape: {}".format(x_16.shape))

        x = self.upconv3(x_16)
        # print("x shape: {}, x_s8 shape: {}".format(x.shape, x_s8.shape))
        # x = self.skipconnect(x_s8, x)
        x = torch.cat([x_s8, x], dim=1)
        x = self.iconv3(x)

        x = self.upconv2(x)
        # print("x shape: {}, x_s4 shape: {}".format(x.shape, x_s4.shape))
        # x = self.skipconnect(x_s4, x)
        x = torch.cat([x_s4, x], dim=1)
        x = self.iconv2(x)

        x_fine = self.conv_fine(x)
        return x_fine


def run():
    import time
    model = EfficientUNet()
    model.cuda()
    model.eval()

    run_time = 1000

    with torch.no_grad():
        in_size = [1, 3, 224, 224]
        dummy_input = torch.randn(*in_size, device='cuda')
        out = model(dummy_input)
        print("out.shape: {}".format(out.shape))

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
        in_size = [1, 3, 224, 224]
        dummy_input = torch.randn(*in_size, device='cuda')
        out = model(dummy_input)
        print("out.shape: {}".format(out.shape))


def to_onnx():
    efficient_unet = EfficientUNet()
    efficient_unet.cuda()
    efficient_unet.eval()
    in_name = ["data",]
    out_name = ["descriptor",]
    dummy_input = torch.randn([1, 3, 224, 224], device=torch.device("cuda:0"))
    onnx_model_path = "./efficient_unet.pth"
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