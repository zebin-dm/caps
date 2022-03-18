import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
import sys
sys.path.append('../')
from dataloader import megadepth
import torch.utils.data
from CAPS.caps_model import CAPSModel
import cv2


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title='',
                            small_text=[]):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for idx, [(x0, y0), (x1, y1), c] in enumerate(zip(mkpts0, mkpts1, color)):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

        if idx > 30:
            break
    # Scale factor for consistent visualization across scales.
    sc = min(H / 640., 2.0)

    # Big text.
    Ht = int(30 * sc)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0*sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8*sc), int(H-Ht*(i+.6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5*sc, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out



class Visualization(object):
    def __init__(self, args):
        dataset = megadepth.MegaDepth(args)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
        self.model = CAPSModel(args)
        self.loader_iter = iter(self.dataloader)

    def random_sample(self):
        self.sample = next(self.loader_iter)

    def plot_img_pair(self, with_std=False, with_epipline=False):
        self.coords = []
        self.colors = []
        self.with_std = with_std
        self.with_epipline = with_epipline
        im1 = self.sample['im1_ori']
        im2 = self.sample['im2_ori']
        self.h, self.w = im1.shape[1], im1.shape[2]
        im1 = im1.squeeze().cpu().numpy()
        im2 = im2.squeeze().cpu().numpy()
        blank = np.ones((self.h, 5, 3)) * 255
        out = np.concatenate((im1, blank, im2), 1).astype(np.uint8)

        self.fig = plt.figure(figsize=(12, 5))
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(out)
        self.ax.axis('off')
        plt.tight_layout()
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        color = tuple(np.random.rand(3).tolist())
        coord = [event.xdata, event.ydata]
        self.coord = coord
        self.color = color
        self.coords.append(coord)
        self.colors.append(color)
        self.ax.scatter(event.xdata, event.ydata, c=color)
        self.find_correspondence()
        self.plot_correspondence()

    def find_correspondence(self):
        data_in = self.sample
        data_in['coord1'] = torch.from_numpy(np.array(self.coord)).float().cuda().unsqueeze(0).unsqueeze(0)
        data_in['coord2'] = data_in['coord1']
        self.model.set_input(data_in)
        coord2_e, std = self.model.test()
        self.correspondence = coord2_e.squeeze().cpu().numpy()
        self.std = std.squeeze().cpu().numpy()

    def run_correspondence(self):
        sample = next(self.loader_iter)
        self.model.set_input(sample)
        coord2_e, std = self.model.test()
        mkpts1 = coord2_e.squeeze().cpu().numpy()
        mkpts0 = sample["coord1"].squeeze().cpu().numpy()
        std = std.squeeze().cpu().numpy()
        # self.plot_result(sample, coord2_e, std)

        color = np.random.rand(len(mkpts0), 3)
        print(sample['im1_ori'].shape)
        gray_0 = cv2.cvtColor(sample['im1_ori'].squeeze().cpu().numpy(), cv2.COLOR_BGR2GRAY)
        gray_1 = cv2.cvtColor(sample['im2_ori'].squeeze().cpu().numpy(), cv2.COLOR_BGR2GRAY)
        print(gray_0.shape)
        print(mkpts0.shape)
        out = make_matching_plot_fast(gray_0,
                                gray_1,
                                None,
                                None,
                                mkpts0=mkpts0,
                                mkpts1=mkpts1,
                                color=color,
                                text="",
                                path=None,
                                show_keypoints=False)
        cv2.imwrite("test.jpg", out)

    def plot_result(self, sample, coord2_e, std, with_std=True, with_epipline=True):
        mid_img_w = 5
        im1 = sample['im1_ori']
        im2 = sample['im2_ori']
        im1_h, im1_w = im1.shape[1], im1.shape[2]
        im1 = im1.squeeze().cpu().numpy()
        im2 = im2.squeeze().cpu().numpy()
        blank = np.ones((im1_h, mid_img_w, 3)) * 255
        out = np.concatenate((im1, blank, im2), 1).astype(np.uint8)

        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)
        ax.imshow(out)
        ax.axis('off')
        plt.tight_layout()

        point1 = sample["coord1"]
        point2 = coord2_e
        point1 = point1.squeeze(0)
        print(point2.shape)
        print(std.shape)
        # point2 = point2.squeeze(0)
        # std = std.squeeze(0)

        for p_n in range(10):
            point1_1 = point1[p_n]
            point2_1 = point2[p_n]
            std_1 = std[p_n]
            color = tuple(np.random.rand(3).tolist())
            point2_1[0] += im1_w + mid_img_w

            ax.scatter(point2_1[0], point2_1[1], color=color)
            ax.scatter(point1_1[0], point1_1[1], color=color)
            if with_std:
                print(point2_1[0])
                print(point2_1[1])
                print(std[0])
                circle = plt.Circle((point2_1[0], point2_1[1]), radius=100 * std_1, fill=False, color=color)
                ax.add_patch(circle)

            if with_epipline:
                line2 = cv2.computeCorrespondEpilines(np.array(point1_1).reshape(-1, 1, 2), 1,
                                                      sample['F'].squeeze().cpu().numpy())
                line2 = np.array(line2).squeeze()
                intersection = np.array([[0, -line2[2]/line2[1]],
                                         [-line2[2]/line2[0], 0],
                                         [im1_w-1, -(line2[2]+line2[0]*(im1_w-1))/line2[1]],
                                         [-(line2[1]*(im1_h-1)+line2[2])/line2[0], im1_h-1]])
                valid = (intersection[:, 0] >= 0) & (intersection[:, 0] <= im1_w-1) & \
                        (intersection[:, 1] >= 0) & (intersection[:, 1] <= im1_h-1)
                if np.sum(valid) == 2:
                    intersection = intersection[valid].astype(int)
                    x0, y0 = intersection[0]
                    x1, y1 = intersection[1]
                    l = mlines.Line2D([x0+im1_w + 5, x1+im1_w + 5], [y0, y1], color=color)
                    ax.add_line(l)
        fig.savefig("test.png")



    def plot_correspondence(self):
        point1 = self.sample["coord1"]
        point2 = self.correspondence
        point2[0] += self.w + 5
        self.ax.scatter(point2[0], point2[1], color=self.color)
        if self.with_std:
            circle = plt.Circle((point2[0], point2[1]), radius=100 * self.std, fill=False, color=self.color)
            self.ax.add_patch(circle)

        if self.with_epipline:
            line2 = cv2.computeCorrespondEpilines(np.array(point1).reshape(-1, 1, 2), 1,
                                                   self.sample['F'].squeeze().cpu().numpy())
            line2 = np.array(line2).squeeze()
            intersection = np.array([[0, -line2[2]/line2[1]],
                                     [-line2[2]/line2[0], 0],
                                     [self.w-1, -(line2[2]+line2[0]*(self.w-1))/line2[1]],
                                     [-(line2[1]*(self.h-1)+line2[2])/line2[0], self.h-1]])
            valid = (intersection[:, 0] >= 0) & (intersection[:, 0] <= self.w-1) & \
                    (intersection[:, 1] >= 0) & (intersection[:, 1] <= self.h-1)
            if np.sum(valid) == 2:
                intersection = intersection[valid].astype(int)
                x0, y0 = intersection[0]
                x1, y1 = intersection[1]
                l = mlines.Line2D([x0+self.w + 5, x1+self.w + 5], [y0, y1], color=self.color)
                self.ax.add_line(l)

        plt.show()
        self.fig.savefig("test.png")









