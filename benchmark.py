import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.io import loadmat
from tqdm import tqdm


def mnn_matcher(descriptors_a, descriptors_b):
    device = descriptors_a.device
    sim = descriptors_a @ descriptors_b.t()
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    matches = torch.stack([ids1[mask], nn12[mask]])
    return matches.t().data.cpu().numpy()


def benchmark_features(read_feats, data_info):
    seq_names = sorted(os.listdir(data_info.data_root))
    lim = [1, 15]
    rng = np.arange(lim[0], lim[1] + 1)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')

    n_feats = []
    n_matches = []
    seq_type = []
    i_err = {thr: 0 for thr in rng}
    v_err = {thr: 0 for thr in rng}

    for seq_idx, seq_name in tqdm(enumerate(seq_names), total=len(seq_names)):
        keypoints_a, descriptors_a = read_feats(seq_name, 1)
        n_feats.append(keypoints_a.shape[0])

        for im_idx in range(2, 7):
            keypoints_b, descriptors_b = read_feats(seq_name, im_idx)
            n_feats.append(keypoints_b.shape[0])

            matches = mnn_matcher(
                torch.from_numpy(descriptors_a).to(device=device),
                torch.from_numpy(descriptors_b).to(device=device)
            )

            homography = np.loadtxt(os.path.join(data_info.data_root, seq_name, "H_1_" + str(im_idx)))

            pos_a = keypoints_a[matches[:, 0], : 2]
            pos_a_h = np.concatenate([pos_a, np.ones([matches.shape[0], 1])], axis=1)
            pos_b_proj_h = np.transpose(np.dot(homography, np.transpose(pos_a_h)))
            pos_b_proj = pos_b_proj_h[:, : 2] / pos_b_proj_h[:, 2:]

            pos_b = keypoints_b[matches[:, 1], : 2]

            dist = np.sqrt(np.sum((pos_b - pos_b_proj) ** 2, axis=1))

            n_matches.append(matches.shape[0])
            seq_type.append(seq_name[0])

            if dist.shape[0] == 0:
                dist = np.array([float("inf")])

            for thr in rng:
                if seq_name[0] == 'i':
                    i_err[thr] += np.mean(dist <= thr)
                else:
                    v_err[thr] += np.mean(dist <= thr)

    seq_type = np.array(seq_type)
    n_feats = np.array(n_feats)
    n_matches = np.array(n_matches)

    return i_err, v_err, [seq_type, n_feats, n_matches]


def summary(stats, data_info):
    seq_type, n_feats, n_matches = stats
    print('# Features: {:f} - [{:d}, {:d}]'.format(np.mean(n_feats), np.min(n_feats), np.max(n_feats)))
    print('# Matches: Overall {:f}, Illumination {:f}, Viewpoint {:f}'.format(
        np.sum(n_matches) / ((data_info.num_i + data_info.num_v) * 5),
        np.sum(n_matches[seq_type == 'i']) / (data_info.num_i * 5),
        np.sum(n_matches[seq_type == 'v']) / (data_info.num_v * 5)))


def generate_read_function(method, feat_info, extension='ppm'):
    def read_function(seq_name, im_idx):
        aux = np.load(os.path.join(feat_info.checkpoint[method].path, seq_name, '%d.%s.%s' % (im_idx, extension, method)))
        if top_k is None:
            return aux['keypoints'], aux['descriptors']
        else:
            assert('scores' in aux)
            ids = np.argsort(aux['scores'])[-top_k :]
            return aux['keypoints'][ids, :], aux['descriptors'][ids, :]
    return read_function


def sift_to_rootsift(descriptors):
    return np.sqrt(descriptors / np.expand_dims(np.sum(np.abs(descriptors), axis=1), axis=1) + 1e-16)


def parse_mat(mat):
    keypoints = mat['keypoints'][:, : 2]
    raw_descriptors = mat['descriptors']
    l2_norm_descriptors = raw_descriptors / np.expand_dims(np.sum(raw_descriptors ** 2, axis=1), axis=1)
    descriptors = sift_to_rootsift(l2_norm_descriptors)
    if top_k is None:
        return keypoints, descriptors
    else:
        assert('scores' in mat)
        ids = np.argsort(mat['scores'][0])[-top_k :]
        return keypoints[ids, :], descriptors[ids, :]


class HPatchInfo(object):
    def __init__(self, data_root, num_i=52, num_v=56):
        self.data_root = data_root
        self.num_i = num_i
        self.num_v = num_v


class CheckPoint(object):
    def __init__(self, path):
        self.path = path


class FeatureInfo(object):
    def __init__(self):
        self.checkpoint = {}


class LocalFeatureBenchMark(object):
    def __init__(self, data_cache_root, top_k=None):
        self.methods = {
            "hesaff": ["Hes. Aff. + Root-SIFT", "black", "-"],
            "hesaffnet": ['HAN + HN++', 'orange', "-"],
            "delf": ['DELF', 'red', "-"],
            "delf-new": ['DELF New', 'red', "-"],
            "superpoint": ['SuperPoint', 'blue', "-"],
            "lf-net": ['LF-Net', 'brown', "-"],
            "d2-net": ['D2-Net', 'purple', "-"],
            "d2-net-ms": ['D2-Net MS', 'green', "-"],
            "d2-net-trained": ['D2-Net Trained', 'purple', '--'],
            "d2-net-trained-ms": ['D2-Net Trained MS', 'green', '--'],
            "caps": ["CAPS+SIFT", 'yellow', 'dashdot']}

        # Change here if you want to use top K or all features.
        # top_k = 2000
        self.top_k = top_k
        self.data_cache_root = data_cache_root
        self.hpatches_info = HPatchInfo(data_root=os.path.join(data_cache_root, "hpatches-sequences-release"))
        self.feat_info = FeatureInfo()
        self.feat_info.checkpoint["caps"] = CheckPoint("/home/dm/work/02.workspace/caps/out/extract_out2")
        self.cache_dir = os.path.join(data_cache_root, "cache")
        if top_k is not None:
            self.cache_dir = os.path.join(data_cache_root, "cache-top")





        errors = {}

    @staticmethod
    def matching(methods, cache_dir, data_info, feat_info):
        errors = {}
        for method in methods.keys():
            output_file = os.path.join(cache_dir, method + '.npy')
            print("find output file: {}".format(output_file))
            print(method)
            if method == 'hesaff':
                read_function = lambda seq_name, im_idx: parse_mat(
                    loadmat(os.path.join(data_info.data_root, seq_name, '%d.ppm.hesaff' % im_idx), appendmat=False))
            else:
                if method == 'delf' or method == 'delf-new':
                    read_function = generate_read_function(method, feat_info, extension='png')
                else:
                    read_function = generate_read_function(method, feat_info)
            if os.path.exists(output_file):
                print('Loading precomputed errors...')
                errors[method] = np.load(output_file, allow_pickle=True)
            else:
                errors[method] = benchmark_features(read_function, data_info)
                np.save(output_file, errors[method])
            summary(errors[method][-1], data_info)
        return errors

    # plotting
    @staticmethod
    def plotting(methods, errors, data_info):
        n_i = data_info.num_i
        n_v = data_info.num_v
        plt_lim = [1, 10]
        plt_rng = np.arange(plt_lim[0], plt_lim[1] + 1)

        plt.rc('axes', titlesize=25)
        plt.rc('axes', labelsize=25)

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        for method, [name, color, ls] in methods.items():
            i_err, v_err, _ = errors[method]
            plt.plot(plt_rng, [(i_err[thr] + v_err[thr]) / ((n_i + n_v) * 5) for thr in plt_rng], color=color, ls=ls,
                     linewidth=3, label=name)
        plt.title('Overall')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylabel('MMA')
        plt.ylim([0, 1])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.legend()

        plt.subplot(1, 3, 2)
        for method, [name, color, ls] in methods.items():
            i_err, v_err, _ = errors[method]
            plt.plot(plt_rng, [i_err[thr] / (n_i * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
        plt.title('Illumination')
        plt.xlabel('threshold [px]')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.gca().axes.set_yticklabels([])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)

        plt.subplot(1, 3, 3)
        for method, [name, color, ls] in methods.items():
            i_err, v_err, _ = errors[method]
            plt.plot(plt_rng, [v_err[thr] / (n_v * 5) for thr in plt_rng], color=color, ls=ls, linewidth=3, label=name)
        plt.title('Viewpoint')
        plt.xlim(plt_lim)
        plt.xticks(plt_rng)
        plt.ylim([0, 1])
        plt.gca().axes.set_yticklabels([])
        plt.grid()
        plt.tick_params(axis='both', which='major', labelsize=20)

        if top_k is None:
            plt.savefig('hseq.png', bbox_inches='tight', dpi=300)
        else:
            plt.savefig('hseq-top.png', bbox_inches='tight', dpi=300)

    def run(self):
        errors = self.matching(methods=self.methods,
                               cache_dir=self.cache_dir,
                               data_info=self.hpatches_info,
                               feat_info=self.feat_info)
        self.plotting(methods=self.methods,
                      errors=errors,
                      data_info=self.hpatches_info)


if __name__ == "__main__":
    data_cache_root = "/home/deepmirror/work/04.dataset/hpatches_sequences"
    top_k = None
    lf_benchmark = LocalFeatureBenchMark(data_cache_root=data_cache_root,
                                         top_k=top_k)
    lf_benchmark.run()


