from utils.functions import Visualization
import config

args = config.get_args()
args.datadir = '/home/dm/work/02.workspace/caps/checkpoint/CAPS-MegaDepth-release-light'  # replace this with your data directory
args.ckpt_path = '/home/dm/work/02.workspace/caps/checkpoint/caps-pretrained.pth'  # replace this with your path of pretrained model
args.phase = 'test'
sample = Visualization(args)
sample.run_correspondence()