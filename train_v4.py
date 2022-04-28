import os
import config
from tensorboardX import SummaryWriter
from CAPS.caps_model_v3 import CAPSModel
from dataloader.megadepth_v3 import MegaDepthLoader
from utils.draw_utils import cycle
from loguru import logger


def train_megadepth(args):
    # save a copy for the current args in out_folder
    print("args.outdir: {}".format(args.outdir))
    print("args.exp_name: {}".format(args.exp_name))

    out_folder = os.path.join(args.outdir, args.exp_name)
    os.makedirs(out_folder, exist_ok=True)
    f = os.path.join(out_folder, 'args.txt')
    with open(f, 'w') as file:
        for arg in vars(args):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    # tensorboard writer
    tb_log_dir = os.path.join(args.logdir, args.exp_name)
    print('tensorboard log files are stored in {}'.format(tb_log_dir))
    writer = SummaryWriter(tb_log_dir)

    # megadepth data loader
    train_loader = MegaDepthLoader(args).load_data()


    # define model
    model = CAPSModel(args)
    start_step = model.start_step
    # train_loader_iterator = iter(cycle(train_loader))
    # training loop
    # for step in range(start_step + 1, start_step + args.n_iters + 1):
    #     data = next(train_loader_iterator)
    #     model.set_input(data)
    #     model.optimize_parameters()
    #     model.write_summary(writer, step)
    #     if step % args.save_interval == 0 and step > 0:
    #         model.save_model(step)

    iters_per_epoch = len(train_loader)
    epoch_num = (args.n_iters // iters_per_epoch + 1)
    logger.info("epoch number: {}, every epoch iter num: {}".format(epoch_num, iters_per_epoch))
    for epoch in range(epoch_num):
        for idx, data in enumerate(train_loader):
            model.set_input(data)
            model.optimize_parameters()

            step = epoch * iters_per_epoch + idx
            model.write_summary(writer, step)
            if step % args.save_interval == 0 and step > 0:
                model.save_model(step)


if __name__ == '__main__':
    args = config.get_args()
    print(args)
    train_megadepth(args)




