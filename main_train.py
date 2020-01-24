import torch
import torch.optim as optim

import GanTrainer
import config
import params
import printer
import utils.model_save_util as model_save_util
from Blocks import LambdaLR

if __name__ == '__main__':
    torch.manual_seed(params.manualSeed)
    opt = config.get_opt()
    printer.print_opt(opt)

    net_G = model_save_util.create_G_net(opt.model, opt.device, opt.isCheckpoint, opt.input_dim, opt.input_images_mean,
                                         opt.filters,
                                         opt.con_operator, opt.unet_depth, opt.add_frame, opt.pyramid_loss)
    net_D = model_save_util.create_D_net(opt.input_dim, opt.d_down_dim, opt.device, opt.isCheckpoint, opt.d_norm)

    printer.print_net("G", net_G, opt, input_size=364)
    printer.print_net("D", net_D, opt, input_size=256)

    # Setup Adam optimizers for both G and D
    optimizer_D = optim.Adam(net_D.parameters(), lr=opt.D_lr, betas=(params.beta1, 0.999))
    optimizer_G = optim.Adam(net_G.parameters(), lr=opt.G_lr, betas=(params.beta1, 0.999))

    # Learning rate update schedulers
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.num_epochs, 0, opt.decay_epoch).step
    )
    lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D, lr_lambda=LambdaLR(opt.num_epochs, 0, opt.decay_epoch).step
    )

    # writer = Writer.Writer(g_t_utils.get_loss_path(result_dir_pref, model, params.loss_path))
    gan_trainer = GanTrainer.GanTrainer(opt, net_G, net_D, optimizer_G, optimizer_D, lr_scheduler_G, lr_scheduler_D)
    gan_trainer.train()
