import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

try:
    import wandb

except ImportError:
    wandb = None

from loss import *
from network import *
from dataset import *
from training_loop import *
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    #parser.add_argument("data_root", type=str, help="Root directory for dataset")
    parser.add_argument("--data_root", type=str, default=r"/home/evgeniy-kononov/Documents/data_transfer/train_prepared", help="Root directory for dataset")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Path to the checkpoints to resume training")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--log_resolution", type=int, default=6, help="The log2 of image resolution")
    parser.add_argument("--d_latent", type=int, default=128, help="The size of the latent space")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers in the mapping network")
    parser.add_argument("--n_features", type=int, default=32,
                        help="Number of features in the convolution layer at the highest resolution")
    parser.add_argument("--max_features", type=int, default=128,
                        help="Maximum number of features in any generator block")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size during training")
    parser.add_argument("--total_steps", type=int, default=45000,
                        help="Number of training steps (~len(data) * 350 / batch_size or 25 000 000 / batch_size steps)")
    parser.add_argument("--lr_multiplier", type=float, default=0.01,
                        help="Learning rate multiplier for the mapping layers")
    parser.add_argument("--plot_freq", type=int, default=100, help="Sample plot frequency")
    parser.add_argument("--save_freq", type=int, default=500, help="Models save frequency")
    parser.add_argument("--initial_step", type=int, default=1, help="Initialization step")
    parser.add_argument("--use_wandb", action="store_true", help="Use weights and biases logging")
    parser.add_argument("--WGAN_GP", action="store_true", help="Whether to use the WGAN-GP")
    parser.add_argument("--gp_weight", type=float, default=10., help="WGAN-GP weight")
    parser.add_argument("--r1_gamma", type=float, default=10.,
                        help="Weight of the r1 regularization (γ ∈ [γ0/5, γ0 · 5], γ0 = 0.0002 · N/M, where N = w × h is the number of pixels and M is the minibatch size)")
    parser.add_argument("--style_mixing_prob", type=float, default=0., help="Probability of latent code mixing")
    parser.add_argument("--path_lenght_use", action="store_true", help="Whether to use path lenght regularization")
    parser.add_argument("--activation", action="store_false", help="Generator output activation")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--calc_fid", action="store_false", help="Whether to calculate FID")
    parser.add_argument("--size", type=int, default=64, help="Size of training images")

    args = parser.parse_args()

    # Size of training images
    height, width, deepth = args.size, args.size, args.size

    if args.activation:
        args.activation = nn.Tanh()

    # Adam hyperparameters
    c = args.total_steps / (args.total_steps + 1)
    beta1 = 0
    beta2 = 0.99
    eps = 10e-8

    # Number of GPUs available (0 for CPU mode)
    device = "cuda"
    n_gpu = int(os.environ["WORLD_SIZE"] if "WORLD_SIZE" in os.environ else 1)
    args.distributed = n_gpu > 1
    if args.distributed:
        # os.environ["MASTER_ADDR"] = "localhost"
        # os.environ["MASTER_PORT"] = "29500"
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    net_M = MappingNetwork(args.d_latent, args.n_layers, args.lr_multiplier).to(device)
    net_M_ema = MappingNetwork(args.d_latent, args.n_layers, args.lr_multiplier).to(device)
    net_M_ema.eval()
    weights_ema(net_M_ema, net_M, 0)

    net_G = Generator(
        args.log_resolution, args.d_latent, args.n_features, args.max_features, activation=args.activation
    ).to(device)
    net_G_ema = Generator(
        args.log_resolution, args.d_latent, args.n_features, args.max_features, activation=args.activation
    ).to(device)
    net_G_ema.eval()
    weights_ema(net_G_ema, net_G, 0)

    net_D = Discriminator(args.log_resolution, args.n_features, args.max_features).to(device)

    if args.WGAN_GP:
        c_G, c_D = 1, 1
        G_reg_interval, D_reg_interval = 1, 1
    else:
        # How often to perform regularization for G?
        G_reg_interval = 8
        c_G = G_reg_interval / (G_reg_interval + 1)

        # How often to perform regularization for D?
        D_reg_interval = 16
        c_D = D_reg_interval / (D_reg_interval + 1)

    G_opt = torch.optim.Adam(
        list(net_M.parameters()) + list(net_G.parameters()),
        lr=args.lr * c_G, betas=(beta1 ** c_G, beta2 ** c_G), eps=eps,
    )
    D_opt = torch.optim.Adam(
        net_D.parameters(),
        lr=args.lr * c_D, betas=(beta1 ** c_D, beta2 ** c_D), eps=eps,
    )

    wandb_run_id = None
    pl_exp_sum_a = None
    pl_steps = None
    if args.ckpt_path is not None:
        print("Load model:", args.ckpt_path)
        checkpoint = torch.load(args.ckpt_path, map_location=lambda storage, loc: storage)
        pl_exp_sum_a = checkpoint.get("pl_exp_sum_a", None)
        pl_steps = checkpoint.get("pl_steps", None)
        initial_step = checkpoint["step"]
        wandb_run_id = checkpoint.get("wandb_run_id", None)
        net_M.load_state_dict(checkpoint["net_M"])
        net_M_ema.load_state_dict(checkpoint["net_M_ema"])
        net_G.load_state_dict(checkpoint["net_G"])
        net_G_ema.load_state_dict(checkpoint["net_G_ema"])
        net_D.load_state_dict(checkpoint["net_D"])
        G_opt.load_state_dict(checkpoint["G_opt"])
        D_opt.load_state_dict(checkpoint["D_opt"])

    if args.distributed:
        net_M = nn.parallel.DistributedDataParallel(
            net_M,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        net_G = nn.parallel.DistributedDataParallel(
            net_G,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        net_D = nn.parallel.DistributedDataParallel(
            net_D,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    dataset = CuboidsDataset(args.data_root)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )

    if get_rank() == 0 and wandb is not None and args.use_wandb:
        config = dict(
            data="inception_dataset",
            loss="R1" if args.WGAN_GP is False else "WGAN_GP",
            r1_gamma=args.r1_gamma,
            activation=args.activation,
            d_latent=args.d_latent,
            n_layers=args.n_layers,
            n_features=args.n_features,
            max_features=args.max_features,
            learning_rate=args.lr,
            initial_step=args.initial_step,
            total_steps=args.total_steps,
            batch_size=args.batch_size,
            G_reg_interval=G_reg_interval,
            D_reg_interval=D_reg_interval,
        )
        wandb.init(project="StyleGAN2", entity="pnrpu", id=wandb_run_id, resume="allow", config=config)

    train(
        loader=dataloader,
        net_M=net_M,
        net_M_ema=net_M_ema,
        net_G=net_G,
        net_G_ema=net_G_ema,
        net_D=net_D,
        G_opt=G_opt,
        D_opt=D_opt,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        initial_step=args.initial_step,
        plot_freq=args.plot_freq,
        save_freq=args.save_freq,
        device=device,
        use_wandb=args.use_wandb,
        WGAN_GP=args.WGAN_GP,
        gp_weight=args.gp_weight,
        r1_gamma=args.r1_gamma,
        path_lenght_use=args.path_lenght_use,
        style_mixing_prob=args.style_mixing_prob,
        pl_exp_sum_a=pl_exp_sum_a,
        pl_steps=pl_steps,
        G_reg_interval=G_reg_interval,
        D_reg_interval=D_reg_interval,
        calc_fid=args.calc_fid,
        distributed=args.distributed,
    )
