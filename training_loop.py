import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from torch import distributed as dist
from loss import *
from dataset import *
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

import sys
sys.path.append(r'C:\Users\Evgeniy\Jupyter\Work\stylegan2\inception_v3')
from InceptionV3 import Inception3 as MODEL
from fid_score import *

try:
    import wandb

except ImportError:
    wandb = None


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def data_plot(data, param=1., cmap='twilight', alpha=1, figsize=(7, 7)):
    # Create the x, y, and z coordinate arrays.  We use 
    # numpy's broadcasting to do all the hard work for us.
    # We could shorten this even more by using np.meshgrid.
    x = np.arange(0, data.shape[0], 1)
    y = np.arange(0, data.shape[1], 1)
    z = np.arange(0, data.shape[2], 1)
    x, y, z = np.meshgrid(x, y, z)

    eps = 10e-15

    # Turn the volumetric data into an RGB array that's
    # just grayscale.  There might be better ways to make
    # ax.scatter happy.
    c = np.tile(data.ravel()[:, None], [1, 1])
    # Оставляем индексы со значением >= param
    mask = np.unique(np.where(c >= param)[0])

    # Do the plotting in a single call.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x.ravel()[mask],
               y.ravel()[mask],
               z.ravel()[mask],
               c=np.arange(0 + eps, 1, 1 / mask.shape[0]),
               alpha=alpha, cmap=cmap)
    return fig


def get_rank():
    if not dist.is_available():
        return 0

    if not dist.is_initialized():
        return 0

    return dist.get_rank()


def requires_grad(model, flag=True):
    for parameters in model.parameters():
        parameters.requires_grad = flag


def weights_ema(model1, model2, decay=0.999):
    """
    For visualizing and evaliating generator output at any given point during the training
    use an exponential running average for the weights of the generator with decay 0.999.
    """
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def generate_noise(batch_size, n_blocks, init_const_shape, device):
    """
    batch_size - batch size
    n_blocks - number of generator blocks
    init_const_shape - initialization constant shape: [height, width, deepth]
    """

    # Generate the noise tuple for each block
    noise = []
    for i in range(n_blocks):
        # Different noise shape for each block
        shape = (batch_size, 1) + tuple(init_const_shape * 2 ** i)
        noise.append(
            (
                torch.randn(shape, device=device),
                torch.randn(shape, device=device)
            )
        )
    return noise


def generate_z(batch_size, d_latent, style_mixing_prob, device):
    """
    batch_size - batch size
    d_latent - the size of the latent space
    style_mixing_prob - probability of geheration of two latent codes
    device - storage memory
    """

    if style_mixing_prob > 0 and random.random() < style_mixing_prob:
        # Generate two latent codes
        return [torch.randn(batch_size, d_latent, device=device),
                torch.randn(batch_size, d_latent, device=device)]

    else:
        # Generate one latent code
        return [torch.randn(batch_size, d_latent, device=device)]


def mixing_regularization(w, n_blocks):
    """
    w - the style code(s)
    n_blocks - number of generator blocks
    """

    if len(w) != 1:
        # Generate a style separation index for mixing regularization
        sep_index = random.randint(1, n_blocks - 1)
        # To the block numbered "sep_index" is taken w1
        w1 = w[0].unsqueeze(0).repeat(sep_index, 1, 1)
        # The next blocks after "sep_index" take w2
        w2 = w[1].unsqueeze(0).repeat(n_blocks - sep_index, 1, 1)
        # Duplicate for each block
        return torch.cat([w1, w2], 0)
    else:
        # Duplicate for each block
        return w[0].unsqueeze(0).repeat(n_blocks, 1, 1)


def train(
        loader,
        net_M,
        net_M_ema,
        net_G,
        net_G_ema,
        net_D,
        G_opt,
        D_opt,
        batch_size,  # Batch size during training
        total_steps,  # Number of steps to learn
        plot_freq,  # Sample plot frequency
        save_freq,  # Models save frequency
        device,  # Where the calculations are made
        initial_step=1,  # Initialization step
        use_wandb=False,  # Use weights and biases logging
        r1_gamma=10.,
        # Weight of the r1 regularization (γ ∈ [γ0/5, γ0 · 5], γ0 = 0.0002 · N/M, where N = w × h is the number of pixels and M is the minibatch size)
        path_lenght_beta=0.99,  # Weight of the exponentional moving average in path lenght regularization
        path_lenght_weight=2.,  # Path lenght regularization constant
        pl_exp_sum_a=None,  # Preserved state for continued learning
        pl_steps=None,  # Preserved state for continued learning
        style_mixing_prob=0.9,  # Probability of latent code mixing
        G_reg_interval=4,  # How often the perform regularization for G? Ignored if lazy_regularization=False
        D_reg_interval=16,  # How often the perform regularization for D? Ignored if lazy_regularization=False
        WGAN_GP=False,  # Whether to use the WGAN-GP
        gp_weight=10,  # Weight of gradient penalty
        ema_decay=0.999,  # Exponentional moving average decay of weights
        calc_fid=True,  # Whether to calculate FID
        fid_samples=1000,  # Number of generations for FID calculation
):
    if get_rank() == 0 and calc_fid:
        inception_state_dict_path = r'C:\Users\Evgeniy\Jupyter\Work\stylegan2\inception_v3\checkpoint\Inception3_000007.pt'
        inception = MODEL().to(device)
        inception.load_state_dict(torch.load(inception_state_dict_path)['net'])
        inception.AuxLogits = None
        inception.dropout = Identity()
        inception.fc = Identity()
        inception.eval()

    pbar = range(total_steps)
    if get_rank() == 0:
        pbar = tqdm(pbar, initial=initial_step, dynamic_ncols=True)

    used_tanh = isinstance(net_G.activation, type(nn.Tanh()))
    n_blocks = net_G.n_blocks
    init_const_shape = np.array(net_G.initial_constant.shape[2:])
    d_latent = list(net_M.parameters())[0].shape[0]
    D_r1_loss = GradientPenalty(gamma=torch.tensor(r1_gamma).to(device))
    G_path_lenght_loss = PathLenghtPenalty(beta=torch.tensor(path_lenght_beta).to(device),
                                           weight=torch.tensor(path_lenght_weight).to(device))
    if pl_exp_sum_a and pl_steps is not None:
        G_path_lenght_loss.exp_sum_a = pl_exp_sum_a.to(device)
        G_path_lenght_loss.steps = pl_steps.to(device)

    loader = generator(loader)

    # Samples for tracking learning
    sample_z = generate_z(1, d_latent, 0, device)
    sample_noise = generate_noise(1, n_blocks, init_const_shape, device)

    for step in pbar:
        step += initial_step
        r1_loss, pl_loss, pl_exp_sum, pl_norm = 0, 0, 0, 0

        real_img = next(loader)
        real_img = real_img.to(device)

        # ---------------------- Discriminator update --------------------------------
        requires_grad(net_M, False)
        requires_grad(net_G, False)
        requires_grad(net_D, True)

        # Generate two z with probability "style_mixing_prob"
        z = generate_z(batch_size, d_latent, style_mixing_prob, device)
        # Map z into a style
        w = [net_M(z_i) for z_i in z]
        # Apply mixing reqularization (if two z are generated)
        w = mixing_regularization(w, n_blocks)
        # Generate the noise tuple for each block
        noise = generate_noise(batch_size, n_blocks, init_const_shape, device)

        fake_img = net_G(w, noise)
        fake_pred = net_D(fake_img)
        real_pred = net_D(real_img)

        if WGAN_GP:
            D_loss = D_wgan(real_pred, fake_pred) + gp_weight * gradient_penalty(net_D, real_img, fake_img)
        else:
            D_loss = D_logistic_loss(real_pred, fake_pred)

        real_score = real_pred.mean()
        fake_score = fake_pred.mean()

        D_opt.zero_grad()
        D_loss.backward()
        D_opt.step()

        # Regularization (lazy) D every "D_reg_interval" steps
        if step % D_reg_interval == 0 and WGAN_GP == False:
            real_img.requires_grad = True
            real_pred = net_D(real_img)
            r1_loss = D_r1_loss(real_img, real_pred)

            net_D.zero_grad()
            (real_pred * 0 + r1_loss).mean().mul(D_reg_interval).backward()
            D_opt.step()
        # ----------------------------------------------------------------------------

        # ---------------------- Generator update ------------------------------------
        requires_grad(net_M, True)
        requires_grad(net_G, True)
        requires_grad(net_D, False)

        # Generate two z with probability "style_mixing_prob"
        z = generate_z(batch_size, d_latent, style_mixing_prob, device)
        # Map z into a style
        w = [net_M(z_i) for z_i in z]
        # Apply mixing reqularization (if two z are generated)
        w = mixing_regularization(w, n_blocks)
        # Generate the noise tuple for each block
        noise = generate_noise(batch_size, n_blocks, init_const_shape, device)

        fake_img = net_G(w, noise)
        fake_pred = net_D(fake_img)

        if WGAN_GP:
            G_loss = G_wgan(fake_pred)
        else:
            G_loss = G_nonsaturating_loss(fake_pred)

        G_opt.zero_grad()
        G_loss.backward()
        G_opt.step()

        # Regularization (lazy) G every "G_reg_interval" steps
        if step % G_reg_interval == 0 and WGAN_GP == False:
            # Generate two z with probability "style_mixing_prob"path_lenght_loss
            z = generate_z(batch_size, d_latent, style_mixing_prob, device)
            # Map z into a style
            w = [net_M(z_i) for z_i in z]
            # Apply mixing reqularization (if two z are generated)
            w = mixing_regularization(w, n_blocks)
            # Generate the noise tuple for each block
            noise = generate_noise(batch_size, n_blocks, init_const_shape, device)
            fake_img = net_G(w, noise)
            pl_loss, pl_exp_sum, pl_norm = G_path_lenght_loss(w, fake_img)

            net_G.zero_grad()
            (fake_img[:, 0, 0, 0, 0] * 0 + pl_loss).mean().mul(G_reg_interval).backward()
            G_opt.step()

            pl_exp_sum = reduce_sum(pl_exp_sum).item() / get_world_size()
            G_path_lenght_loss.exp_sum_a = pl_exp_sum

        weights_ema(net_M_ema, net_M, ema_decay)
        weights_ema(net_G_ema, net_G, ema_decay)

        loss_dict = {
            "G_loss": G_loss,
            "D_loss": D_loss,
            "real_score": real_score,
            "fake_score": fake_score,
            "r1_loss": r1_loss,
            "pl_loss": pl_loss,
            "pl_exp_sum": pl_exp_sum,
            "pl_norm": pl_norm,
        }

        loss_reduced = reduce_loss_dict(loss_dict)

        G_loss = loss_reduced["G_loss"].mean().item()
        D_loss = loss_reduced["D_loss"].mean().item()
        real_score = loss_reduced["real_score"].mean().item()
        fake_score = loss_reduced["fake_score"].mean().item()
        r1_loss = loss_reduced["r1_loss"].mean().item()
        pl_loss = loss_reduced["pl_loss"].mean().item()
        pl_exp_sum = loss_reduced["pl_exp_sum"].mean().item()
        pl_norm = loss_reduced["pl_norm"].mean().item()
        # ----------------------------------------------------------------------------

        if get_rank() == 0:
            if WGAN_GP:
                pbar.set_description((f'D: {D_loss:.4f}; G: {G_loss:.4f}; '))
            else:
                pbar.set_description(
                    (
                        f'D: {D_loss:.4f}; G: {G_loss:.4f}; r1: {r1_loss:.4f}; '
                        f'pl: {pl_loss:.4f}; pl_exp_sum: {pl_exp_sum:.4f}; pl_norm: {pl_norm:.4f}; '
                    )
                )

            if wandb and use_wandb:
                if WGAN_GP:
                    wandb.log(
                        {
                            'Generator': G_loss,
                            'Discriminator': D_loss,
                            'Real Score': real_score,
                            'Fake Score': fake_score,
                        },
                        step=step,
                    )
                else:
                    wandb.log(
                        {
                            'Generator': G_loss,
                            'Discriminator': D_loss,
                            'R1': r1_loss,
                            'Path Length Regularization': pl_loss,
                            'Path Lenght Exponential Sum': pl_exp_sum,
                            'Path Lenght Norm': pl_norm,
                            'Real Score': real_score,
                            'Fake Score': fake_score,
                        },
                        step=step,
                    )
                # Optional
                # wandb.watch((net_M, net_G, net_D))

            if step % plot_freq == 0:
                with torch.no_grad():
                    net_M_ema.eval()
                    net_G_ema.eval()

                    sample_w = [net_M_ema(z_i) for z_i in sample_z]
                    sample_w = mixing_regularization(sample_w, n_blocks)
                    sample = net_G_ema(sample_w, sample_noise)[0, 0]

                    try:
                        if used_tanh:
                            sample = sample.cpu().numpy() * 0.5 + 0.5
                        else:
                            sample = torch.tanh(sample).cpu().numpy() * 0.5 + 0.5

                        fig_25 = data_plot(sample, param=0.25)
                        plt.close(fig_25)
                        fig_50 = data_plot(sample, param=0.5)
                        plt.close(fig_50)
                        fig_75 = data_plot(sample, param=0.75)
                        plt.close(fig_75)

                        fig_25.savefig(f'sample/{str(step).zfill(6)}_25.png')
                        fig_50.savefig(f'sample/{str(step).zfill(6)}_50.png')
                        fig_75.savefig(f'sample/{str(step).zfill(6)}_75.png')

                        # wandb.log({'sample_25': wandb.Image(fig_25)}, step=step)
                        # wandb.log({'sample_50': wandb.Image(fig_50)}, step=step)
                        # wandb.log({'sample_75': wandb.Image(fig_75)}, step=step)
                    except Exception as ex:
                        print(ex)

            if step % save_freq == 0:
                save_dict = {
                    'pl_exp_sum_a': G_path_lenght_loss.exp_sum_a.cpu(),
                    'pl_steps': G_path_lenght_loss.steps.cpu(),
                    'step': step,
                    'net_M': net_M.state_dict(),
                    'net_M_ema': net_M_ema.state_dict(),
                    'net_G': net_G.state_dict(),
                    'net_G_ema': net_G_ema.state_dict(),
                    'net_D': net_D.state_dict(),
                    'G_opt': G_opt.state_dict(),
                    'D_opt': D_opt.state_dict()
                }

                if wandb and use_wandb:
                    save_dict['wandb_run_id'] = wandb.run.id

                torch.save(
                    save_dict,
                    f'checkpoint/{str(step).zfill(6)}.pt',
                )

                if calc_fid:
                    with torch.no_grad():
                        net_M_ema.eval()
                        net_G_ema.eval()
                        fake_path = r"fakes/"
                        real_path = r"C:\Users\Evgeniy\Jupyter\Work\stylegan2\inception_v3\inception_dataset_mu_sigma.npz"

                        try:
                            os.mkdir(fake_path)
                        except FileExistsError:
                            shutil.rmtree(fake_path)
                            os.mkdir(fake_path)

                        for batch in range(fid_samples // batch_size + 1):
                            z = generate_z(batch_size, d_latent, style_mixing_prob, device)
                            w = [net_M_ema(z_i) for z_i in z]
                            w = mixing_regularization(w, net_G_ema.n_blocks)
                            noise = generate_noise(batch_size, net_G_ema.n_blocks, init_const_shape, device)
                            fake = net_G_ema(w, noise)
                            fake = torch.where(fake >= 0, 1., -1.).cpu().numpy()
                            for j, cub in enumerate(fake):
                                name = batch * batch_size + j
                                np.save(f"{fake_path}/{name}", cub)

                        fid = calculate_fid_given_paths([real_path, fake_path], batch_size, device, model=inception)
                        shutil.rmtree(fake_path)

                    if wandb and use_wandb:
                        wandb.log({'FID': fid}, step=step)
