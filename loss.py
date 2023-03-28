import numpy as np
import torch
import math
import torch.nn.functional as F
from torch import nn


def G_wgan(fake_pred):
    return -fake_pred.mean()


def D_wgan(real_pred, fake_pred):
    return fake_pred.mean() - real_pred.mean()


def gradient_penalty(net_D, real_img, fake_img):
    batch_size = real_img.shape[0]
    # Get the interpolated image
    epsilon = torch.rand(batch_size, 1, 1, 1, 1).to(real_img.device)
    interpolated = epsilon * real_img + (1 - epsilon) * fake_img
    interpolated.requires_grad = True
    # Get the discriminator output for this interpolated image
    fake_pred = net_D(interpolated)
    # Calculate the gradients w.r.t to this interpolated image
    grads = torch.autograd.grad(outputs=fake_pred,
                                inputs=interpolated,
                                grad_outputs=fake_pred.new_ones(fake_pred.shape),
                                create_graph=True,
                               )[0]
    
    grads = grads.reshape(batch_size, -1)
    norm = grads.norm(2, dim=-1)
    return torch.mean((norm - 1) ** 2)


class GradientPenalty(nn.Module):
    """
    This is the R1 regularization penalty from the paper
    'Which Training Methods for GANs do actually Converge?'.
    That is we try to reduce the L2 norm of gradients of
    the discriminator with respect to images, for real images (PD).
    """
    
    def __init__(self, gamma: float):
        """
        gamma - the constant 'gamma' used to calculate
        regularization coefficient
        """
        super().__init__()
        
        self.gamma = gamma 
    
    def forward(self, x: torch.Tensor, d: torch.Tensor):
        """
        x - x ~ D
        d - D(x)
        """
        
        batch_size = x.shape[0]
        gradients, *_ = torch.autograd.grad(outputs=d,
                                            inputs=x,
                                            grad_outputs=d.new_ones(d.shape),
                                            create_graph=True)
        gradients = gradients.reshape(batch_size, -1)
        norm = gradients.norm(2, dim=-1)
        return torch.mean(norm ** 2) * (self.gamma / 2)


class PathLenghtPenalty(nn.Module):
    """
    This regularization encourages a fixed-size step in w
    to result in a fixed-magnitude change in the image.
    """
    
    def __init__(self, beta: float, weight: float):
        """
        beta - the constant 'beta' used to calculate
        the exponential moving average 'alpha'
        weight - the regularization constant
        """
        super().__init__()
        
        self.beta = beta
        self.weight = weight
        # Number of steps calculated N
        self.steps = nn.Parameter(torch.tensor(0.).to(beta.device), requires_grad=False)
        self.exp_sum_a = nn.Parameter(torch.tensor(0.).to(beta.device), requires_grad=False)
        
    def forward(self, w: torch.Tensor, x: torch.Tensor):
        """
        w - the batch of w of shape [batch_size, d_latent]
        x - the generated images of shape [batch_size, 1, height, width, deepth]
        """
        
        device = x.device
        # Get number of voxels
        image_size = x.shape[2] * x.shape[3] * x.shape[4]
        # Calculate y ~ N(0, I)
        y = torch.randn(x.shape, device=device)
        # Calculate (g(w) * y) and normalize
        output = (x * y).sum() / math.sqrt(image_size)
        # Calculate gradients to get Jacobian
        gradients, *_ = torch.autograd.grad(outputs=output,
                                            inputs=w,
                                            grad_outputs=torch.ones(output.shape, device=device),
                                            create_graph=True)
        # Calculate L2-norm of Jacobian
        norm = (gradients ** 2).sum(dim=2).mean(dim=1).sqrt()
        #norm = (gradients ** 2).sum(dim=1).mean(dim=0).sqrt()
        # Regularize after first step
        if self.steps > 0:
            #a = self.exp_sum_a / (1 - self.beta ** self.steps)
            a = self.exp_sum_a
            # Calculate the penalty
            loss = torch.mean((norm - a) ** 2) * self.weight
        else:
            # Return a dummy loss if we can't calculate 'a'
            a = self.exp_sum_a
            loss = norm.new_tensor(0)
        # Calculate the mean of Jacobian
        mean = norm.mean().detach()
        # Update exponential sum
        self.exp_sum_a.mul_(self.beta).add_(mean, alpha=1 - self.beta)
        # Increment N
        self.steps.add_(1.)
        return loss, a.detach(), mean


def D_logistic_loss(real_pred, fake_pred):
    # Maximize logits for real images
    real_loss = F.softplus(-real_pred) # -log(sigmoid(real_pred))
    # Minimize logits for generated images
    fake_loss = F.softplus(fake_pred) # -log(1 - sigmoid(fake_pred))
    return real_loss.mean() + fake_loss.mean()


def G_nonsaturating_loss(fake_pred):
    # Maximize logits for generated images
    loss = F.softplus(-fake_pred) # -log(sigmoid(fake_pred))
    return loss.mean()