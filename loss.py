import torch
from torch import conv2d, nn, autograd
from torch.nn import functional as F

#WS
def grad_penalty(D, x_real, x_fake, batch_size, n_ch, device):
    t = torch.rand(batch_size, n_ch, 512, 512).to(device)
    t = t.expand_as(x_real)
    mid = t * x_real + (1 - t) * x_fake
    mid.requires_grad_()
    pred = D(mid)
    grad = autograd.grad(outputs=pred, inputs=mid, grad_outputs=torch.ones_like(pred), create_graph=True, retain_graph=True, only_inputs=True)[0]
    gp = torch.pow(grad.norm(2,dim=1) - 1, 2).mean()
    return gp 

def dis_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)
    return real_loss.mean() + fake_loss.mean()

def gen_loss(fake_pred):
    output = F.softplus(-fake_pred).mean()
    output.requires_grad_()
    return output

