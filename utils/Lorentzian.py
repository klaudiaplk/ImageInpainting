import torch


def lorentzian(img_real, img_fake):
    return torch.mean(torch.log(1.0 + torch.abs(img_real - img_fake)))
