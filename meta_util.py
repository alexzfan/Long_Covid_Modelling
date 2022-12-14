"""Utilities for scoring the model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def score(logits, labels):
    """Returns the mean accuracy of a model's predictions on a set of examples.

    Args:
        logits (torch.Tensor): model predicted logits
            shape (examples, classes)
        labels (torch.Tensor): classification labels from 0 to num_classes - 1
            shape (examples,)
    """

    assert logits.dim() == 2
    assert labels.dim() == 1
    assert logits.shape[0] == labels.shape[0]
    y = torch.argmax(logits, dim=-1) == labels
    y = y.type(torch.float)
    return torch.mean(y).item()

def increase_image_channels(images, num_out_channels, device):
    """Updates an image with updated number of channels to feed into a pretrained model

    Args:
        image (torch.Tensor): batch image
            shape (B, C, H, W)
        num_out_channels: int
    """
    temp = torch.empty((images.size(0), num_out_channels, images.size(2), images.size(3)))

    image_mean = torch.mean(images, axis = 1)
    for i in range(num_out_channels):
        if i < images.size(1):
            temp[:, i, :, :] = images[:, i, :, :]
        else:
            temp[:, i, :, :] = image_mean
    
    return temp.to(device)

class aug_net_block(nn.Module):

    def __init__(
        self,
        in_channel,
        out_channel,
        aug_noise_prob,
        num_augs
    ):
        """Inits the augmentation network for MetaAugNet on MAML"""
        super(aug_net_block, self).__init__()

        self.lin_param = nn.Parameter(nn.init.normal_(
                    torch.empty(
                        out_channel,
                        in_channel,
                        requires_grad=True,
                        device = DEVICE
                    ),
                    mean =0,# 0.000001
                    std = 1e-8
                ))
        self.lin_bias = nn.Parameter(nn.init.zeros_(
                    torch.empty(
                        out_channel,
                        requires_grad=True,
                        device = DEVICE
                    )
                ))
        self.lin_identity_weight = nn.init.eye_(
            torch.empty(
                out_channel, 
                in_channel, 
                requires_grad = False,
                device = DEVICE
                )
            )

        self.aug_noise_prob = aug_noise_prob
        self.num_augs = num_augs

    def forward(self, x):
        """x: input image (N*S, C, H, W)"""
        res =  F.linear(input = x, weight = self.lin_identity_weight, bias = None)
        x = F.linear(
            input = x,
            weight = self.lin_param,
            bias = self.lin_bias
        )
        B, L = x.size()
        tB = int(B / self.num_augs)
        # new way of generating augs
        noise = torch.cat([nn.init.normal_(torch.empty((tB, L), 
                                             requires_grad = False, 
                                             device = DEVICE), 
                                 mean = 0, 
                                 std = 0.1*torch.std(x.detach()).item()
                                ) if random.uniform(0,1) < self.aug_noise_prob else nn.init.zeros_(torch.empty((tB, L), 
                                             requires_grad = False, 
                                             device = DEVICE)
                                ) for _ in range(self.num_augs)
                ], 
                dim = 0
                )
        assert noise.size() == x.size()
        x = x + noise
        x = F.dropout(x, 0.4)
        x = torch.clamp(x, min=0)
        return x + res

class mean_pool_along_channel(nn.Module):
    def __init__(self):
        super(mean_pool_along_channel, self).__init__()

    def forward(self, x):
        assert len(x.shape) == 3
        return torch.mean(x, dim = [2])


class manual_relu(nn.Module):

    def __init__(
        self
    ):

        super(manual_relu, self).__init__()

    def forward(self, x):
        return torch.max(x, 0)



    