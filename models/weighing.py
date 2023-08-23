import torch
import torch.nn as nn

import numpy as np
import os

class ViewWeightModule(nn.Module):
    def __init__(self, num_views, width, height):
        super(ViewWeightModule, self).__init__()
        self.num_views = num_views
        self.width = width
        self.height = height

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        x, y = torch.meshgrid([torch.arange(0, width, dtype=torch.float32, device=device),
                                     torch.arange(0, height, dtype=torch.float32, device=device)])
        self.input = torch.stack((x, y), dim=-1).unsqueeze(0).permute((0, 3, 1, 2))
        self.input.to(device)
        
        self.ffn = nn.Sequential(
            nn.Conv2d(2, num_views*3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_views*3, num_views*3, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_views*3, num_views-1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.ffn.to(device)

    def forward(self):
        return self.ffn(self.input).squeeze(0)

class Weighing(nn.Module):
    def __init__(self, num_views):
        super(Weighing, self).__init__()
        self.num_views = num_views

        self.networks = [
            ViewWeightModule(num_views, 96, 96) for _ in range(num_views)
        ]

        self.mappings = [
            [i] + [src for src in range(num_views) if src != i] for i in range(num_views)
        ]

    def forward(self, ref_cam_index, width, height):
        # ref_cam_index is Bx1
        B = ref_cam_index.shape[0]
        out = torch.stack([self.networks[index]() for index in ref_cam_index], dim=0)

        # upsample
        out = nn.functional.interpolate(out, size=(width, height), mode='bilinear', align_corners=False)

        # save to disk
        os.makedirs('weights_experiment/', exist_ok=True)
        np.save(f'weights_experiment/weights{ref_cam_index}.npy', out.cpu().detach().numpy())

        return out, [self.mappings[i] for i in ref_cam_index]