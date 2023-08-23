import torch
import math
import torch.nn as nn
import torch.nn.functional as F

import json
import numpy as np
from scipy.spatial.transform import Rotation as R


class RelativePositionalEncoder(nn.Module):
    def __init__(self, emb_dim, max_position=512):
        super(RelativePositionalEncoder, self).__init__()
        self.max_position = max_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_position * 2 + 1, emb_dim))

        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, seq_len_q, seq_len_k):
        with torch.no_grad():
           # seq_len_q, seq_len_k = seq_len_q // upsample, seq_len_k // upsample
            range_vec_q = torch.arange(seq_len_q)
            range_vec_k = torch.arange(seq_len_k)
            relative_matrix = range_vec_k[None, :] - range_vec_q[:, None]
            clipped_relative_matrix = torch.clamp(relative_matrix, -self.max_position, self.max_position)
            relative_position_matrix = clipped_relative_matrix + self.max_position
            embeddings = self.embeddings_table[relative_position_matrix]
            
            # nearest neighbor upsample embeddings
            embeddings = embeddings.unsqueeze(0).permute(0, 3, 1, 2)
            embeddings = F.interpolate(embeddings, scale_factor=1, mode='nearest').squeeze(0)

            return embeddings
        
class WPEG(nn.Module):
    def __init__(self, data_root, input_dim, hypothesis_range, channels):
        assert 768 % input_dim == 0

        super(WPEG, self).__init__()
        self.hypothesis_range = hypothesis_range
        positional_encoder = RelativePositionalEncoder(channels, max_position=512)
        self.scale_factor = input_dim / 768
        self.map = positional_encoder(256*8, 256*8)
        self.cams = json.load(open(f"{data_root}/cams.json"))

        self.input_dim = input_dim

        self.channels = channels

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            self.embeddings = [
                self.get_positional_projections(i).to(device) for i in range(6)
            ]

        self.conv0 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.bnorm0 = nn.BatchNorm2d(channels)
        self.swish = nn.SiLU()

        print("Done initializing WPEG")

    def forward(self, cam_indices: list, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)

        with torch.no_grad():
            embedding = torch.cat([self.embeddings[i].unsqueeze(0) for i in cam_indices], dim=0)

        x = torch.cat([x, embedding], dim=1)
        x = self.conv0(x)
        x = self.bnorm0(x)
        x = self.swish(x)

        return x
        

    def get_positional_projections(self, cam_index: int):
        intr, extr = np.array(self.cams[str(cam_index)]["intrinsic"]), np.array(self.cams[str(cam_index)]["extrinsic"])

        intr[0, 0] *= self.scale_factor
        intr[1, 1] *= self.scale_factor
        intr[:2, 2] *= self.scale_factor

        steps = np.arange(start=self.hypothesis_range[0], stop=self.hypothesis_range[1], step=(self.hypothesis_range[1] - self.hypothesis_range[0]) / self.channels)

        zz = torch.zeros((len(steps), self.input_dim, self.input_dim))

        for i, step in enumerate(steps):
            zz[i, :, :] = torch.Tensor(self.warp_image(intr, extr, step, self.map[i, :, :]))

        return zz

    def warp_image(self, intr, extr, plane_height, image):
       # plane_height -= 200
        extr = extr.copy()
        extr[:3, 2] *= -1

        intr_inv = np.linalg.inv(intr)

        xx, yy = np.meshgrid(np.arange(self.input_dim), np.arange(self.input_dim))
        xx, yy = xx.reshape(-1), yy.reshape(-1)

        coords = np.vstack((xx, yy, np.ones_like(xx)))
        world_coords = intr_inv @ coords
        world_coords /= world_coords[2, :]
        world_coords = np.vstack((world_coords, np.ones_like(xx)))
        world_coords = extr @ world_coords

        camera_center = world_coords

        #camera_direction = extr[:3, 2]
        pixel_directions = world_coords[:3, :] - extr[:3, 3][:, None]
        pixel_directions /= np.linalg.norm(pixel_directions, axis=0)

        
        plane_normal = np.array([0, 0, -1], dtype=float)
        plane_point = np.array([0, 0, plane_height], dtype=float)
        plane_point = plane_point[:, None]

        normal_times_point = np.dot(plane_normal, plane_point)
        normal_times_line = np.dot(plane_normal, camera_center[:3, :])
        
        distance_to_plane = (-normal_times_point - normal_times_line) / np.dot(plane_normal, pixel_directions) + plane_height

        distance_to_plane = distance_to_plane
        

        # calculate intersection point on plane coordinates
        intersection_point = camera_center[:3, :] + distance_to_plane.T * pixel_directions
        points = intersection_point.reshape(3, self.input_dim, self.input_dim)[:2, :, :]

        h, w = image.shape
        points[0, :] += h // 2
        points[1, :] += w // 2

        sampled = image[points[1, :].astype(int), points[0, :].astype(int)]

        return sampled
    
def pos2posemb3d(pos, num_pos_feats=128, fourier_type='exponential', temperature=10000, max_freq=8):
    if fourier_type == 'exponential':
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        pos_embed = []
        for i in range(pos.shape[-1]):
            pos_x = pos[..., i, None] / dim_t
            pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
            pos_embed.append(pos_x)

        pos_embed = torch.cat(pos_embed, dim=-1)

    elif fourier_type == 'linear':
        min_freq = 1.0
        # Nyquist frequency at the target resolution:
        if isinstance(max_freq, int):
            max_freq = [max_freq for _ in range(pos.shape[-1])]
        else:
            assert len(max_freq) == pos.shape[-1]
        freq_bands = [
            torch.linspace(start=min_freq, end=freq, steps=num_pos_feats // 2, device=pos.device) for freq in max_freq
        ]
        freq_bands = torch.stack(freq_bands, dim=0).repeat(*pos.shape[:-1], 1, 1)

        # Get frequency bands for each spatial dimension.
        # Output is size [n, d * num_bands]
        pos_embed = pos[..., None] * freq_bands
        pos_embed = pos_embed.flatten(-2)

        # Output is size [n, 2 * d * num_bands]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pos_embed = torch.cat([torch.sin(np.pi * pos_embed), torch.cos(np.pi * pos_embed)], dim=-1).to(device)
    return pos_embed

class FourierMLPEncoding(nn.Module):

    def __init__(self,
                 input_channels=10,
                 hidden_dims=[1024],
                 embed_dim=256,
                 fourier_type='exponential',
                 fourier_channels=-1,
                 temperature=10000,
                 max_frequency=64):
        super(FourierMLPEncoding, self).__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        self.fourier_type = fourier_type
        self.fourier_channels = fourier_channels
        self.temperature = temperature
        self.max_frequency = max_frequency

        start_channels = fourier_channels if fourier_channels > 0 else input_channels

        mlp = []
        for l, (in_channel, out_channel) in enumerate(zip([start_channels] + hidden_dims, hidden_dims + [embed_dim])):
            mlp.append(nn.Linear(in_channel, out_channel))
            if l < len(hidden_dims):
                mlp.append(nn.GELU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, pos):
        if self.fourier_channels > 0:
            pos = pos2posemb3d(
                pos,
                num_pos_feats=self.fourier_channels // self.input_channels,
                fourier_type=self.fourier_type,
                temperature=self.temperature,
                max_freq=self.max_frequency)
        return self.mlp(pos)
        
class VPEG(nn.Module):
    def __init__(self, data_root, input_dim, channels):
        super().__init__()

       # assert 768 % input_dim == 0

        self.scale_factor = input_dim / 768
        self.channels = channels

        self.cams = json.load(open(f"{data_root}/cams.json"))

        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.embeddings = [
            self.get_positional_projections(i) for i in range(6)
        ]

        self.fourier_enc = FourierMLPEncoding(input_channels=10, hidden_dims=[1024], embed_dim=channels, fourier_type='linear', fourier_channels=10*2*64, temperature=10000, max_frequency=16)

    def forward(self, cam_indices):
        B = len(cam_indices)
        H, W = self.input_dim, self.input_dim

        positions = torch.stack([self.get(i) for i in cam_indices], dim=0).to(self.device)

        positions = positions.permute(0, 2, 3, 1)
        positions = positions.reshape(B * H * W, -1)

        out = self.fourier_enc.forward(positions)

        out = out.reshape(B, H*W, self.channels)#.permute(0, 2, 1)
        return out

        

    def get(self, cam_index: int):
        directions, q, t = self.embeddings[cam_index]

        directions = directions.view(3, -1)

        q_repeated = q.unsqueeze(1).repeat(1, self.input_dim**2)
        t_repeated = t.unsqueeze(1).repeat(1, self.input_dim**2)


        out = torch.cat([directions, q_repeated, t_repeated], dim=0).to(self.device)

        return out.view(-1, self.input_dim, self.input_dim)
        

    def get_positional_projections(self, cam_index: int):
        intr, extr = np.array(self.cams[str(cam_index)]["intrinsic"]), np.array(self.cams[str(cam_index)]["extrinsic"])

        intr[0, 0] *= self.scale_factor
        intr[1, 1] *= self.scale_factor
        intr[:2, 2] *= self.scale_factor

        directions, q, t = self.warp_image(intr, extr)

        return directions, q, t

    def warp_image(self, intr, extr):

        intr_inv = np.linalg.inv(intr)
        extr = extr.copy()
        
        extr[:3, :3] = extr[:3, :3] @ R.from_euler('z', 180, degrees=True).as_matrix()

        xx, yy = np.meshgrid(np.arange(self.input_dim), np.arange(self.input_dim))
        xx, yy = xx.reshape(-1), yy.reshape(-1)

        coords = np.vstack((xx, yy, np.ones_like(xx)))
        world_coords = intr_inv @ coords
        world_coords /= world_coords[2, :]
        world_coords = np.vstack((world_coords, np.ones_like(xx)))
        world_coords = extr @ world_coords

        pixel_directions = world_coords[:3, :] - extr[:3, 3][:, None]
        pixel_directions /= np.linalg.norm(pixel_directions, axis=0)

        rotmat = extr[:3, :3]

        # use scipy to transform rotmat to quaternion
        quat = torch.Tensor(R.from_matrix(rotmat).as_quat())


        directions = -torch.Tensor(pixel_directions.reshape(3, self.input_dim, self.input_dim))

        translation = torch.Tensor(extr[:3, 3].flatten() / np.linalg.norm(extr[:3, 3]))

        directions = directions.to(self.device)
        quat = quat.to(self.device)
        translation = translation.to(self.device)

        return directions, quat, translation