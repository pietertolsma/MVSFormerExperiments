import os
import random
import json
import glob
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.spatial.transform import Rotation as R
import cv2

import scipy.io as sio

from datasets.data_io import read_pfm
from .color_jittor import ColorJitter

np.random.seed(123)
random.seed(123)


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, img, gamma):
        # gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)


# the DTU dataset preprocessed by Yao Yao (only for training)
class ClearPoseMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, crop=False, augment=False,
                 aug_args=None, height=256, width=320, patch_size=16, **kwargs):
        super(ClearPoseMVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.height = height
        self.width = width
        self.patch_size = patch_size
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        if mode != 'train':
            self.crop = False
            self.augment = False
            self.random_mask = False
        else:
            self.crop = crop
            self.augment = augment
        self.kwargs = kwargs
        self.rotated = kwargs.get("rotated", False)
        self.multi_scale = kwargs.get('multi_scale', False)
        self.multi_scale_args = kwargs['multi_scale_args']
        self.resize_scale = kwargs.get('resize_scale', 0.5)
        self.scales = self.multi_scale_args['scales'][::-1]
        self.resize_range = self.multi_scale_args['resize_range']
        self.consist_crop = kwargs.get('consist_crop', False)
        self.batch_size = kwargs.get('batch_size', 4)
        self.world_size = kwargs.get('world_size', 1)
        self.img_size_map = []
        self.split = kwargs.get("split", [95, 5])

        self.material="diffuse"

        # print("mvsdataset kwargs", self.kwargs)

        if self.augment and mode == 'train':
            self.color_jittor = ColorJitter(brightness=aug_args['brightness'], contrast=aug_args['contrast'],
                                            saturation=aug_args['saturation'], hue=aug_args['hue'])
            self.to_tensor = transforms.ToTensor()
            self.random_gamma = RandomGamma(min_gamma=aug_args['min_gamma'], max_gamma=aug_args['max_gamma'], clip_image=True)
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.depth_ranges = {}

        assert self.mode in ["train", "val", "test"]
        self.scenes = os.listdir(self.datapath)

        self.cams = self.read_cam_file()

        self.build_list()

    # def get_scan_labels(self, scan: str):
    #     return json.load(open(f"{self.datapath}/{scan}/meta.json", "r"))["labels"]

    # def get_segmentation(self, scan: str, view: int):
    #     return np.load(f"{self.datapath}/{scan}/instance/{view}.npy")
    
    # def get_backgroundmask(self, segmentation_map):
    #     return (segmentation_map != 1).astype(bool)

    def read_cam_file(self):
        out = {}

        width_offset = (640-480)//2
        scale = 768/480
        
        for scene in self.scenes:
            out[scene] = {}
            mat = sio.loadmat(f"{self.datapath}/{scene}/metadata.mat")
            cams = [cam for cam in mat.keys() if not cam.startswith("__")]
            for cam in cams:
                camera_intrinsics = mat[cam][0, 0][3]

                camera_intrinsics[0, 2] -= width_offset
                camera_intrinsics[:2, :] *= scale

                camera_extrinsics = np.identity(4)
                camera_extrinsics[:3, :] = mat[cam][0, 0][5]

                out[str(scene)][str(cam)] = {
                    "intrinsic": camera_intrinsics,
                    "extrinsic": camera_extrinsics
                }

        return out

    def build_list(self):
        self.metas = []
        scenes = self.scenes

        stride=8
        for scene in scenes:
            cams = list(self.cams[scene].keys())
            for i in range(len(cams) // (6*stride)):
                self.metas += [(scene, cams[i*6*stride], np.array(cams[i*6*stride+1:(i+1)*stride*6:stride]))]

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1

        # random img size:256~512
        if self.mode == 'train':
            barrel_num = int(len(self.metas) / (self.batch_size * self.world_size))
            barrel_num += 2
            self.img_size_map = np.arange(0, len(self.scales))

    def __len__(self):
        # return len(self.generate_img_index)
        return len(self.metas)

    def read_img(self, scene, cam):
        img = np.array(Image.open(f"{self.datapath}/{scene}/{cam}-color.png"))
        depth = np.array(Image.open(f"{self.datapath}/{scene}/{cam}-depth.png"))

        h,w = img.shape[:2]

        # Make square
        cutoff_side_size = (w - h) // 2
        img = cv2.resize(img[:, cutoff_side_size:-cutoff_side_size, :], (768, 768))
        depth = cv2.resize(depth[:, cutoff_side_size:-cutoff_side_size], (768, 768), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        return img, depth[:, :]

    def generate_stage_depth(self, depth):
        h, w = depth.shape
        depth_ms = {
            "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth
        }
        return depth_ms

    def center_crop_img(self, img, new_h=None, new_w=None):
        h, w = img.shape[:2]

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            img = img[start_h:finish_h, start_w:finish_w]
        return img

    def center_crop_cam(self, intrinsics, h, w, new_h=None, new_w=None):
        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
            new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
            return new_intrinsics
        else:
            return intrinsics

    def pre_resize(self, img, depth, intrinsic, mask, resize_scale):
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, :] *= resize_scale
        output_intrinsics[1, :] *= resize_scale

        if depth is not None:
            depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        if mask is not None:
            mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        return img, depth, output_intrinsics, mask

    def final_crop(self, img, depth, intrinsic, mask, crop_h, crop_w, offset_y=None, offset_x=None):
        h, w, _ = img.shape
       # print(f"{h}:{w} -> {crop_h}:{crop_w} and {offset_y}:{offset_x}")
        if offset_x is None or offset_y is None:
            if self.crop:
                offset_y = random.randint(0, h - crop_h)
                offset_x = random.randint(0, w - crop_w)
            else:
                offset_y = (h - crop_h) // 2
                offset_x = (w - crop_w) // 2
        cropped_image = img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w, :]

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        if depth is not None:
            cropped_depth = depth[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_depth = None

        if mask is not None:
            cropped_mask = mask[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_mask = None

        return cropped_image, cropped_depth, output_intrinsics, cropped_mask, offset_y, offset_x

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # if self.mode == 'train':
        #     np.random.shuffle(src_views)
        view_ids = np.array([ref_view] + list(src_views[:(self.nviews - 1)]))
        # view_ids = [ref_view] + src_views

        imgs = []
        depth_values = None
        mask = None
        proj_matrices = []

        offset_y = None
        offset_x = None
        if self.augment:
            fn_idx = torch.randperm(4)
            brightness_factor = torch.tensor(1.0).uniform_(self.color_jittor.brightness[0], self.color_jittor.brightness[1]).item()
            contrast_factor = torch.tensor(1.0).uniform_(self.color_jittor.contrast[0], self.color_jittor.contrast[1]).item()
            saturation_factor = torch.tensor(1.0).uniform_(self.color_jittor.saturation[0], self.color_jittor.saturation[1]).item()
            hue_factor = torch.tensor(1.0).uniform_(self.color_jittor.hue[0], self.color_jittor.hue[1]).item()
            gamma_factor = self.random_gamma.get_params(self.random_gamma._min_gamma, self.random_gamma._max_gamma)
        else:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gamma_factor = None, None, None, None, None, None

        # background_mask = self.get_backgroundmask(self.get_segmentation(scan, ref_view))
        # scaled_background_masks = {
        #     f"stage{i+1}": background_mask[::scale, ::scale] for i, scale in enumerate([8,4,2,1])
        # }


        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            # NOTE that DTU_origin/Rectified saves the images with the original size (1200x1600)
            img_filename = os.path.join(self.datapath, 
                            f"{scan}/rgbd/{vid}.png")
            
            extrinsics = np.array(self.cams[str(scan)][str(vid)]["extrinsic"])
            intrinsics = np.array(self.cams[str(scan)][str(vid)]["intrinsic"])
       #     (depth_min, d_max) = self.depth_ranges[scan][vid]


            img, depth_hr = self.read_img(scan, vid)
            depth_min = max(depth_hr.min(), 0.5)
            depth_max= 3
            depth_interval = (depth_max-depth_min) / self.ndepths
            depth_mask_hr = np.ones_like(depth_hr)
            # resize according to crop size
            img = np.asarray(img)
            w, h, _ = img.shape

            if self.mode == 'train' and self.multi_scale:
                [crop_h, crop_w] = self.scales[self.idx_map[idx] % len(self.scales)]
                enlarge_scale = self.resize_range[0] + random.random() * (self.resize_range[1] - self.resize_range[0])
                resize_scale_h = np.clip((crop_h * enlarge_scale) / h, 0.45, 1.0)
                resize_scale_w = np.clip((crop_w * enlarge_scale) / w, 0.45, 1.0)
                resize_scale = max(resize_scale_h, resize_scale_w)
            else:
                crop_h, crop_w = self.height, self.width
                resize_scale = self.resize_scale


            if resize_scale != 1.0:
                img, depth_hr, intrinsics, depth_mask_hr = self.pre_resize(img, depth_hr, intrinsics, depth_mask_hr, resize_scale)

            if i == 0:  # reference view
                while True:  # get resonable offset
                    # finally random crop

                    img_, depth_hr_, intrinsics_, depth_mask_hr_, offset_y, offset_x = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                                       crop_h=crop_h, crop_w=crop_w)
                    mask_read_ms_ = self.generate_stage_depth(depth_mask_hr_)
                    if self.mode != 'train' or np.any(mask_read_ms_['stage1'] > 0.0):
                        break

                depth_ms = self.generate_stage_depth(depth_hr_)
                mask = mask_read_ms_
                img = img_
                intrinsics = intrinsics_
                # get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

            else:
                if self.consist_crop:
                    img, depth_hr, intrinsics, depth_mask_hr, _, _ = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                     crop_h=crop_h, crop_w=crop_w,
                                                                                     offset_y=offset_y, offset_x=offset_x)
                else:
                    img, depth_hr, intrinsics, depth_mask_hr, _, _ = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                     crop_h=crop_h, crop_w=crop_w)

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            img = Image.fromarray(img)
            if not self.augment:
                imgs.append(self.transforms(img))
            else:
                img_aug = self.color_jittor(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                img_aug = self.to_tensor(img_aug)
                img_aug = self.random_gamma(img_aug, gamma_factor)
                img_aug = self.normalize(img_aug)
                imgs.append(img_aug)

        # all
        imgs = torch.stack(imgs)
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125

        proj_matrices_ms = {
            "stage1": stage0_pjmats,  # 1/8
            "stage2": stage1_pjmats,  # 1/4
            "stage3": stage2_pjmats,  # 1/2
            "stage4": proj_matrices  # 1/1
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "cam_indices": list(view_ids.astype(int)),
                "mask": mask}