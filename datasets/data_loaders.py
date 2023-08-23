import numpy as np
from torch.utils.data import DataLoader

from .general_eval import MVSDataset
import random
import torch
np.random.seed(1234)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)

class TOTELoader(DataLoader):
    def __init__(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None,
                 dataset_eval='dtu', refine=True, num_workers=4, crop=False, iterative=False,
                 augment=False, aug_args=None, height=None, width=None, data_set_type='stage4', **kwargs):
        if data_set_type == 'stage4':
            from .totemvs_dataset import TOTEMVSDataset
        elif data_set_type == 'multi_scale':
            from .totemvs_dataset import TOTEMVSDataset
        else:
            raise NotImplementedError
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = TOTEMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                             shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, crop=crop,
                                             augment=augment, aug_args=aug_args, refine=refine, height=height, width=width, **kwargs)
        else:
            self.mvs_dataset = TOTEMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                             shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, crop=crop,
                                             augment=augment, aug_args=aug_args, refine=refine, height=height, width=width, **kwargs)
        drop_last = True if mode == 'train' else False
        super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker)

        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)
    
class ClearLoader(DataLoader):
    def __init__(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None,
                 dataset_eval='dtu', refine=True, num_workers=4, crop=False, iterative=False,
                 augment=False, aug_args=None, height=None, width=None, data_set_type='stage4', **kwargs):
        if data_set_type == 'stage4':
            from .clearpose_dataset import ClearPoseMVSDataset
        elif data_set_type == 'multi_scale':
            from .clearpose_dataset import ClearPoseMVSDataset
        else:
            raise NotImplementedError
        if (mode == 'train') or (mode == 'val'):
            self.mvs_dataset = ClearPoseMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                             shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, crop=crop,
                                             augment=augment, aug_args=aug_args, refine=refine, height=height, width=width, **kwargs)
        else:
            self.mvs_dataset = ClearPoseMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                             shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, crop=crop,
                                             augment=augment, aug_args=aug_args, refine=refine, height=height, width=width, **kwargs)
        drop_last = True if mode == 'train' else False
        super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker)

        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)

class DTULoader(DataLoader):
    def __init__(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None,
                 dataset_eval='dtu', refine=True, num_workers=4, crop=False, iterative=False,
                 augment=False, aug_args=None, height=None, width=None, data_set_type='stage4', **kwargs):
        if (mode == 'train') or (mode == 'val'):

            data_list = './lists/dtu/val.txt' if mode == 'val' else './lists/dtu/train.txt'
            if data_set_type == 'stage4':
                from .dtu_dataset import DTUMVSDataset
            elif data_set_type == 'multi_scale':
                from .dtu_dataset_ms import DTUMVSDataset
            else:
                raise NotImplementedError
            self.mvs_dataset = DTUMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                             shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, crop=crop,
                                             augment=augment, aug_args=aug_args, refine=refine, height=height, width=width, **kwargs)
        else:
            self.mvs_dataset = MVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                          shuffle=shuffle, seq_size=seq_size, batch_size=batch_size,
                                          max_h=max_h, max_w=max_w, fix_res=fix_res, dataset=dataset_eval, refine=refine,
                                          iterative=iterative, **kwargs)
        drop_last = True if mode == 'train' else False
        super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker)

        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)


class BlendedLoader(DataLoader):

    def __init__(self, data_path, data_list, mode, num_srcs, num_depths, interval_scale=1.0,
                 shuffle=True, seq_size=49, batch_size=1, fix_res=False, max_h=None, max_w=None,
                 refine=True, num_workers=4, crop=False, iterative=False, augment=False, aug_args=None,
                 height=None, width=None, data_set_type='stage4', **kwargs):
        if (mode == 'train') or (mode == 'val'):
            if data_set_type == 'stage4':
                from .blended_dataset import BlendedMVSDataset
            if data_set_type == 'multi_scale':
                from .blended_dataset_ms import BlendedMVSDataset
            else:
                raise NotImplementedError
            self.mvs_dataset = BlendedMVSDataset(data_path, data_list, mode, num_srcs, num_depths, interval_scale,
                                                 shuffle=shuffle, seq_size=seq_size, batch_size=batch_size, crop=crop,
                                                 augment=augment, aug_args=aug_args, refine=refine, height=height, width=width, **kwargs)
        else:
            raise NotImplementedError
        drop_last = True if mode == 'train' else False
        super().__init__(self.mvs_dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True, drop_last=drop_last, worker_init_fn=seed_worker)

        self.n_samples = len(self.mvs_dataset)

    def get_num_samples(self):
        return len(self.mvs_dataset)
