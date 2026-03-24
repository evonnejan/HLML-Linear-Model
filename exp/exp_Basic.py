import os
import torch
import numpy as np


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        use_gpu = bool(getattr(self.args, 'use_gpu', False))
        use_multi_gpu = bool(getattr(self.args, 'use_multi_gpu', False))

        if use_gpu and torch.cuda.is_available():
            if use_multi_gpu:
                devices = getattr(self.args, 'devices', str(getattr(self.args, 'gpu', 0)))
                os.environ["CUDA_VISIBLE_DEVICES"] = str(devices)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(getattr(self.args, 'gpu', 0))

            gpu_id = int(getattr(self.args, 'gpu', 0))
            device = torch.device(f'cuda:{gpu_id}')
            print(f'Use GPU: cuda:{gpu_id}')
            return device

        mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        if use_gpu and mps_available:
            device = torch.device('mps')
            print('Use GPU: mps')
            return device

        device = torch.device('cpu')
        if use_gpu and not torch.cuda.is_available() and not mps_available:
            print('GPU requested but CUDA/MPS unavailable, fallback to CPU')
        else:
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass