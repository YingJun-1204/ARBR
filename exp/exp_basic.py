import os
import torch
from models.model_factory import ModelFactory


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
    
        self.model_dict = {
            'SplatTS': ModelFactory
        }
        
        self.device = self._acquire_device()
        # 这里会触发子类 Exp_Main 的 _build_model()
        self.model = self._build_model().to(self.device)

        # total_params = sum(p.numel() for p in self.model.parameters())
        # trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print(f"[Model Parameter Count] Total Parameters: {total_params:,} | Trainable Parameters: {trainable_params:,}")

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
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