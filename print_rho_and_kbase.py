import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader

# Add project root to path to ensure imports work correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom

class MockArgs:
    def __init__(self, data, data_path, seq_len=512, pred_len=96, features='M', target='OT'):
        self.data = data
        self.data_path = data_path
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.root_path = './data'
        self.augmentation_ratio = 0
        self.num_workers = 0

def calculate_rho_and_kbase(dataset_class, args, patch_len=24, num_gaussians=8, stride=12):
    # Instantiate dataset for training
    dataset = dataset_class(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag='train',
        size=[args.seq_len, 0, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=0,
        freq='h'
    )
    
    # Use DataLoader to load all training data
    # We can compute it in batches or load the whole train set at once
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    
    total_energy_low = 0.0
    total_energy_total = 0.0
    rho_sum = 0.0
    total_samples = 0
    
    for batch_x, _, _, _ in loader:
        # batch_x shape: (B, T, C)
        B, T, C = batch_x.shape
        batch_x = batch_x.float()
        
        # Apply RevIN normalization (sample-wise, channel-wise)
        # Mean and stdev across time dimension (dim=1)
        mean = batch_x.mean(dim=1, keepdim=True)
        stdev = torch.sqrt(batch_x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (batch_x - mean) / stdev  # (B, T, C)
        
        # Permute to (B, C, T) to match Splatting input
        x_repr_in = x_norm.permute(0, 2, 1)
        x_flat = x_repr_in.reshape(B * C, T)
        
        # Calculate low-frequency trend component via average pooling
        x_seq_temp = x_flat.unsqueeze(1) # (B*C, 1, T)
        x_low = F.avg_pool1d(x_seq_temp, kernel_size=patch_len, stride=1, padding=patch_len // 2)
        if x_low.shape[-1] != x_seq_temp.shape[-1]:
            x_low = x_low[..., :x_seq_temp.shape[-1]]
            
        # Calculate energies
        energy_low = torch.sum(x_low ** 2, dim=-1)
        energy_total = torch.sum(x_seq_temp ** 2, dim=-1) + 1e-5
        
        # Calculate rho for each sequence channel
        rho = energy_low / energy_total
        rho_sum += torch.sum(rho).item()
        total_samples += (B * C)
        
    rho_mean = rho_sum / total_samples
    k_val = int(round(rho_mean * num_gaussians * (stride / patch_len)))
    max_k = max(1, int(round(num_gaussians * (stride / patch_len))))
    k_base = max(1, min(max_k, k_val))
    
    return rho_mean, k_base

def main():
    datasets = [
        {"name": "ETTh1", "class": Dataset_ETT_hour, "data_path": "ETTh1.csv"},
        {"name": "ETTh2", "class": Dataset_ETT_hour, "data_path": "ETTh2.csv"},
        {"name": "ETTm1", "class": Dataset_ETT_minute, "data_path": "ETTm1.csv"},
        {"name": "ETTm2", "class": Dataset_ETT_minute, "data_path": "ETTm2.csv"},
        {"name": "weather", "class": Dataset_Custom, "data_path": "weather.csv"},
        {"name": "electricity", "class": Dataset_Custom, "data_path": "electricity.csv"},
    ]
    
    print("\n" + "="*70)
    print(f"{'Dataset':<15} | {'Low-Freq Ratio (rho_mean)':<30} | {'k_base':<10}")
    print("="*70)
    
    for ds in datasets:
        args = MockArgs(
            data=ds["name"] if ds["name"] not in ["weather", "electricity"] else "custom",
            data_path=ds["data_path"]
        )
        try:
            rho_mean, k_base = calculate_rho_and_kbase(ds["class"], args)
            print(f"{ds['name']:<15} | {rho_mean:<30.6f} | {k_base:<10}")
        except Exception as e:
            print(f"{ds['name']:<15} | Error: {e}")
            
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
