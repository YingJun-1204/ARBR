import torch

ckpt_path = "checkpoints/long_term_forecast_etth1_jet_96_SplatTS/checkpoint.pth"
state_dict = torch.load(ckpt_path, map_location="cpu")
print("Type of checkpoint:", type(state_dict))
if isinstance(state_dict, dict):
    print("Keys in state_dict (first 10):", list(state_dict.keys())[:10])
