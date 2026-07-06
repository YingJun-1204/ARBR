import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from exp.exp_main import Exp_Main

def get_parser():
    parser = argparse.ArgumentParser(description='Phase Shift Robustness Evaluation (Variant 1) for FEDformer')

    # basic config
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--task_id', type=str, default='test', help='task id')
    parser.add_argument('--model', type=str, default='FEDformer',
                        help='model name, options: [FEDformer, Autoformer, Informer, Transformer]')

    # supplementary config for FEDformer model
    parser.add_argument('--version', type=str, default='Fourier',
                        help='for FEDformer, there are two versions to choose, options: [Fourier, Wavelets]')
    parser.add_argument('--mode_select', type=str, default='random',
                        help='for FEDformer, there are two mode selection method, options: [random, low]')
    parser.add_argument('--modes', type=int, default=64, help='modes to be selected random 64')
    parser.add_argument('--L', type=int, default=3, help='ignore level')
    parser.add_argument('--base', type=str, default='legendre', help='mwt base')
    parser.add_argument('--cross_activation', type=str, default='tanh',
                        help='mwt cross atention activation function tanh or softmax')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, '
                             'S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, '
                             'b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--detail_freq', type=str, default='h', help='like freq, but use in predict')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', default=[24], help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=3, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1', help='device ids of multi gpus')

    # phase shift evaluation specific
    parser.add_argument("--shifts", type=str, default="0,1,2,3,4,5,6,8,10,12", help="Comma-separated shift steps to test")

    return parser

def evaluate_shift(model, test_data, seq_len, label_len, pred_len, delta_t, batch_size, device, features):
    model.eval()
    
    total_len = len(test_data)
    max_index = total_len - delta_t
    
    preds = []
    trues = []
    
    f_dim = -1 if features == "MS" else 0
    
    with torch.no_grad():
        for i in range(0, max_index, batch_size):
            batch_x_list = []
            batch_y_list = []
            batch_x_mark_list = []
            batch_y_mark_list = []
            
            for j in range(i, min(i + batch_size, max_index)):
                # Get unshifted time stamps from index j
                _, _, seq_x_mark, seq_y_mark = test_data[j]
                # Get shifted signal values from index j + delta_t
                seq_x, seq_y, _, _ = test_data[j + delta_t]
                
                batch_x_list.append(seq_x)
                batch_y_list.append(seq_y)
                batch_x_mark_list.append(seq_x_mark)
                batch_y_mark_list.append(seq_y_mark)
            
            if not batch_x_list:
                break
                
            batch_x = torch.tensor(np.array(batch_x_list), dtype=torch.float32).to(device)
            batch_y = torch.tensor(np.array(batch_y_list), dtype=torch.float32).to(device)
            batch_x_mark = torch.tensor(np.array(batch_x_mark_list), dtype=torch.float32).to(device)
            batch_y_mark = torch.tensor(np.array(batch_y_mark_list), dtype=torch.float32).to(device)
            
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            
            # Forward pass
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Slice output to prediction window
            outputs = outputs[:, -pred_len:, f_dim:]
            batch_y = batch_y[:, -pred_len:, f_dim:]
            
            preds.append(outputs.cpu().numpy())
            trues.append(batch_y.cpu().numpy())
            
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # Calculate MSE and MAE
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    return mse, mae

def main():
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = get_parser()
    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    device = torch.device(f"cuda:{args.gpu}" if args.use_gpu else "cpu")
    print(f"Using device: {device}")

    # Reconstruct the settings path
    ii = 0
    setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_id,
        args.model,
        args.mode_select,
        args.modes,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.d_model,
        args.n_heads,
        args.e_layers,
        args.d_layers,
        args.d_ff,
        args.factor,
        args.embed,
        args.distil,
        args.des,
        ii
    )

    # Initialize experiment
    exp = Exp_Main(args)
    
    # Load model state dict
    checkpoint_path = os.path.join(args.checkpoints, setting, "checkpoint.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    exp.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    exp.model.to(device)
    
    # Load test dataset
    test_data, _ = exp._get_data(flag="test")
    print(f"Loaded test dataset: {args.data} (Size: {len(test_data)})")

    # Shift steps
    shift_steps = [int(x.strip()) for x in args.shifts.split(",")]
    
    print("\n" + "=" * 60)
    print(f" Evaluating Phase Shift Robustness (Variant 1) for {args.model} ")
    print("=" * 60)
    
    results = []
    base_mse = None
    
    for delta_t in shift_steps:
        mse, mae = evaluate_shift(
            exp.model, test_data, args.seq_len, args.label_len, args.pred_len, 
            delta_t, args.batch_size, device, args.features
        )
        if delta_t == 0:
            base_mse = mse
        
        rel_increase = (mse - base_mse) / base_mse if base_mse is not None else 0.0
        
        print(f"Shift Delta t = {delta_t:2d} | MSE: {mse:.6f} | MAE: {mae:.6f} | Relative MSE Increase: {rel_increase:.2%}")
        results.append((delta_t, mse, mae, rel_increase))

    # Print markdown table
    print("\n### Phase Shift Robustness (Variant 1) Results (Markdown Table)\n")
    print("| Shift Step ($\\Delta t$) | Test MSE | Test MAE | Relative MSE Increase |")
    print("|---|---|---|---|")
    for delta_t, mse, mae, rel_increase in results:
        print(f"| {delta_t} | {mse:.6f} | {mae:.6f} | {rel_increase:.2%} |")
    print()

if __name__ == "__main__":
    main()
