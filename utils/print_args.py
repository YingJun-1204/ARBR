def print_args(args):
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f'  {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f'  {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f'  {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f'  {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f'  {"Target:":<20}{args.target:<20}{"Freq:":<20}{args.freq:<20}')
    print(f'  {"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['long_term_forecast', 'short_term_forecast']:
        print("\033[1m" + "Forecasting Task" + "\033[0m")
        print(f'  {"Seq Len:":<20}{args.seq_len:<20}{"Label Len:":<20}{args.label_len:<20}')
        print(f'  {"Pred Len:":<20}{args.pred_len:<20}{"Seasonal Patterns:":<20}{args.seasonal_patterns:<20}')
        print(f'  {"Inverse:":<20}{args.inverse:<20}')
        print()

    if args.task_name == 'imputation':
        print("\033[1m" + "Imputation Task" + "\033[0m")
        print(f'  {"Mask Rate:":<20}{getattr(args, "mask_rate", "N/A"):<20}')
        print()

    if args.task_name == 'anomaly_detection':
        print("\033[1m" + "Anomaly Detection Task" + "\033[0m")
        print(f'  {"Anomaly Ratio:":<20}{getattr(args, "anomaly_ratio", "N/A"):<20}')
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f'  {"Top k:":<20}{getattr(args, "top_k", "N/A"):<20}{"Num Kernels:":<20}{getattr(args, "num_kernels", "N/A"):<20}')
    print(f'  {"Enc In:":<20}{getattr(args, "enc_in", "N/A"):<20}{"Dec In:":<20}{getattr(args, "dec_in", "N/A"):<20}')
    print(f'  {"C Out:":<20}{getattr(args, "c_out", "N/A"):<20}{"d model:":<20}{getattr(args, "d_model", "N/A"):<20}')
    print(f'  {"n heads:":<20}{getattr(args, "n_heads", "N/A"):<20}{"e layers:":<20}{getattr(args, "e_layers", "N/A"):<20}')
    print(f'  {"d layers:":<20}{getattr(args, "d_layers", "N/A"):<20}{"d FF:":<20}{getattr(args, "d_ff", "N/A"):<20}')
    print(f'  {"Moving Avg:":<20}{getattr(args, "moving_avg", "N/A"):<20}{"Factor:":<20}{getattr(args, "factor", "N/A"):<20}')
    print(f'  {"Distil:":<20}{getattr(args, "distil", "N/A"):<20}{"Dropout:":<20}{getattr(args, "dropout", "N/A"):<20}')
    print(f'  {"Embed:":<20}{getattr(args, "embed", "N/A"):<20}{"Activation:":<20}{getattr(args, "activation", "N/A"):<20}')
    print(f'  {"Output Attention:":<20}{getattr(args, "output_attention", "N/A"):<20}')
    print()

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{getattr(args, "loss", "N/A"):<20}')
    print(f'  {"Lradj:":<20}{args.lradj:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()

    print("\033[1m" + "De-stationary Projector Params" + "\033[0m")
    p_hidden_dims = getattr(args, "p_hidden_dims", [])
    p_hidden_dims_str = ', '.join(map(str, p_hidden_dims)) if p_hidden_dims else "N/A"
    print(f'  {"P Hidden Dims:":<20}{p_hidden_dims_str:<20}{"P Hidden Layers:":<20}{getattr(args, "p_hidden_layers", "N/A"):<20}') 
    print()
