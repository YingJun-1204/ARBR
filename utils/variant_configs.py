def apply_variant_configs(args):
    if not hasattr(args, "representation"):
        args.representation = "gs"
    return args
