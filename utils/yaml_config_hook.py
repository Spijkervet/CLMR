import os
import yaml


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files (I like using config files
    insteaad of using argparser)
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg

def post_config_hook(args, _run):
    if len(_run.observers) > 1:
        out_dir = _run.observers[1].dir
    else:
        out_dir = _run.observers[0].dir

    args.out_dir = out_dir

    # create TensorBoard name
    lin_txt = ""
    if args.lin_eval:
        lin_txt = "-lineval"

    tb_str = f"{args.domain}-{args.model_name}-{args.sample_rate}-{args.batch_size}bs-{args.projection_dim}proj-{args.temperature}temp-{lin_txt}"
    tb_dir = os.path.join(args.out_dir, tb_str) # _run.experiment_info["name"]
    args.tb_dir = tb_dir
    os.makedirs(args.tb_dir)
    return args
