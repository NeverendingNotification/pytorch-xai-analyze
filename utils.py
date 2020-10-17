import os
import random

import yaml
import numpy as np
import torch

DEFAULT_SEED = 777
DEFAULT_PARAM_FILE = "params.yml"
DEFAULT_LOG_DIR = "log"

def load_initialize(yaml_file, **kwargs):
    params = load_params(yaml_file)
    override_params(params, **kwargs)
    initial_setting(params["main"])
    return params

def load_from_result(result_dir, **kwargs):
    yaml_file = os.path.join(result_dir, DEFAULT_PARAM_FILE)
    params = load_params(yaml_file)
    override_params(params, **kwargs)
    initial_setting(params)
    return params

def override_params(params, **kwargs):
    for key, value in kwargs.items():
        if key in params:
            params[key] = value

def load_params(param_file):
    with open(param_file) as hndl:
        params = yaml.load(hndl, Loader=yaml.Loader)
    return params

def save_params(params, out_path=None, out_name=DEFAULT_PARAM_FILE):
    if out_path is None:
        log_dir = params["main"].get("log_dir", "./")
        out_path = os.path.join(log_dir, out_name)
    with open(out_path, "w") as hndl:
        yaml.dump(params, hndl)

def initial_setting(main_params):
    log_dir = main_params.get("log_dir", DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)

    is_cuda = main_params.get("cuda", False)
    device = "cuda" if (is_cuda and torch.cuda.is_available()) else "cpu"
    main_params["device"] = device 

    seed = main_params.get("seed", DEFAULT_SEED)
    deterministic = main_params.get("deterministric", True)
    set_seed(seed, device, deterministic)

def set_seed(seed, device, deterministic):
    if isinstance(seed, int):
        seed = [seed] * 4
    assert isinstance(seed, list)

    random.seed(seed[0])
    np.random.seed(seed[1])
    torch.manual_seed(seed[2])

    if device == "cuda":
        torch.cuda.manual_seed(seed[3]) 
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False