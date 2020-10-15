
from utils import load_initialize, save_params
from data import get_dataset
from model import get_model

def run(param_file="setting.yml", mode="train"):
    params = load_initialize(param_file)

    main_params = params["main"]
    print("Loading Data")
    data = get_dataset(main_params, **params["data"])
    print("Configure Model")
    model = get_model(main_params, **params["model"])

    save_params(params)
    print("Run : ", mode)
    if mode == "train":
        from train import train_model
        train_model(main_params, data, model, **params["train"])
    elif mode == "analyze":
        from xai import analyze_model
        analyze_model(main_params, data, model, **params["analyze"])
    else:
        raise NotImplementedError(mode)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--param-file", type=str, default="setting.yml")
    parser.add_argument("--mode", type=str, choices=["train", "analyze"], default="train")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    run(param_file=args.param_file, mode=args.mode)

if __name__ == "__main__":
    main()
