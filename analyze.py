from utils import load_initialize, save_params, load_from_result
from data import get_dataset
from model import get_model
from xai import analyze

def main(param_file="train.yml", from_result=None):
    if from_result is not None:
        params = load_from_result(from_result)
    else:
        params = load_initialize(param_file)
    print(params)

    main_params = params["main"]
    data = get_dataset(main_params, **params["data"])
    model = get_model(main_params, **params["model"])

    analyze(main_params, data, model, **params["run"])

if __name__ == "__main__":
    main()
