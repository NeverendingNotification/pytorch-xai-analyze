

from utils import load_initialize, save_params
from data import get_dataset
from model import get_model
from runner import run

def main(param_file="train.yml"):
    params = load_initialize(param_file)
    print(params)

    main_params = params["main"]
    data = get_dataset(main_params, **params["data"])
    model = get_model(main_params, **params["model"])

    save_params(params)
    run(main_params, data, model, **params["run"])

if __name__ == "__main__":
    main()
