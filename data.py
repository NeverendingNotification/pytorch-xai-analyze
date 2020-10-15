import os
from functools import partial

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

DEFAULT_BATCH_SIZE = 32

def get_dataset(main_params, data_type="fashion-mnist", path="tmp", train=True, valid=False):
    batch_size = main_params.get("batch_size", DEFAULT_BATCH_SIZE)
    os.makedirs(path, exist_ok=True)

    # choose dataset and get dataset specific information
    ds_func, add_info = get_dataset_info(data_type)
    for key, value in add_info.items():
        main_params[key] = value
    
    # make dataloader for train and valid data
    results = {}
    mode_params = {}
    if train:
        mode_params["train"] = [train, True, True]
    if valid:
        mode_params["valid"] = [valid, False, False]

    for key, (mode, train_flag, shuffle) in mode_params.items():
        aug = mode.get("augmentation") if isinstance(mode, dict) else None
        bs_size = mode.get("batch_size", batch_size) if isinstance(mode, dict) else batch_size
        data_loader = get_dataloader(ds_func, path, aug, train=train_flag, batch_size=bs_size, shuffle=shuffle)
        results[key] = data_loader

    return results

def get_dataset_info(dstype):
    if dstype == "fashion-mnist":
        ds_func = partial(torchvision.datasets.FashionMNIST, download=True, target_transform=None)
        add_info = {"in_channel": 1, "num_classes":10, "img_size": 28}
    else:
        raise NotImplementedError(dstype)
    return ds_func, add_info

def get_dataloader(ds_func, path, augmentation=None, train=True, batch_size=DEFAULT_BATCH_SIZE, shuffle=True):
    trans_ops = []
    if isinstance(augmentation, dict):
        for key, aug in augmentation.items():
            if key == "horizontal":
                trans_ops.append(transforms.RandomHorizontalFlip(**aug))
            elif key == "vertical":
                trans_ops.append(transforms.RandomVerticalFlip(**aug))
            elif key == "rotation":
                trans_ops.append(transforms.RandomRotation(**aug))
            elif key == "resize_crop":
                trans_ops.append(transforms.RandomResizedCrop(**aug))
            else:
                raise NotImplementedError(key)
    trans_ops.append(transforms.ToTensor())
    transform = transforms.Compose(trans_ops)

    dataset = ds_func(path, train=train, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == "__main__":
    main_params = {}
    results = get_dataset(main_params)
    print(results.keys())
    print(main_params)




