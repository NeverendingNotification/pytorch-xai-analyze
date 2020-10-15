import os

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn

DEFAULT_LOG_DIR = "log"
DEFAULT_MODEL_NAME = "model"
OPT_DICT = {
    "adam":[torch.optim.Adam, 1e-3]
}

LOSS_DICT = {
    "ce": nn.CrossEntropyLoss
}

def run(main_params, data, model, run_type="train", **params):
    if run_type == "train":
        train(main_params, data, model, **params)
    elif run_type == "analyze":
        pass
        # analyze(main_params, data, model, **params)
    else:
        raise NotImplementedError()

def train(main_params, data, model, num_epochs=30, opt_type="adam", loss_type="ce", lr_scheduler=None, grad_accum=None):
    log_dir = main_params.get("log_dir", DEFAULT_LOG_DIR)
    os.makedirs(log_dir, exist_ok=True)
    model_path = get_model_path(main_params)
    device = main_params.get("device", "cpu")

    # optimzier setting
    opt_func, lr0 = OPT_DICT[opt_type]
    optimizer = opt_func(model.parameters(), lr=lr0)
    optimizer.zero_grad()
    # loss setting
    criterion = LOSS_DICT[loss_type]()
    # learning rate schedule
    scheduler, accum_func = get_learing_control(optimizer, num_epochs, lr_scheduler, grad_accum)

    assert "train" in data
    train_loader = data["train"]
    validation = "valid" in data

    logs = []
    for epoch in range(1, num_epochs + 1):
        prg = tqdm(train_loader)
        model.train()
        prg.set_description("train　Epoch {}".format(epoch))

        num_accum = accum_func(epoch)
        loss_factor = 1.0 / num_accum
        it, count, hit = 0, 0, 0
        losses = []
        for batch in prg:
            imgs, labels = [b.to(device) for b in batch]
            preds = model(imgs)
            loss = criterion(preds, labels) * loss_factor
            loss.backward()
            # calculate train accuracy
            it += 1
            count += len(labels)
            hit += (preds.argmax(dim=1) == labels).sum().item()
            # gradient accumulation
            if it % num_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            losses.append(loss.item() / loss_factor)
            
        train_acc = hit / count
        train_loss = np.mean(losses)
        log_epoch = [epoch, train_loss, train_acc]
        if validation:
            valid_loss, valid_df = evaluation(model, data["valid"], criterion, device)
            try:
                valid_df.to_csv(os.path.join(log_dir, "valid.csv"))
            except:
                print("cannot write valid.csv")
            valid_acc = (valid_df["true"]==valid_df["pred"]).mean()
            print("Epoch {} , Loss train / {:.3f} valid / {:.3f}, Accuracy  train /{:.3f} valid {:.3f}".format(epoch, train_loss, valid_loss, train_acc, valid_acc))
            logs.append((epoch, train_loss, train_acc,  valid_loss, valid_acc))
        else:
            print("Epoch {} , Loss train / {:.3f} valid / {:.3f}, Accuracy  train /{:.3f} valid {:.3f}".format(epoch, train_loss, valid_loss, train_acc, valid_acc))
            logs.append((epoch, train_loss, valid_loss))
        logs.append(log_epoch)
        scheduler.step()
    torch.save(model.state_dict(), model_path)
    
    # visualization logs
    log_col = ["Epoch", "train_loss", "train_acc"]
    if validation:
        log_col.extend(["valid_loss", "valid_acc"])
    logs_df = pd.DataFrame(logs, columns=log_col).set_index("Epoch")
    logs_df.to_csv(os.path.join(log_dir, "logs.csv"))

    fig, _ = plt.subplots(figsize=(15, 4), ncols=2)
    logs_df[["train_loss", "valid_loss"]].plot(ax=fig.axes[0], title="loss")
    logs_df[["train_acc", "valid_acc"]].plot(ax=fig.axes[1], title="accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(log_dir, "lr_curve.jpg"))

    return logs_df


def get_model_path(main_params):
    log_dir = main_params.get("log_dir", DEFAULT_LOG_DIR)
    model_name = main_params.get("model_name", DEFAULT_MODEL_NAME)
    model_path = os.path.join(log_dir, model_name)
    return model_path


def evaluation(model, valid_loader, criteria, device):
    model.eval()
    prg = tqdm(valid_loader)
    prg.set_description("valid")
    results = [[], []]
    losses = []
    for batch in prg:
        imgs, labels = [b.to(device) for b in batch]
        with torch.no_grad():
            preds = model(imgs)
            loss = criteria(preds, labels)
            losses.append(loss.item())

        results[0].append(labels.cpu().numpy())    
        results[1].append(preds.argmax(dim=1).cpu().numpy())

    valid_loss = np.mean(losses)
    result_df = pd.DataFrame(np.array([np.concatenate(col) for col in results]).T, columns=["true", "pred"])
    return valid_loss, result_df


def get_learing_control(optimizer, num_epochs, lr_scheduler, grad_accum):
    if lr_scheduler is not None:
        assert isinstance(lr_scheduler, dict)
        scheduler = get_sheduler(optimizer, num_epochs, *lr_scheduler)
    else:
        class NoOpClass(object):
            def step(self):
                pass
        scheduler = NoOpClass()
    # accumulation
    if grad_accum is not None:
        assert isinstance(grad_accum, dict)
        accum_func = get_accum(num_epochs, ** grad_accum)
    else:
        accum_func = lambda epoch: 1
    return scheduler, accum_func

def get_sheduler(optimizer, num_epochs, lr_type="step", step_ratio=0.85, after_step=0.1):
    if lr_type == "step":
        def lr_func(epoch):
            if epoch < num_epochs * step_ratio:
                return 1.0
            else:
                return after_step
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)
    else:
        raise NotImplementedError(lr_type)
    return scheduler

def get_accum(num_epochs, accum_type="step", step_ratio=0.85, after_step=10):
    if accum_type == "step":
        def accum_func(epoch):
            if epoch < num_epochs * step_ratio:
                return 1
            else:
                return after_step
    else:
        raise NotImplementedError(accum_type)
    return accum_func



