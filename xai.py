import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from skimage.segmentation import slic, quickshift, watershed
from alibi.explainers import AnchorImage
import shap
from gradcam import GradCAM

from utils import DEFAULT_LOG_DIR
from train import VALID_FILE, get_model_path

def analyze_model(main_params, data, model, xai_dirname="xai", algos={}, num_images=10, num_bg_images=100, gc_mod_name=None, **kwargs):
    device = main_params.get("device", "cpu")
    log_dir = main_params.get("log_dir", DEFAULT_LOG_DIR)
    img_size = main_params["img_size"]
    model_path = get_model_path(main_params)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    xai_dir = os.path.join(log_dir, xai_dirname)
    os.makedirs(xai_dir, exist_ok=True)

    valid_df = pd.read_csv(os.path.join(log_dir, VALID_FILE))

    valid_dataset = data["valid"].dataset
    class_names = valid_dataset.classes
    bg_images = torch.cat([valid_dataset[i][0][None, :] for i in valid_df.sample(num_bg_images).index]).to(device)

    if gc_mod_name is None:
        print([k for k , v in model.named_modules()])
        print("Choose grad-cam module name")
        gc_mod_name = input()

    for name, modl in model.named_modules():
        if name == gc_mod_name:
            gc_targ = modl
            break
    else:
        gc_targ = model.feature
    print("Grad-Cam target module : ", gc_targ)

    targ_indices = [valid_df[valid_df["true"] == class_].sample(num_images).index for class_ in range(len(class_names))]

    for class_ in range(len(class_names)):
        print("Analyzing : ", class_names[class_])
        targ_index = targ_indices[class_]
        targ_images = torch.cat([valid_dataset[i][0][None, :] for i in targ_index]).to(device)

        with torch.no_grad():
            probs = nn.Softmax(dim=1)(model(targ_images)).cpu().numpy()
        
        num_algos = 3
        font_size = 16
        nrows = 1 + num_algos
        ncols = num_images
        figsize = (3 * ncols, 3 * nrows)
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
        numpy_images = tensor_to_numpy(targ_images)
        for i, prob in enumerate(probs):
            idx = np.argmax(prob)
            p = prob[idx]
            axes[0, i].imshow(numpy_images[i].squeeze(), cmap="gray")
            title = "{} : {:.1f}%".format(class_names[idx], p * 100)
            axes[0, i].set_title(title, fontsize=font_size)
            axes[0, i].set_axis_off()

        for algo_index, (key, (imgs, ops)) in enumerate({
            "Anchors":analyze_anchors(targ_images, model, probs, device=device),
            "SHAP": analyze_shap(targ_images, model, probs, bg_images=bg_images),
            "GradCAM": analzye_gradcam(targ_images, model, probs, feature_mod=gc_targ)
        }.items()):
            for i, img in enumerate(imgs):
                axes[algo_index + 1, i].imshow(img.squeeze(), **ops)
                axes[algo_index + 1, i].set_title(key, fontsize=font_size)
                axes[algo_index + 1, i].set_axis_off()


        fig.tight_layout()
        fig.savefig(os.path.join(xai_dir, class_names[class_].replace("/","_") + ".jpg"))

    # anchors, anc_ops = analyze_anchors(targ_images, model, probs, device=device)
    # print(len(anchors), anchors[0].shape, anchors[0].dtype)
    # shaps, shp_ops = analyze_shap(targ_images, model, probs, bg_images=bg_images)
    # print(len(shaps), shaps[0].shape, shaps[0].dtype)
    # gc_maps, gc_ops = analzye_gradcam(targ_images, model, probs, feature_mod=model.feature[1][-1])
    # print(len(gc_maps), gc_maps[0].shape, gc_maps[0].dtype)

    # fig.savefig("tmp.jpg")






    # assert isinstance(algos, dict)
    # for key, algo in algos.items():
    #     out_dir = os.path.join(xai_dir, key)
    #     os.makedirs(out_dir, exist_ok=True)
        
    #     algo()

def superpixel(image, size=(4, 7)):
    segments = np.zeros([image.shape[0], image.shape[1]])
    row_idx, col_idx = np.where(segments == 0)
    for i, j in zip(row_idx, col_idx):
        segments[i, j] = int((image.shape[1]/size[1]) * (i//size[0]) + j//size[1])
    return segments

def tensor_to_numpy(tensor):
    numpy_lists = []
    if len(tensor.shape) == 4:
        for ten in tensor:
            arr = (ten.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
            numpy_lists.append(arr)
    return numpy_lists

def numpy_to_tesnor(arr):
    return torch.from_numpy((arr.transpose(0, 3, 1, 2)/255.0).astype(np.float32))


def analyze_anchors(test_images, model, probs, device="cpu"):
    numpy_images = tensor_to_numpy(test_images)
    def predict_fn(img):
        ten = numpy_to_tesnor(img).to(device)
        with torch.no_grad():
            logits = model(ten)
            probs = nn.Softmax(dim=1)(logits).cpu().numpy()
        return probs

    image_shape = numpy_images[0].shape
    expl = AnchorImage(predict_fn, image_shape, segmentation_fn=superpixel)
    anchors = []
    for arr in numpy_images:
        explanation = expl.explain(arr, threshold=.95, p_sample=.8, seed=0)
        anchors.append(explanation.anchor)
    return anchors, {"cmap":"gray"}
    

def analyze_shap(test_images, model, probs, bg_images=None):
    assert bg_images is not None
    class_idxs = [np.argmax(prob) for prob in probs]
    e = shap.DeepExplainer(model, bg_images)
    shap_values = e.shap_values(test_images)
    max_val = np.percentile(np.abs(np.concatenate(shap_values)), 99.9)
    plot_ops = {
        "cmap":"bwr",
        "vmin":-max_val, 
        "vmax":max_val
    }
    return [shap_values[i][j].transpose((1, 2, 0)) for j, i in enumerate(class_idxs)], plot_ops

def analzye_gradcam(test_images, model, probs, feature_mod=None):
    assert feature_mod is not None
    class_idxs = [np.argmax(prob) for prob in probs]
    gc = GradCAM(model, feature_mod)
    gc_maps = []
    for img, class_idx in zip(test_images, class_idxs):
        gc_map, _ = gc(img[None, :], class_idx=class_idx)
        gc_maps.append(gc_map[0].cpu().numpy().transpose((1, 2 ,0)))
    return gc_maps, {"cmap":"gray"}

