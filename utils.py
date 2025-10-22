

import torch
import numpy as np
import cv2
import os
import wandb
import flip_evaluator
import matplotlib
from skimage.metrics import structural_similarity


import torch.nn as nn
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import wandb
import os, re, glob
import imageio.v3 as iio

def load_npy(train_data_dir, row, col):
    fpath = os.path.join(train_data_dir, f"{row}_{col}.npy")
    return np.load(fpath).reshape(-1, 3)  
gamma = lambda img: np.power(np.maximum(0, img) / (img.max() + 1e-10), 1 / 2.2)



def generate_reflectance_image(model, dataset, device="cuda", figure_label="reflectance", epoch=0):
    model.eval()
    H, W = 32, 64

    with torch.no_grad():
        reflectance_GT = dataset.reflectance_GT.to(device)  # (P, 2048, 3)
        preds, _ = model(reflectance_GT)
        preds = preds.cpu().numpy()
        gts = reflectance_GT.cpu().numpy()
        mask = dataset.mask.view(-1, 1, 1).cpu().numpy()

    # --- Apply mask ---
    preds *= mask
    gts *= mask

    # --- Pick 25 sample pixels (5×5 grid) ---
    num_samples = min(25, len(dataset))
    idxs = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)

    pred_list, gt_list = [], []
    for i in idxs:
        pred_list.append(preds[i].reshape(H, W, 3))
        gt_list.append(gts[i].reshape(H, W, 3))

    # --- Tile into 5×5 grid ---
    def tile_grid(imgs):
        imgs = np.array(imgs).reshape(5, 5, H, W, 3)
        return imgs.transpose(0, 2, 1, 3, 4).reshape(5 * H, 5 * W, 3)

    pred_img, gt_img = map(tile_grid, [pred_list, gt_list])
    pred_img, gt_img = pred_img/pred_img.max(), gt_img/gt_img.max()
    
    # --- Compute metrics (after gamma) ---
    mse_val = float(((pred_img - gt_img) ** 2).mean())
    ssim_val, ssim_map = structural_similarity(pred_img, gt_img, channel_axis=-1, data_range=1.0, full=True)
    ssim_err = 1.0 - ssim_map.mean(axis=2)
    flip_err, _, _ = flip_evaluator.evaluate(gt_img, pred_img, "LDR")

    # --- Visualization ---
    ssim_vis = matplotlib.cm.magma(ssim_err)[..., :3]
    gap = np.ones((pred_img.shape[0], 10, 3), dtype=pred_img.dtype)
    
    pred_img, gt_img = map(gamma, [pred_img, gt_img])
    stacked = np.hstack([pred_img, gap, gt_img, gap, ssim_vis, gap, flip_err])

    label = f"Pred | GT | SSIM | FLIP --- MSE={mse_val:.2e}, SSIM={ssim_val:.3f}"
    h, w = stacked.shape[:2]
    labeled = np.vstack([(stacked * 255).astype(np.uint8), np.zeros((20, w, 3), np.uint8)])
    cv2.putText(labeled, label,
                ((w - cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]) // 2, h + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    wandb.log({
        "epoch": epoch,
        figure_label: wandb.Image(labeled, caption=label),
        "MSE": mse_val,
        "SSIM": ssim_val
    })

    return labeled



# --- Helper functions ---
pat = re.compile(r'(\d+)[_-](\d+)\.exr$')  # "row_col.exr"

def exr_sort_key(path):
    m = pat.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9)

def _read_exr(cfg, path):
    """Helper to load + crop EXR image."""
    r0, r1 = cfg["row_slice"]
    c0, c1 = cfg["col_slice"]
    img = iio.imread(path).astype(np.float32)
    img = img[r0:r1, c0:c1, :]
    # return torch.from_numpy(img).permute(2, 0, 1)  # (3,H,W)
    return img

def load_OLAT(cfg, file_paths, num_threads=8):
    """Multithreaded EXR loading."""
    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        imgs = list(ex.map(lambda p: _read_exr(cfg, p), file_paths))
    
    return np.stack(imgs, axis=0)  # (N,H,W,3)



# ==================================================
# Flexible MLP builder
# ==================================================
def build_mlp(layer_dims, activation_name="relu", final_activation=None):
    """Generic MLP builder for flexible experiments."""
    act_map = {
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "sigmoid": nn.Sigmoid,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "none": nn.Identity
    }
    activation = act_map.get(activation_name.lower(), nn.ReLU)
    layers = []
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        if i < len(layer_dims) - 2:
            layers.append(activation())
    if final_activation:
        layers.append(act_map[final_activation.lower()]())
    return nn.Sequential(*layers)
