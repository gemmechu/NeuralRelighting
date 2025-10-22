import os, glob, re
import torch
import numpy as np
import imageio.v3 as iio
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import wandb
import cv2
import matplotlib
from skimage.metrics import structural_similarity

# ==================================================
# Utility functions
# ==================================================
gamma = lambda img: np.power(np.maximum(0, img) / (img.max() + 1e-10), 1 / 2.2)
pat = re.compile(r'(\d+)[_-](\d+)\.exr$')  # matches "row_col.exr" or "row-col.exr"

def exr_sort_key(path):
    m = pat.search(os.path.basename(path))
    return (int(m.group(1)), int(m.group(2))) if m else (10**9, 10**9)


def _read_exr(cfg, path):
    """Helper function for threaded EXR loading with slicing."""
    r0, r1 = cfg["row_slice"]
    c0, c1 = cfg["col_slice"]

    img = iio.imread(path).astype(np.float32)  # (H, W, 3)
    img = img[r0:r1, c0:c1, :]                 # crop region
    return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)


def load_OLAT(cfg, file_paths, num_threads=8):
    """Load and stack OLAT images using multithreading."""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        imgs = list(executor.map(lambda p: _read_exr(cfg, p), file_paths))
    return torch.cat(imgs, dim=0)  # (3*N, H, W)


def log_reflectance_to_wandb(pred_img, gt_img, epoch, label="reflectance_map", flip_evaluator=None):
    """
    pred_img, gt_img: (H, W, 6144)
    Select 25 (5x5) pixels, reshape to (32,64,3), tile, and log GT|Pred|SSIM|FLIP.
    """
    H, W, C = pred_img.shape
    assert C == 6144, f"Expected 6144 channels, got {C}"
    Hmap, Wmap = 32, 64

    # --- Pick 25 sample pixel indices (5x5 grid) ---
    row_idx = np.linspace(0, H - 1, 5, dtype=int)
    col_idx = np.linspace(0, W - 1, 5, dtype=int)
    idx_grid = [(r, c) for r in row_idx for c in col_idx]

    refl_pred, refl_gt = [], []
    for (r, c) in idx_grid:
        pred_map = pred_img[r, c].reshape(Hmap, Wmap, 3)
        gt_map   = gt_img[r, c].reshape(Hmap, Wmap, 3)
        refl_pred.append(pred_map)
        refl_gt.append(gt_map)

    # --- Tile into grids ---
    refl_pred = np.array(refl_pred).reshape(5, 5, Hmap, Wmap, 3)
    refl_gt   = np.array(refl_gt).reshape(5, 5, Hmap, Wmap, 3)
    img_pred = refl_pred.transpose(0, 2, 1, 3, 4).reshape(5 * Hmap, 5 * Wmap, 3)
    img_gt   = refl_gt.transpose(0, 2, 1, 3, 4).reshape(5 * Hmap, 5 * Wmap, 3)

    # img_pred, img_gt = map(lambda x: np.clip(x, 0, 1), [img_pred, img_gt])
    
    img_pred, img_gt = map(gamma, [img_pred, img_gt])

    # --- Metrics ---
    mse_val = float(((img_pred - img_gt) ** 2).mean())
    ssim_map = structural_similarity(img_pred, img_gt, channel_axis=-1, data_range=1.0, full=True)[1]
    ssim_err = (1.0 - ssim_map).mean(axis=2)
    ssim_val = 1.0 - float(ssim_err.mean())

    # --- Visualization ---
    gap = np.ones((img_pred.shape[0], 10, 3))
    ssim_vis = matplotlib.cm.magma(ssim_err)[..., :3]
    stacked = np.hstack([img_pred, gap, img_gt, gap, ssim_vis])

    label_text = f"Pred | GT | SSIM | FLIP --- MSE={mse_val:.2e}, SSIM={ssim_val:.3f}"
    h, w = stacked.shape[:2]
    labeled_img = np.vstack([(stacked * 255).astype(np.uint8), np.zeros((20, w, 3), np.uint8)])
    cv2.putText(
        labeled_img, label_text,
        ((w - cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]) // 2, h + 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
    )

    wandb.log({
        label: wandb.Image(labeled_img, caption=label_text),
        "mse": mse_val,
        "ssim": ssim_val,
        "epoch": epoch,
    })


# ==================================================
# Dataset
# ==================================================
class OLATDataset(Dataset):
    def __init__(self, cfg, num_threads=8):
        self.cfg = cfg
        self.data, self.names = [], []

        for folder in cfg["folders"]:
            data_dir = os.path.join(cfg["root_dir"], folder, "olat")
            file_paths = sorted(glob.glob(os.path.join(data_dir, "*.exr")), key=exr_sort_key)
            if not file_paths:
                print(f"⚠️ No EXRs found in {data_dir}")
                continue

            stacked = load_OLAT(cfg, file_paths, num_threads=num_threads)
            self.data.append(stacked)
            self.names.append(folder)

        print(f"✅ Loaded {len(self.data)} OLAT samples using {num_threads} threads")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return same tensor as input & target for autoencoder training
        return self.data[idx], self.data[idx]


# ==================================================
# Dataloader factory (integrates with existing config)
# ==================================================
def get_dataloader(cfg, num_threads=8):
    dataset = OLATDataset(cfg, num_threads=num_threads)
    bs = cfg.get("batch_size", 1)
    return DataLoader(dataset, batch_size=bs, shuffle=False), dataset


# --------------------------------------------------
# Double Conv
# --------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------
# Encoder
# --------------------------------------------------
class UNetEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["encoder_channels"]
        in_ch = cfg["in_channels"]
        latent_dim = cfg["latent_dim"]
        k = cfg["kernel_size"]
        p = cfg["padding"]
        self.spatial_latent = cfg["spatial_latent"]

        self.enc1 = DoubleConv(in_ch, c[0], k, p)
        self.enc2 = DoubleConv(c[0], c[1], k, p)
        self.enc3 = DoubleConv(c[1], c[2], k, p)
        self.enc4 = DoubleConv(c[2], c[3], k, p)

        if not self.spatial_latent:
            self.fc_mu = nn.Linear(c[3], latent_dim)

    def forward(self, x):
        B, _, H, W = x.shape
        x = self.enc1(x)
        x = F.max_pool2d(x, 2)
        x = self.enc2(x)
        x = F.max_pool2d(x, 2)
        x = self.enc3(x)
        x = F.max_pool2d(x, 2)
        x = self.enc4(x)
        x = F.max_pool2d(x, 2)

        _, C, H_out, W_out = x.shape
        if self.spatial_latent:
            return x, (H_out, W_out)
        else:
            x = F.adaptive_avg_pool2d(x, 1).view(B, -1)
            z = self.fc_mu(x)
            return z, (H_out, W_out)


# --------------------------------------------------
# Decoder
# --------------------------------------------------
class UNetDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg["decoder_channels"]
        out_ch = cfg["in_channels"]
        latent_dim = cfg["latent_dim"]
        self.spatial_latent = cfg["spatial_latent"]

        self.fc = None  # initialized dynamically in forward

        self.up4 = nn.ConvTranspose2d(c[0], c[1], 2, stride=2)
        self.dec4 = DoubleConv(c[1], c[1])
        self.up3 = nn.ConvTranspose2d(c[1], c[2], 2, stride=2)
        self.dec3 = DoubleConv(c[2], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[3], 2, stride=2)
        self.dec2 = DoubleConv(c[3], c[3])
        self.up1 = nn.ConvTranspose2d(c[3], out_ch, 2, stride=2)

    def forward(self, z, latent_hw, first_ch):
        if not self.spatial_latent:
            # create fc layer on first forward to match shape
            if self.fc is None:
                self.fc = nn.Linear(z.size(1), first_ch * latent_hw[0] * latent_hw[1]).to(z.device)
            B = z.size(0)
            x = self.fc(z).view(B, first_ch, *latent_hw)
        else:
            x = z

        x = self.dec4(self.up4(x))
        x = self.dec3(self.up3(x))
        x = self.dec2(self.up2(x))
        out = self.up1(x)
        return out


# --------------------------------------------------
# Combined
# --------------------------------------------------
class ReflectanceAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = UNetEncoder(cfg)
        self.decoder = UNetDecoder(cfg)

    def forward(self, x):
        z, latent_hw = self.encoder(x)
        first_ch = self.cfg["decoder_channels"][0]
        out = self.decoder(z, latent_hw, first_ch)
        return out, z

# ==================================================
# Training with optional W&B logging
# ==================================================
def train_autoencoder(model, dataloader, cfg):
    tcfg = cfg["training"]
    device = tcfg["device"]
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=tcfg["lr"])
    loss_fn = nn.MSELoss()

    use_wandb = cfg.get("use_wandb", False) and wandb is not None
    if use_wandb:
        wandb.init(project=cfg["project"], config=cfg)
        wandb.watch(model, log="gradients", log_freq=10)

    for epoch in range(tcfg["epochs"]):
        total_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            recon, _ = model(x)
            loss = loss_fn(recon, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{tcfg['epochs']} | Avg Loss: {avg_loss:.6f}")
        
        # Visual log every few epochs
        if (epoch + 1) % 4 == 0 and use_wandb:
        # if (epoch + 1) % 4 == 0:
            wandb.log({"loss": avg_loss, "epoch": epoch + 1})
            x_vis, y_vis = next(iter(dataloader))
            x_vis, y_vis = x_vis.to(device), y_vis.to(device)
            with torch.no_grad():
                pred, _ = model(x_vis)
                pred_img = pred[0].permute(1, 2, 0).cpu().numpy()
                gt_img   = y_vis[0].permute(1, 2, 0).cpu().numpy()
                log_reflectance_to_wandb(pred_img, gt_img, epoch, flip_evaluator=False)

    if use_wandb:
        wandb.finish()



# --------------------------------------------------
# CONFIG
# --------------------------------------------------

config = {
    "model": {
        "in_channels": 96,
        "latent_dim": 128,
        "encoder_channels": [64, 128, 256, 512],
        "decoder_channels": [512, 256, 128, 64],
        "kernel_size": 3,
        "padding": 1,
        "spatial_latent": True,  # 🔁 toggle between spatial or global latent
    },
    "training": {
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    },
    "data": {
        "N": 4,
        "C": 96,
        "H": 112,
        "W": 112,
        "root_dir": "/home/gmh72/3DReconstruction/Blender_Rendering/data",
        "folders": ["diffuse_suzanne_white"],
        "row_slice": [135, 135+112],
        "col_slice": [135, 135+112],
    },
    "use_wandb": True,
    "project":"ReflectanceAE"  
}

if __name__ == "__main__":
   
    dataloader, dataset = get_dataloader(config["data"], num_threads=8)

    # Detect input channels automatically
    in_channels = dataset[0][0].shape[0]
    print(f"Detected input channels: {in_channels}")

    # Merge into model config
    config["model"]["in_channels"] = in_channels

    model = ReflectanceAutoEncoder(config["model"])
    train_autoencoder(model, dataloader, config)
