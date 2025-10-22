import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import os
import wandb
import flip_evaluator
import matplotlib
from skimage.metrics import structural_similarity


device ="cuda" if torch.cuda.is_available() else "cpu"


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





class ReflectanceDataset(Dataset):
    def __init__(self, config):
        self.H = config["row_slice"][1] - config["row_slice"][0]
        self.W = config["col_slice"][1] - config["col_slice"][0]
        mask = cv2.imread(config["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # convert 0–255 → 0/1
        self.mask = torch.tensor(mask[config["row_slice"][0]:config["row_slice"][1],
                        config["col_slice"][0]:config["col_slice"][1]])

        # --- GT Reflectance ---
        with ThreadPoolExecutor() as executor:
            reflectance_GT_files = np.array(list(executor.map(lambda rc: load_npy(config["reflectance_data_dir"],*rc), [(row, col) for row in range(*config["row_slice"]) for col in range(*config["col_slice"])])))
        
        self.reflectance_GT = torch.from_numpy(reflectance_GT_files).float() * self.mask.view(-1,1,1)
        self.reflectance_GT = self.reflectance_GT/self.reflectance_GT.max()
        # import pdb; pdb.set_trace()
        self.mask = self.mask.view(-1)
        
        
    def __len__(self): 
        return self.H*self.W
    def __getitem__(self, idx):
        return self.reflectance_GT[idx], self.mask[idx]


# -------------------------------
# Simple Autoencoder
# -------------------------------
class ReflectanceAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config["input_dim"] * 3
        h1, h2 = config["hidden_dims"]
        latent_dim = config["latent_dim"]

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2),
            nn.ReLU(),
            nn.Linear(h2, h1),
            nn.ReLU(),
            nn.Linear(h1, in_dim),
        )

    def forward(self, x):
        # x: (B, input_dim, 3)
        B = x.shape[0]
        x = x.view(B, -1)
        z = self.encoder(x)
        out = self.decoder(z)
        out = torch.relu(out)
        return out.view(B, config["input_dim"], 3), z


# -------------------------------
# Loss Function
# -------------------------------
def masked_mse_loss(pred, gt, mask, eps=1e-8):
    while mask.dim() < pred.dim() - 1:
        mask = mask.unsqueeze(-1)
    mask = mask.unsqueeze(-1)
    diff = (pred - gt) ** 2
    masked_diff = diff * mask
    return masked_diff.sum() / (mask.sum() * pred.shape[-1] + eps)


# -------------------------------
# Training Functions
# -------------------------------
def train_batch(model, optimizer, loss_fn, batch, device):
    ref_gt, mask = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad()
    recon, _ = model(ref_gt)
    loss = loss_fn(recon, ref_gt, mask)
    loss.backward()
    optimizer.step()
    return loss.item()


def train(model, dataloader, dataset, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = masked_mse_loss
    epochs = config["epochs"]
    log_interval = config["log_interval"]

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            loss = train_batch(model, optimizer, loss_fn, batch, device)
            total_loss += loss
        avg_loss = total_loss / len(dataloader)

        if epoch % log_interval == 0:
            print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.6f}")
            if config["use_wandb"]:
                wandb.log({"epoch": epoch, "loss": avg_loss})
                generate_reflectance_image(model, dataset)


# -------------------------------
# Config
# -------------------------------
config = {
    "exp_name": "suzanne_reflectance_128d",
    "row_slice": [100, 200],
    "col_slice": [100, 200],
    "mask_path": "/home/gmh72/3DReconstruction/Blender_Rendering/data/diffuse_suzanne_white/mask.png",
    "reflectance_data_dir": "/home/gmh72/3DReconstruction/Blender_Rendering/data/diffuse_suzanne_white/reflectance",
    "input_dim": 2048,
    "hidden_dims": [1024, 256],
    "latent_dim": 128,
    "lr": 1e-4,
    "valid_reg": 1.0,
    "epochs": 1000,
    "batch_size": 100,
    "log_interval": 10,
    "use_wandb": True,
    "wandb_project": "PixelMLP",
    "save_path": "/home/gmh72/3DReconstruction/2DRelighting/output/auto/reflectance_autoencoder.pth"
}


# -------------------------------
# Main Function
# -------------------------------
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["use_wandb"]:
        wandb.init(
            project=config["wandb_project"],
            name=config["exp_name"],
            config=config,
            save_code=True
        )

    dataset = ReflectanceDataset(config)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    print("✅ Data loaded!")
    print("Learning rate:", config["lr"])

    model = ReflectanceAutoEncoder(config).to(device)
    train(model, dataloader, dataset, config, device)

    # Save final model
    torch.save(model.state_dict(), config["save_path"])
    print(f"✅ Model saved to {config['save_path']}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main(config)
