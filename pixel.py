import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
import wandb
import utils
import os, re, glob
import imageio.v3 as iio
device ="cuda" if torch.cuda.is_available() else "cpu"



# --- Reflectance Dataset ---
class ReflectanceDataset(Dataset):
    def __init__(self, config, num_threads=8):
        self.cfg = config
        self.H = config["row_slice"][1] - config["row_slice"][0]
        self.W = config["col_slice"][1] - config["col_slice"][0]

        # --- Mask ---
        mask = cv2.imread(config["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)
        self.mask = torch.tensor(mask[config["row_slice"][0]:config["row_slice"][1],config["col_slice"][0]:config["col_slice"][1]])

        # --- Load OLAT images dynamically ---
        data_dir = config["olat_dir"]
        file_paths = sorted(glob.glob(os.path.join(data_dir, "*.exr")), key=utils.exr_sort_key)
        if not file_paths:
            raise FileNotFoundError(f"No EXR files found in {data_dir}")

        print(f"🧩 Loading {len(file_paths)} OLAT images from {data_dir}")
        olat_stack = utils.load_OLAT(config, file_paths, num_threads=num_threads)  # (2048,3,H,W) # (2048,H,W, 3)
        reflectance = olat_stack.transpose(1, 2, 0, 3).reshape(-1, 2048, 3)

        # --- Apply mask and normalize ---
        self.reflectance_GT = torch.from_numpy(reflectance).float() * self.mask.view(-1,1,1)
        self.reflectance_GT = self.reflectance_GT.clip(0, np.percentile(self.reflectance_GT, 99))
        self.reflectance_GT /= self.reflectance_GT.max()
        
        self.mask = self.mask.view(-1)
        

        print(f"✅ Reflectance map computed: {self.reflectance_GT.shape}")

    def __len__(self):
        return self.H * self.W

    def __getitem__(self, idx):
        return self.reflectance_GT[idx], self.mask[idx]


# ==================================================
# Reflectance Autoencoder
# ==================================================
class ReflectanceAutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_dim = config["input_dim"] * 3
        latent_dim = config["latent_dim"]
        hidden_dims = config["hidden_dims"]

        encoder_dims = [in_dim] + hidden_dims + [latent_dim]
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [in_dim]

        self.encoder = utils.build_mlp(
            encoder_dims,
            activation_name=config.get("activation", "relu"),
            final_activation=config.get("final_encoder_activation", None)
        )
        self.decoder =utils.build_mlp(
            decoder_dims,
            activation_name=config.get("activation", "relu"),
            final_activation=config.get("final_decoder_activation", "relu")
        )

        self.input_dim = config["input_dim"]

    def forward(self, x):
        B = x.shape[0]
        # x = x.view(B, -1)
        x = x.reshape(B, -1)
        z = self.encoder(x)
        out = self.decoder(z)
        return out.view(B, self.input_dim, 3), z



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
                utils.generate_reflectance_image(model, dataset)



# ==================================================
# Config
# ==================================================
config = {
    "exp_name": "activation",
    "row_slice": [100, 200],
    "col_slice": [100, 200],
    "mask_path": "/home/gmh72/3DReconstruction/Blender_Rendering/data/diffuse_suzanne_white/mask.png",
    "reflectance_data_dir": "/home/gmh72/3DReconstruction/Blender_Rendering/data/diffuse_suzanne_white/reflectance",
    "olat_dir":"/home/gmh72/3DReconstruction/Blender_Rendering/data/diffuse_suzanne_white/olat",
    "input_dim": 2048,
    "hidden_dims": [1024, 256],
    "latent_dim": 128,
    "activation": "leakyrelu",  # relu, leakyrelu, sigmoid, tanh, elu, gelu
    "final_encoder_activation": None,
    "final_decoder_activation": "sigmoid",
    "lr": 1e-4,
    "epochs": 1000,
    "batch_size": 100,
    "log_interval": 10,
    "use_wandb": False,
    "wandb_project": "PixelOLAT",
    "save_path": "/home/gmh72/3DReconstruction/2DRelighting/output/auto/reflectance_autoencoder_flex.pth"
}


# -------------------------------
# Main Function
# -------------------------------
def main(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["use_wandb"]:
        exp_name = f'{config["exp_name"]}_middle={config["activation"]}_final={config["final_decoder_activation"]}'
        wandb.init(
            project=config["wandb_project"],
            name=exp_name,
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
    # torch.save(model.state_dict(), config["save_path"])
    # print(f"✅ Model saved to {config['save_path']}")


# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    main(config)
