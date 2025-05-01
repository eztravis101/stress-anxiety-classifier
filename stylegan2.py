from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

import pytorch_lightning as pl
from torchmetrics.image.fid import FrechetInceptionDistance

# ─────────────────────────────── Globals
CLASS_MAPPING: Dict[int, str] = {0: "relaxed", 1: "mild_stress", 2: "anxious", 3: "ptsd"}
NUM_CLASSES = len(CLASS_MAPPING)
IMAGE_SIZE  = 75
LATENT_DIM  = 100

# ─────────────────────────────── Transforms
def build_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

# ─────────────────────────────── Generator
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM + NUM_CLASSES, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, 3 * IMAGE_SIZE * IMAGE_SIZE),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        img = self.model(x)
        return img.view(img.size(0), 3, IMAGE_SIZE, IMAGE_SIZE)

# ─────────────────────────────── Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, NUM_CLASSES)
        self.model = nn.Sequential(
            nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE + NUM_CLASSES, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], dim=1)
        return self.model(x)

# ─────────────────────────────── GAN LightningModule
class StressGAN(pl.LightningModule):
    def __init__(self, lr: float = 2e-4, b1: float = 0.5, b2: float = 0.999):
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator()
        self.discriminator = Discriminator()

        self.adversarial_loss = nn.BCELoss()
        self.automatic_optimization = False

        self.validation_z = torch.randn(8, LATENT_DIM)
        self.validation_labels = torch.arange(0, 8) % NUM_CLASSES

        self.fid = FrechetInceptionDistance(feature=2048)

    def forward(self, z, labels):
        return self.generator(z, labels)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(self.hparams.b1, self.hparams.b2))
        return [opt_g, opt_d]

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        batch_size = imgs.size(0)
        valid = torch.ones(batch_size, 1, device=self.device)
        fake = torch.zeros(batch_size, 1, device=self.device)

        noise = torch.randn(batch_size, LATENT_DIM, device=self.device)
        gen_imgs = self(z=noise, labels=labels)

        opt_g, opt_d = self.optimizers()

        # Train Discriminator
        self.toggle_optimizer(opt_d)
        real_loss = self.adversarial_loss(self.discriminator(imgs, labels), valid)
        fake_loss = self.adversarial_loss(self.discriminator(gen_imgs.detach(), labels), fake)
        d_loss = (real_loss + fake_loss) / 2
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Train Generator
        self.toggle_optimizer(opt_g)
        gen_imgs = self(z=noise, labels=labels)
        g_loss = self.adversarial_loss(self.discriminator(gen_imgs, labels), valid)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log('g_loss', g_loss, prog_bar=True)
        self.log('d_loss', d_loss, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        if (self.current_epoch + 1) % 25 != 0:
            return
        imgs, labels = batch
        noise = torch.randn(imgs.size(0), LATENT_DIM, device=self.device)
        gen_imgs = self(z=noise, labels=labels)

        def denorm(x):
            x = (x * 0.5 + 0.5).clamp(0, 1)
            x = (x * 255).to(torch.uint8)
            return x

        self.fid.update(denorm(imgs), real=True)
        self.fid.update(denorm(gen_imgs), real=False)

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % 25 == 0:
            fid_score = self.fid.compute()
            self.log("FID", fid_score, prog_bar=True)
            self.fid.reset()

    def on_train_epoch_end(self):
        z = self.validation_z.to(self.device)
        labels = self.validation_labels.to(self.device)
        sample_imgs = self(z, labels)
        grid = utils.make_grid(sample_imgs, nrow=4, normalize=True)
        self.logger.experiment.add_image("generated_faces", grid, self.current_epoch)

        if (self.current_epoch + 1) % 25 == 0:
            Path("generated_samples").mkdir(parents=True, exist_ok=True)
            utils.save_image(grid, f"generated_samples/epoch_{self.current_epoch + 1}.png")

# ─────────────────────────────── DataModule
class StressDataModule(pl.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 64, num_workers: int = 4):
        super().__init__()
        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str | None = None):
        self.ds_train = ImageFolder(self.data_root / "train", transform=build_transforms())
        self.ds_val   = ImageFolder(self.data_root / "val",   transform=build_transforms())

    def train_dataloader(self):
        return DataLoader(self.ds_train, self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.ds_val, self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

# ─────────────────────────────── CLI Entrypoint
if __name__ == "__main__":
    from pytorch_lightning.cli import LightningCLI
    LightningCLI(StressGAN, StressDataModule, seed_everything_default=42, save_config_callback=None)
