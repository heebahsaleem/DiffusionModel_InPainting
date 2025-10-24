import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

device = "cuda:7" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

#unet
class HandwritingUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._block(2, 128)
        self.enc2 = self._block(128, 256)
        self.enc3 = self._block(256, 512)
        self.enc4 = self._block(512, 1024)
        self.bottleneck = self._block(1024, 1024)
        self.time_embed = nn.Sequential(
            nn.Linear(1, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
        )
        self.dec1 = self._block(1024 + 1024, 512)
        self.dec2 = self._block(512 + 512, 256)
        self.dec3 = self._block(256 + 256, 128)
        self.dec4 = self._block(128 + 128, 64)
        self.final = nn.Conv2d(64, 2, kernel_size=1)
        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, t):
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        t_embed = t_embed.unsqueeze(-1).unsqueeze(-1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.downsample(e1))
        e3 = self.enc3(self.downsample(e2))
        e4 = self.enc4(self.downsample(e3))
        bottleneck = self.bottleneck(self.downsample(e4))
        bottleneck = bottleneck + t_embed
        d1 = self.dec1(torch.cat([self.upsample(bottleneck), e4], dim=1))
        d2 = self.dec2(torch.cat([self.upsample(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.upsample(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.upsample(d3), e1], dim=1))
        return self.final(d4)

#dataset loader
class IAMDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.image_paths = []
        for subdir, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))
        print(f"Found {len(self.image_paths)} images")
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((64, 64)),  # Match training
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = self.transform(image)
        image = image.repeat(2, 1, 1)  # [1,64,256] -> [2,64,256]
        return image, 0

#diffusin params
T = 1000
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T, device=device)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
model = HandwritingUNet().to(device)
model_path = "/Home/siv36/hesal5042/Research/NORCE/hello/RePaint/guided_diffusion_mnist/guided_diffusion/handwriting_output/model_epoch_100.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
# print(f"loaded model from {model_path}")
model.eval()
dataset = IAMDataset(root_dir='iamHandwriting_dataset/words')
B = 4
indices = random.sample(range(len(dataset)), min(B, len(dataset)))
images = torch.stack([dataset[i][0] for i in indices]).to(device)


mask = torch.ones_like(images)
mask[:, :, :, 32:] = 0
mask = TF.gaussian_blur(mask, kernel_size=9, sigma=3)
mask = mask.clamp(0, 1)
# known_regions = images * mask
image_mean = images.mean(dim=[2, 3], keepdim=True)
noisy_fill = image_mean + torch.randn_like(images) * 0.1
known_regions = images * mask + noisy_fill * (1 - mask)


#repaint
def repaint_handwriting(model, x, mask, steps=500, jump_length=5, jump_n_sample=15):
    B = x.size(0)
    x_t = torch.randn_like(x)
    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t] * B, device=device, dtype=torch.long)
        with torch.no_grad():
            noise_pred = model(x_t, t_tensor)
        alpha_t = alphas_cumprod[t]
        beta_t = 1 - alpha_t
        noise = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)
        x_t_prev = (1 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1 - alpha_t)) * noise_pred) + torch.sqrt(beta_t) * noise
        x_t_prev = mask * x + (1 - mask) * x_t_prev
        if t % jump_length == 0 and t > 0:
            for _ in range(jump_n_sample):
                noise_jump = torch.randn_like(x_t_prev)
                x_t_prev = torch.sqrt(alpha_t) * x_t_prev + torch.sqrt(1 - alpha_t) * noise_jump
                x_t_prev = mask * x + (1 - mask) * x_t_prev
                x_t = 0.7 * x_t_prev + 0.3 * x_t  
    output = (x_t + 1) / 2
    return torch.clamp(output, 0, 1)

# print("RePaint from here")
output = repaint_handwriting(model, known_regions, mask)

def main():





    os.makedirs('handwriting_repaint_64x64', exist_ok=True)
    for i in range(B):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  
        images_to_plot = [
            (images[i, 0].cpu(), "Original"),
            (known_regions[i, 0].cpu(), "Masked"),
            (output[i, 0].cpu(), "RePainted")
        ]
        for ax, (img, title) in zip(axes, images_to_plot):
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_title(title)
            # ax.axis('off')
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(2)
        plt.tight_layout()
        plt.savefig(f'handwriting_repaint_64x64/result_{i}.png', dpi=150, bbox_inches='tight')
        plt.close()
    print("ePaint completed and saved in 'handwriting_repaint_64x64/'")


if __name__ == "__main__":
    main()
