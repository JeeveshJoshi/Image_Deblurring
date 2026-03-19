import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,ConcatDataset
from torchvision import transforms
from models.generator import Generator
from models.discriminator import Discriminator
from losses.charbonnier_loss import CharbonnierLoss
from losses.vit_perceptual_loss import ViTPerceptualLoss
from torch.utils.data import random_split
from PIL import Image
import os
from tqdm import tqdm

def train_vit_gan(generator, discriminator, dataloader, epochs, device):
    
    optimizer_Generator = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizer_Discriminator = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))

    criterion_GAN = nn.BCEWithLogitsLoss() 
    criterion_charbonnier = CharbonnierLoss()
    criterion_perceptual = ViTPerceptualLoss().to(device) 

    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()

    print("Starting training...")
    for epoch in range(epochs):
        for i, (blurred_imgs, sharp_imgs) in enumerate(dataloader):
            blurred_imgs = blurred_imgs.to(device)
            sharp_imgs = sharp_imgs.to(device)

            optimizer_Discriminator.zero_grad()

            real_output = discriminator(sharp_imgs)
            discriminator_loss_real = criterion_GAN(real_output, torch.ones_like(real_output))

            generated_deblurred_imgs, _ = generator(blurred_imgs)
            fake_output = discriminator(generated_deblurred_imgs.detach())
            discriminator_loss_fake = criterion_GAN(fake_output, torch.zeros_like(fake_output))

            discriminator_loss = (discriminator_loss_real + discriminator_loss_fake) / 2
            discriminator_loss.backward()
            optimizer_Discriminator.step()

            optimizer_Generator.zero_grad()

            generated_deblurred_imgs, generated_sharp_imgs = generator(blurred_imgs)

            g_output_for_gan = discriminator(generated_deblurred_imgs)
            g_loss_gan = criterion_GAN(g_output_for_gan, torch.ones_like(g_output_for_gan))

            g_loss_charbonnier = criterion_charbonnier(generated_sharp_imgs, sharp_imgs)

            g_loss_perceptual = criterion_perceptual(generated_deblurred_imgs, sharp_imgs)

            lambda_gan = 1.0
            lambda_charbonnier = 1.0
            lambda_perceptual = 0.1 

            g_loss = (lambda_gan * g_loss_gan +
                      lambda_charbonnier * g_loss_charbonnier +
                      lambda_perceptual * g_loss_perceptual)
            g_loss.backward()
            optimizer_Generator.step()

            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(dataloader)}] "
                      f"D Loss: {discriminator_loss.item():.4f} G Loss: {g_loss.item():.4f} "
                      f"G_GAN: {g_loss_gan.item():.4f} G_Charb: {g_loss_charbonnier.item():.4f} "
                      f"G_Perceptual: {g_loss_perceptual.item():.4f}")

    print("Training complete.")

class DeblurDataset(torch.utils.data.Dataset):
    def __init__(self, blurry_dir, sharp_dir, transform=None):
        super().__init__()
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.transform = transform
        self.filenames = sorted(os.listdir(blurry_dir))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blurry_dir, self.filenames[idx])
        sharp_path = os.path.join(self.sharp_dir, self.filenames[idx])

        blurry_img = Image.open(blur_path).convert("RGB")
        sharp_img = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blurry_img = self.transform(blurry_img)
            sharp_img = self.transform(sharp_img)
        return blurry_img, sharp_img

def main():

    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    blur_sharp_dir_pairs = [
        ("D:/Image_Deblurring/Dataset/ProcessedDataset/BlurredData","D:/Image_Deblurring/Dataset/ProcessedDataset//CleanData")
    ]

    BATCH_SIZE = 20
    EPOCHS = 25
    LEARNING_RATE = 2e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    datasets = [DeblurDataset(blur, sharp, transform=transform) for blur, sharp in blur_sharp_dir_pairs]
    combined_dataset = ConcatDataset(datasets)

    train_size = int(0.9 * len(combined_dataset))
    test_size  = len(combined_dataset) - train_size
    gen = torch.Generator().manual_seed(42) 
    train_dataset, test_dataset = random_split(combined_dataset, [train_size, test_size], generator=gen)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Model initialization
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # Loss Functions
    pixel_loss_fn = CharbonnierLoss().to(device)
    perceptual_loss_fn = ViTPerceptualLoss().to(device)
    adversarial_loss_fn = nn.BCEWithLogitsLoss().to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)

    os.makedirs('checkpoints', exist_ok=True)
    # Training loop
    for epoch in range(EPOCHS):
        generator.train()
        discriminator.train()

        progressBar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for blurry_images, sharp_images in progressBar:
            blurry_images = blurry_images.to(device)
            sharp_images = sharp_images.to(device)
            
            opt_g.zero_grad()
            fake_images, _ = generator(blurry_images)

            
            d_out_fake = discriminator(fake_images)
            real_label_fake = torch.ones_like(d_out_fake)

            pixel_loss = pixel_loss_fn(fake_images, sharp_images)
            perceptual_loss = perceptual_loss_fn(fake_images, sharp_images)
            adversarial_loss = adversarial_loss_fn(d_out_fake, real_label_fake)

            total_g_loss = pixel_loss + perceptual_loss + 0.01 * adversarial_loss
            total_g_loss.backward()
            opt_g.step()

            opt_d.zero_grad()
            d_out_real = discriminator(sharp_images)
            d_out_fake = discriminator(fake_images.detach())
            real_label = torch.ones_like(d_out_real)
            fake_label = torch.zeros_like(d_out_fake)

            real_loss = adversarial_loss_fn(d_out_real, real_label)
            fake_loss = adversarial_loss_fn(d_out_fake, fake_label)

            total_d_loss = (real_loss + fake_loss) / 2
            total_d_loss.backward()
            opt_d.step()

            progressBar.set_postfix(G_loss=total_g_loss.item(), D_loss=total_d_loss.item())
            
        if ((epoch + 1) % 10 == 0) or (epoch + 1 == EPOCHS):
            checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': opt_g.state_dict(),
                'optimizer_d_state_dict': opt_d.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

if __name__ == "__main__":
    main()
