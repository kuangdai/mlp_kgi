import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm


# Argument parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train UNet for Denoising on CelebA")
    parser.add_argument("--max_data", type=int, default=None, help="Maximum number of images")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--test_every", type=int, default=10, help="Test every n epochs")
    parser.add_argument("--noise_level", type=float, default=0.1,
                        help="Standard deviation of Gaussian noise added to images")
    parser.add_argument("--KGI", action="store_true", help="Activate KGI")
    return parser.parse_args()


# Add noise to images
def add_gaussian_noise(img, amp=0.1):
    noise = torch.randn_like(img) * amp
    return img + noise


# Plot noisy and clean images side by side for visual inspection
def plot_noisy_vs_clean(clean_dataset, amp=0.1, num_samples=5):
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 15))
    for i in range(num_samples):
        # Select a random index
        idx = random.randint(0, len(clean_dataset) - 1)

        # Retrieve clean image and its label (ignore label if not used)
        clean_img = clean_dataset[idx]

        # Add noise to the clean image
        noisy_img = add_gaussian_noise(clean_img, amp)

        # Convert from tensor to numpy and denormalize
        clean_img = clean_img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # Un-normalize
        noisy_img = noisy_img.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5  # Un-normalize

        # Clip values to [0, 1] for proper visualization
        noisy_img = np.clip(noisy_img, 0, 1)
        clean_img = np.clip(clean_img, 0, 1)

        # Plot noisy image
        axes[i, 0].imshow(noisy_img)
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Noisy Image")

        # Plot clean image
        axes[i, 1].imshow(clean_img)
        axes[i, 1].axis('off')
        axes[i, 1].set_title("Clean Image")

    plt.tight_layout()
    plt.show()


class CelebADataset(Dataset):
    """
    Custom dataset to handle flat directory structure for CelebA images.
    Allows limiting the total number of data points.
    """

    def __init__(self, root_dir, transform=None, max_data=None):
        """
        Args:
            root_dir (str): Directory containing image files.
            transform (callable, optional): Transform to apply to each image.
            max_data (int, optional): Maximum number of images to include in the dataset. If None, use all images.
        """
        self.root_dir = root_dir
        self.image_paths = sorted(
            [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(".jpg")]
        )

        if max_data is not None:
            self.image_paths = self.image_paths[:max_data]  # Limit dataset size

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Training function
def train_epoch(model, noise_level, dataloader, criterion, optimizer, device, epoch, epochs):
    model.train()
    running_loss = 0.0
    for clean in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
        clean = clean.to(device)
        noise = add_gaussian_noise(clean, noise_level)
        optimizer.zero_grad()
        outputs = model(noise)
        loss = criterion(outputs, clean)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * clean.size(0)
    return running_loss / len(dataloader.dataset)


# Testing function
def test_epoch(model, noise_level, dataloader, criterion, device, epoch, epochs):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for clean in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", unit="batch"):
            clean = clean.to(device)
            noise = add_gaussian_noise(clean, noise_level)
            outputs = model(noise)
            loss = criterion(outputs, clean)
            running_loss += loss.item() * clean.size(0)
    return running_loss / len(dataloader.dataset)


# Main function
def main():
    args = parse_args()
    set_seed(args.seed)

    # Paths for saving results
    os.makedirs("./results/unet_paper", exist_ok=True)
    train_results_path = f"./results/unet_paper/train_{args.seed}_{args.KGI}.txt"
    test_results_path = f"./results/unet_paper/test_{args.seed}_{args.KGI}.txt"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and dataloaders
    transforms_clean = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Path to the extracted CelebA dataset
    img_dir = "./datasets_torch/img_align_celeba"
    # Load clean and noisy datasets
    all_dataset = CelebADataset(root_dir=img_dir, transform=transforms_clean, max_data=args.max_data)

    # Split into train and test sets
    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=3, init_features=32, pretrained=False)
    model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    train_mse_history = []
    test_mse_history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, args.noise_level, train_loader, criterion,
                                 optimizer, device, epoch, args.epochs)
        train_mse_history.append(train_loss)
        print(f"Epoch {epoch}/{args.epochs}, Train MSE: {train_loss:.4f}")

        if epoch % args.test_every == 0:
            test_loss = test_epoch(model, args.noise_level, test_loader, criterion, device, epoch, args.epochs)
            test_mse_history.append(test_loss)
            print(f"Epoch {epoch}/{args.epochs}, Test MSE: {test_loss:.4f}")

    # Save results
    np.savetxt(train_results_path, train_mse_history)
    np.savetxt(test_results_path, test_mse_history)
    print(f"Results saved to {train_results_path} and {test_results_path}")


if __name__ == "__main__":
    main()
