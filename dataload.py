import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

data_path='data/dermamnist_64.npz'

class DermaMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        """
        Custom dataset for loading DermaMNIST data
        Args:
            images (numpy.ndarray): Image array
            labels (numpy.ndarray): Label array
            transform: Image transformations (e.g., PyTorch transforms)
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get a single image and corresponding label
        img = self.images[idx]  # Original image shape: [H, W, C] or [1, H, W, C]
        label = self.labels[idx]

        # Ensure label is a 1D tensor
        if isinstance(label, np.ndarray):  # If it's a NumPy array, convert to integer
            label = label.item()  # Use .item() to avoid deprecated behavior
        label = torch.tensor(label, dtype=torch.long)  # Convert to PyTorch tensor (long type)

        # Check and adjust image dimensions to [C, H, W]
        if img.ndim == 4 and img.shape[0] == 1:  # [1, H, W, C]
            img = img.squeeze(0)  # Remove the 0th dimension, become [H, W, C]
        if img.ndim == 3 and img.shape[-1] == 3:  # [H, W, C] -> [C, H, W]
            img = np.transpose(img, (2, 0, 1))

        # Convert to PyTorch Tensor
        img = torch.tensor(img, dtype=torch.float32)

        # Apply transform if available
        if self.transform:
            img = self.transform(img)

        return img, label

def load_npz_data(npz_file, batch_size=32):
    """
    Load .npz file and create data loaders
    Args:
        npz_file (str): Path to the .npz file
        batch_size (int): Batch size
    Returns:
        train_loader, val_loader, test_loader: Data loaders
    """
    # Load npz file
    data = np.load(npz_file)
    train_images, train_labels = data["train_images"], data["train_labels"]
    val_images, val_labels = data["val_images"], data["val_labels"]
    test_images, test_labels = data["test_images"], data["test_labels"]

    # Normalize images to [0, 1] range
    train_images = train_images.astype(np.float32) / 255.0
    val_images = val_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0

    # # Add channel dimension (from (N, 64, 64) -> (N, 1, 64, 64))
    # train_images = train_images[:, np.newaxis, :, :]
    # val_images = val_images[:, np.newaxis, :, :]
    # test_images = test_images[:, np.newaxis, :, :]

    # Define data augmentation and preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
        transforms.RandomCrop((200, 200)),  # Random cropping
        transforms.Normalize(mean=[0.5], std=[0.5])  # Standardize
    ])

    # Create PyTorch datasets
    train_dataset = DermaMNISTDataset(train_images, train_labels, transform=transform)
    val_dataset = DermaMNISTDataset(val_images, val_labels, transform=transform)
    test_dataset = DermaMNISTDataset(test_images, test_labels, transform=transform)

    return train_dataset, val_dataset, test_dataset

def check_sample_distribution(dataset, split_name):
    """
    Check the number of samples and class distribution in the dataset
    Args:
        dataset: PyTorch dataset
        split_name (str): Name of the split (e.g., 'train', 'val', 'test')
    """
    total_samples = len(dataset)
    labels = [label.item() for _, label in dataset]
    class_distribution = {label: labels.count(label) for label in set(labels)}
    imbalance = max(class_distribution.values()) / min(class_distribution.values())
    print(f"{split_name.capitalize()} Split:")
    print(f"  Total Samples: {total_samples}")
    print(f"  Class Distribution: {class_distribution}")
    print(f"  Imbalance Ratio (max/min): {imbalance:.2f}\n")

def visualize_samples(dataset, num_samples=5):
    """
    Visualize dataset samples
    Args:
        dataset: PyTorch dataset
        num_samples (int): Number of samples to display
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        img, label = dataset[i]
        # If the image is a PyTorch tensor, convert to NumPy array
        if torch.is_tensor(img):
            img = img.numpy()

        # Dynamically adjust image dimensions
        if img.ndim == 3 and img.shape[0] == 3:  # RGB: [C, H, W] -> [H, W, C]
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 3 and img.shape[0] == 1:  # Grayscale: [1, H, W] -> [H, W]
            img = img.squeeze(0)
        elif img.ndim == 2:  # [H, W], no adjustment needed
            pass
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")

        # De-normalize back to [0, 1]
        img = img * 0.5 + 0.5

        axes[i].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i].set_title(f"Sample Label: {label}")  # Updated label text
        axes[i].axis("off")
    plt.show()

def data_loader():
    train_dataset, val_dataset, test_dataset = load_npz_data(data_path)
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}\n")

    check_sample_distribution(train_dataset, "train")
    check_sample_distribution(val_dataset, "val")
    check_sample_distribution(test_dataset, "test")

    # visualize_samples(train_dataset)

    return train_dataset, val_dataset, test_dataset

# data_loader(data_path)