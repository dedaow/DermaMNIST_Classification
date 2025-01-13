import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from resnet_net import ResNet18
from dataload import data_loader

def train_resnet18(train_dataset, val_dataset, num_classes=7, epochs=20, batch_size=32, learning_rate=0.001, early_stopping_patience=5):
    """
    Train the ResNet-18 model with specified parameters, including early stopping and checkpointing.

    Args:
        train_dataset (Dataset): PyTorch dataset for training.
        val_dataset (Dataset): PyTorch dataset for validation.
        num_classes (int): Number of output classes.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for DataLoader.
        learning_rate (float): Learning rate for optimizer.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.
    """
    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the ResNet-18 model
    model = ResNet18(num_classes=num_classes)

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # To store results and best model checkpoint
    results = []
    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        # Validation phase
        val_loss, val_accuracy = evaluate_resnet18(model, val_loader, criterion, device, epoch, epochs, results, train_loss, train_accuracy)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model checkpoint
            torch.save(model.state_dict(), "model/resnet18_checkpoint.pth")
            print("Model checkpoint saved.")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Save final training results
    with open("eval/resnet/resnet18_evaluation_training.txt", "w") as f:
        for result in results:
            f.write(f"Epoch: {result['epoch']}, Train Loss: {result['train_loss']:.4f}, Train Accuracy: {result['train_accuracy']:.2f}%, ")
            f.write(f"Validation Loss: {result['val_loss']:.4f}, Validation Accuracy: {result['val_accuracy']:.2f}%\n")
    print("Evaluation results saved to evaluation_training.txt")

    print("Training complete.")

def evaluate_resnet18(model, data_loader, criterion, device, epoch, epochs, results, train_loss, train_accuracy):
    """
    Evaluate the ResNet-18 model and save results.

    Args:
        model (nn.Module): The trained ResNet-18 model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): The device to run the evaluation on.
        epoch (int): Current epoch number.
        epochs (int): Total number of epochs.
        results (list): List to store evaluation results.
        train_loss (float): Training loss of the current epoch.
        train_accuracy (float): Training accuracy of the current epoch.

    Returns:
        val_loss (float): Validation loss.
        val_accuracy (float): Validation accuracy.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss /= len(data_loader)
    val_accuracy = 100.0 * correct / total
    print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    results.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,  # Added train loss to results
        "train_accuracy": train_accuracy,  # Added train accuracy to results
        "val_loss": val_loss,
        "val_accuracy": val_accuracy
    })

    return val_loss, val_accuracy

train_dataset, val_dataset, test_dataset = data_loader()
train_resnet18(train_dataset, val_dataset)
