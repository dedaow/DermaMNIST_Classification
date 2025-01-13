import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from dataload import data_loader
from swinT_net import CustomSwinTransformer
import cv2

def evaluate_swin_transformer(test_dataset, model_path, num_classes=7, batch_size=64):
    """
    Evaluate the Swin Transformer model on the test set and report accuracy, precision, recall, F1-score, 
    plot ROC curve, compute AUC, and visualize confusion matrix.

    Args:
        test_dataset (Dataset): PyTorch dataset for testing.
        model_path (str): Path to the trained model's checkpoint.
        num_classes (int): Number of output classes.
        batch_size (int): Batch size for DataLoader.
    """
    # Define test data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the Swin Transformer model
    model = CustomSwinTransformer(num_classes=num_classes)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluation mode
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update metrics
            test_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.extend(probs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print(f"Test Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")

    # Compute ROC curve and AUC for each class
    all_probs = np.array(all_probs)
    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve((np.array(all_labels) == i).astype(int), all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("eval/swinT/swinT_roc_curve.png")
    plt.show()

    # Compute and visualize confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(num_classes))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("eval/swinT/swinT_confusion_matrix.png")
    plt.show()

    # Generate and visualize combined attention maps, limit to 2 per class
    attention_maps = {i: [] for i in range(num_classes)}
    for idx, (inputs, labels) in enumerate(test_loader):
        if all(len(v) >= 2 for v in attention_maps.values()):  # Limit to 2 samples per class
            break

        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True

        outputs = model(inputs)
        _, predicted = outputs.max(1)

        # Generate gradients for attention
        model.zero_grad()
        one_hot = torch.zeros(outputs.shape).to(device)
        one_hot[range(outputs.size(0)), predicted] = 1
        outputs.backward(gradient=one_hot)

        gradients = inputs.grad.detach().cpu().numpy()
        inputs_np = inputs.detach().cpu().numpy()

        # Generate attention maps for the batch
        for i in range(inputs.shape[0]):
            label = labels[i].item()
            if len(attention_maps[label]) >= 2:  # Skip if already have 2 samples for this class
                continue

            gradient = gradients[i].mean(axis=0)  # Average across channels
            attention_map = np.maximum(gradient, 0)
            attention_map /= attention_map.max()  # Normalize to [0, 1]

            # Overlay attention map on the original image
            original_image = inputs_np[i].transpose(1, 2, 0)
            original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

            heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.uint8(255 * original_image), 0.6, heatmap, 0.4, 0)

            attention_maps[label].append((original_image, overlay, label, predicted[i].item()))

    # Combine and save attention maps in a single image
    for label, maps in attention_maps.items():
        if len(maps) == 0:
            continue

        plt.figure(figsize=(10, 5))
        for i, (original, overlay, true_label, pred_label) in enumerate(maps):
            plt.subplot(2, len(maps), i + 1)
            plt.imshow(original)
            plt.title(f"Original (True: {true_label})")
            plt.axis("off")

            plt.subplot(2, len(maps), i + len(maps) + 1)
            plt.imshow(overlay)
            plt.title(f"Attention (Pred: {pred_label})")
            plt.axis("off")

        plt.savefig(f"eval/swinT/swinT_combined_attention_maps/swinT_class_{label}.png")
        plt.show()

    # Save the results
    with open("eval/swinT/swinT_evaluation_testing.txt", "w") as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1-Score: {f1:.2f}\n")
        for i in range(num_classes):
            f.write(f"Class {i} AUC: {roc_auc[i]:.2f}\n")
    print("Test results saved to eval/swinT/swinT_test_results.txt")

# Example usage
train_dataset, val_dataset, test_dataset = data_loader()
evaluate_swin_transformer(test_dataset, model_path="model/swin_transformer_checkpoint.pth")
