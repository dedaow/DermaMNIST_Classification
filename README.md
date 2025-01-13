# README

## **Project Structure Explanation**

This project contains the following components:

- **`dataload.py`**: Script for loading and preprocessing the dataset.
- **`resnet_net.py`**: Implementation of the ResNet-18 architecture.
- **`resnet_trainer.py`**: Training script for ResNet-18.
- **`resnet_evaluate.py`**: Evaluation script that includes metrics reporting, ROC curve plotting, attention visualization, and confusion matrix generation.
- **`swinT_net.py`**: Implementation of the Swin_transformer architecture.
- **`swinT_trainer.py`**: Training script for Swin_transformer.
- **`swinT_evaluate.py`**: Evaluation script that includes metrics reporting, ROC curve plotting, attention visualization, and confusion matrix generation.
- **`eval/`**: Directory to store evaluation results, including ROC curves, confusion matrices, and attention maps.
- **`model/`**: Directory to store the trained model checkpoints.
- **`data/`**: Directory for the dataset.

## **Environment Setup Instructions**

### Requirements

- Python 3.9
- PyTorch 2.2.0
- CUDA 12.1
- Required Python libraries:
  - `torch`
  - `torchvision`
  - `scikit-learn`
  - `matplotlib`
  - `opencv-python`

### Installation

#### 1. Clone the repository:


```
   git clone https://github.com/cherry1113/DermaMNIST_Classification.git

   cd project-name
   ```
   
#### 2.Install dependencies:
pip install -r requirements.txt

#### 3.Download the dataset and place it in the [data/]() directory.
Data Link:https://pan.baidu.com/s/1yTP5-Q2C-uiaQLD8Dy-qRA?pwd=dx2h \
Password: dx2h

## **Running Instructions**

### Training the Model

1.To train the ResNet-18 model, run the following command:


```
python resnet_trainer.py
```
This will save the ResNet-18 model checkpoint [resnet18_checkpoint.pth]() in the [model/]() directory.

2.To train the Swin-transformer model, run the following command:


```
python swinT_trainer.py
```
This will save the Swin-transformer model checkpoint [swinT_checkpoint.pth]() in the [model/]() directory.

### Evaluating the Model

1.To evaluate the ResNet-18 model on the test set, run:

```
python resnet_evaluate.py
```
Evaluation results, including ROC curves, confusion matrices, and attention maps, will be saved in the [eval/]() directory.

2.To evaluate the Swin-transformer model on the test set, run:

```
python swinT_evaluate.py
```
Evaluation results, including ROC curves, confusion matrices, and attention maps, will be saved in the [eval/]() directory.

## **Reproduction Guidelines**
### Dataset Preparation
1.Ensure the dataset is preprocessed in [data/]() as required by the [dataload.py]() script.

2.If using a different dataset, modify the [dataload.py]() file to match the dataset structure.

### Model Training
1.Adjust hyperparameters (e.g., learning rate, batch size, epochs) if needed in [resnet_trainer.py]() to train ResNet-18 model, or in [swinT_trainer.py]() to train Swin-transformer model, and run it.

2.Ensure sufficient computational resources (e.g., GPU support) for faster training.

### Model Evaluation
1.Confirm the model checkpoint is correctly saved in the [model/]() directory.

2.Adjust evaluation settings (e.g., number of visualized samples per class) in [resnet_evaluate.py]() or [swinT_evaluate.py]() if required, and run it.

### Results Analysis
1.Check the [eval/]() directory for generated evaluation files.

2.Use attention maps and confusion matrices to analyze model performance and decision-making.
