import torch
import matplotlib.pyplot as plt
import pandas as pd
from PartB_MLP import MLP
from PartC_CNN import CNN
from PartA_Data_Loader import get_dataloaders
from evaluation import evaluate
from evaluation import count_parameters

mlp_history = torch.load("mlp_history.pth")
cnn_history = torch.load("cnn_history.pth")

def plot_comparison(mlp_history, cnn_history):
    epochs = range(1, len(mlp_history["val_acc"]) + 1)

    # Validation Accuracy
    plt.figure(figsize=(8,4))
    plt.plot(epochs, mlp_history["val_acc"], label="MLP")
    plt.plot(epochs, cnn_history["val_acc"], label="CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.legend()
    plt.savefig("val_accuracy_comparison.png")
    plt.show()

    # Validation Loss
    plt.figure(figsize=(8,4))
    plt.plot(epochs, mlp_history["val_loss"], label="MLP")
    plt.plot(epochs, cnn_history["val_loss"], label="CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Validation Loss Comparison")
    plt.legend()
    plt.savefig("val_loss_comparison.png")
    plt.show()

plot_comparison(mlp_history, cnn_history)

# test set
train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)

mlp = MLP()
mlp.load_state_dict(torch.load("mlp_model.pth"))
mlp.eval()

cnn = CNN()
cnn.load_state_dict(torch.load("cnn_model.pth"))
cnn.eval()

mlp_test_loss, mlp_test_acc = evaluate(mlp, test_loader)
cnn_test_loss, cnn_test_acc = evaluate(cnn, test_loader)

results = pd.DataFrame({
    "Model": ["MLP", "CNN"],
    "Test Accuracy": [mlp_test_acc, cnn_test_acc]
})
print("\nFinal Test Accuracy:")
print(results.to_string(index=False))

param_table = pd.DataFrame({
    "Model": ["MLP", "CNN"],
    "Trainable Parameters": [count_parameters(mlp), count_parameters(cnn)]
})
print("\nModel Complexity:")
print(param_table.to_string(index=False))
