from PartA_Data_Loader import get_dataloaders
import torch
import torch.nn as nn
from train import train_model
from plot import plot_history
from evaluation import evaluate
from evaluation import count_parameters

import os
'''
    OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
'''
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: (batch_size, 3, 32, 32)
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        '''
            Input size: H * W = 32 * 32
            MaxPool2d output size: H + 2 * padding - dilation * (kernel_size - 1) / stride + 1
            padding is 0, dilation is 1
        '''
        # After block1, the shape of the tensor is (batch_size, 32, 15, 15)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # After block2, the shape of the tensor is (batch_size, 64, 7, 7)
        self.fc1 = nn.Linear(64*7*7, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    cnn = CNN()
    train_loader, val_loader, test_loader =get_dataloaders(batch_size=64)
    cnn_history = train_model(cnn, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=0.0001)
    torch.save(cnn.state_dict(), "cnn_model.pth")
    torch.save(cnn_history, "cnn_history.pth")
    plot_history(cnn_history, save_path="cnn_training_curves.png")
    print("CNN parameters: ", count_parameters(cnn))
    test_loss, test_acc = evaluate(cnn, test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.3f}")