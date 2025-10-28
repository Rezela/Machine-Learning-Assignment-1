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

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # keep batch size, other dimensions flattened
        x = x.view(x.size(0), -1)
        '''
        Requirements:
            a) First, a fully connected layer with 1024 output units, followed by a ReLU activation and a dropout layer (p = 0.5).
            b) Next, a fully connected layer with 512 output units, followed by a ReLU activation and a dropout layer (p = 0.5).
            c) A fully connected layer mapping to the 10 CIFAR-10 classes.
        '''
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    mlp = MLP()
    mlp_history = train_model(mlp, train_loader, val_loader, epochs=10)
    torch.save(mlp.state_dict(), "mlp_model.pth")
    torch.save(mlp_history, "mlp_history.pth")
    plot_history(mlp_history, save_path="mlp_training_curves.png")
    print("MLP parameters: ", count_parameters(mlp))
    test_loss, test_acc = evaluate(mlp, test_loader)
    print(f"Final Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.3f}")
