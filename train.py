import torch
import torch.nn as nn
from tqdm import tqdm

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, weight_decay=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        # train
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Training]"):
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = out.max(1)  # label: choose the class with the highest probability
            correct += pred.eq(y).sum().item()
            total += y.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc = correct / total

        # validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Validation]"):
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item()
                _, pred = out.max(1)
                correct += pred.eq(y).sum().item()
                total += y.size(0)

        val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        tqdm.write(f"Epoch {epoch + 1}/{epochs}\n"
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f}\n"
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.3f}"
              )
    return history
