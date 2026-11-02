"""
Lab 13: Gene Accessibility Prediction Using CNNs

Data: HCT116 DNase-seq peaks and the hg38 reference genome.
Input: 600-bp DNA sequences encoded using four channels (A, C, G, T).
Output: Binary classification — peak-associated (1) or no detected peak (0).
"""
#------------------------------------------------------#
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
#-----------------------------------------------------#
# 1) Define cnn
class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv1d(4,32,kernel_size=7,padding=3)
        self.conv2 = nn.Conv1d(32,64,kernel_size=5,padding=2)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(64 * 150, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



def main():
    # 2) Data preparation
    data_path= "/home/ibab/DL-Lab/Data/Lab13_Gene_Accessibility/binary_accessibility_dataset.csv"
    data = pd.read_csv(data_path.replace("~", __import__("os").path.expanduser("~")))

    # one-hot encode DNA sequence : A,C,G,T
    bases = "ACGT"
    X = np.zeros((len(data), 4, 600), dtype=np.float32)

    for i ,seq in enumerate(data["sequence"]):
        for j, base in enumerate(seq):
            X[i, bases.index(base),j] = 1
    y = data["label"].to_numpy(dtype=np.int64)

    # split by genomic coordinates to reduce nearby-regions leaksge
    data["group"] = (data["start"] // 10000).astype(int)
    print(data)

    # Split by genomic coordinates to reduce nearby-region leakage
    data["group"] = (data["start"] // 10000).astype(int)
    groups = data["group"].to_numpy()

    unique_groups = np.unique(groups)
    train_groups, test_groups = train_test_split(unique_groups, test_size=0.2, random_state=999)

    train_idx = np.where(np.isin(groups, train_groups))[0]
    test_idx = np.where(np.isin(groups, test_groups))[0]

    X_train , X_test = X[train_idx], X[test_idx]
    y_train , y_test = y[train_idx], y[test_idx]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=128,
        shuffle=False
    )

    # 3) training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BasicCNN().to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(20):
        model.train()
        total_loss = 0

        for data_batch, target in train_loader:
            data_batch = data_batch.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data_batch)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1:02d}/20 |"
              f"Loss: {total_loss/len(train_loader):.4f}"
        )
    # Evaluation
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data_batch,target in test_loader:
            output = model(data_batch.to(device))

            probs = torch.softmax(output, dim=1)[:,1]
            preds = output.argmax(1)

            all_labels.extend(target.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("ROC-AUC", round(roc_auc_score(all_labels, all_preds), 4))

if __name__ == "__main__":
    main()

