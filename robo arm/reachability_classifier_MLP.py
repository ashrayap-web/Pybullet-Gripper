import numpy as np
import pandas as pd
import torch

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt


df = pd.read_csv("arm_reachability_down.csv")

success_data = df.loc[df["Result"]==1.0]
fail_data = df.loc[df["Result"]==0.0]

# visualise data
plt.figure(figsize=(10,8))
plt.scatter(success_data["x"], success_data["y"], c='green', marker='o', s=20, label='success')
plt.scatter(fail_data["x"], fail_data["y"], c='red', marker='x', s=20, label='failure')
plt.show()

# preprocessing
X = df.drop(columns=["Result"])
y = df["Result"]

# create train, val and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

# feature Normalisation - returns numpy arrays
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # use fit_transform on train
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)   # use fitted transform to transform test & val same way

# convert to torch tensors
# need .values for y since it was previously a pandas series
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1)  #reshape to col vector
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1,1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)
test_data = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)


# must inherit from nn.Module
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        dropout = 0.15
        self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
                nn.Sigmoid()
        )

    def forward(self,x):
        return self.layers(x)


model = MLP(X_train.shape[1]) # num features in X_train
criterion = nn.BCELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4) #weight_decay=5e-4

best_val_loss = float('inf')
epochs = 200

for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)  # predictions
        loss = criterion(outputs, y_batch)  # MEAN LOSS
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * y_batch.size(0) # total loss

    # put model in evla mode for validation
    model.eval()
    correct_preds = 0
    total_samples = 0
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch) # this gives MEAN LOSS
            val_loss += loss.item() * y_batch.size(0) # Total Loss
            preds_cls = (outputs > 0.5).float() # convert output probs to 1 or 0
            correct_preds += preds_cls.eq(y_batch).sum().item() # count matches

    avg_val_loss = val_loss/len(val_loader.dataset)

    # .dataset gives all samples
    if (epoch+1) % 10 == 0:
        print(f"Epoch: {(epoch+1)}/{epochs}   "  
              f"Train Loss: {train_loss/len(train_loader.dataset):.4f}   "
              f"Val Accuracy: {100*correct_preds/len(val_loader.dataset):.2f}% -> {correct_preds}/{len(val_loader.dataset)}   "
              f"Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        best_model_epoch = epoch

# test accuracy
print(f"loading best model epoch {best_model_epoch}")
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

with torch.no_grad():
    preds = model(X_test)
    preds_class = (preds > 0.5).float()
    accuracy = (preds_class.eq(y_test).sum() / y_test.shape[0]).item()

print(f"\nTest Accuracy: {accuracy:.4f}")


y_test = y_test.numpy()
y_pred = preds_class.numpy()

print("Confusion Matrix", confusion_matrix(y_test, y_pred),"\n")
print("Classification Report", classification_report(y_test, y_pred),"\n")


# --------------- DECISION BOUNDARY -------------------#
X_vals = df.drop(columns=["Result"]).values

x_min, x_max = X_vals[:, 0].min() - 0.05, X_vals[:, 0].max() + 0.05
y_min, y_max = X_vals[:, 1].min() - 0.05, X_vals[:, 1].max() + 0.05
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
grid_df = pd.DataFrame(grid, columns=X.columns)
grid_scaled = scaler.transform(grid_df)
grid_tensor = torch.tensor(grid_scaled, dtype=torch.float32)

with torch.no_grad():
    Z = model(grid_tensor).numpy().reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
plt.scatter(success_data["x"], success_data["y"], c='green', marker='o', s=20, label='Success')
plt.scatter(fail_data["x"], fail_data["y"], c='red', marker='x', s=20, label='Failure')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('MLP Decision Boundary')
plt.legend()
plt.axis('equal')
plt.show()