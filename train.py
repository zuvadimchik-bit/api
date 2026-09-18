import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from model import CargoDataset, CargoPredictor, get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Работаем на: {device}")

train_df = pd.read_csv("data/train/train.csv")
groups_df = pd.read_csv("data/train/train_groups.csv")
df = pd.merge(train_df, groups_df, on="image_id")

# Честная валидация без утечки групп кузовов
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(df, groups=df["group_id"]))

train_loader = DataLoader(
    CargoDataset(df.iloc[train_idx], "data/train/images", get_transforms(True)),
    batch_size=16, shuffle=True
)
val_loader = DataLoader(
    CargoDataset(df.iloc[val_idx], "data/train/images", get_transforms(False)),
    batch_size=16, shuffle=False
)

model = CargoPredictor().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

EPOCHS = 5
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
criterion = nn.L1Loss()

best_mae = float("inf")
print(f"Начинаем углублённое обучение на {EPOCHS} эпох...")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(targets)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]

    model.eval()
    errors = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            preds = model(imgs.to(device)).cpu().numpy()
            errors.extend(np.abs(preds - targets.numpy()))

    mae = np.mean(errors)
    acc_10 = np.mean(np.array(errors) <= 10.0) * 100.0
    print(f"Эпоха {epoch:02d}/{EPOCHS} | LR: {current_lr:.6f} | Ошибка MAE: {mae:.2f}% | Точность (<=10 п.п.): {acc_10:.1f}%")

    if mae < best_mae:
        best_mae = mae
        torch.save(model.state_dict(), "best_model.pth")
        print(f"  -> Новая лучшая модель сохранена (MAE: {best_mae:.2f}%)")

print(f"\nОбучение завершено! Лучший результат MAE: {best_mae:.2f}%")