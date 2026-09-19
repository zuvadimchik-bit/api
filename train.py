import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from model import CargoDataset, CargoPredictor, get_transforms


def seed_everything(seed: int = 42):
    """Фиксация всех генераторов случайных чисел для 100% воспроизводимости."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    train_df = pd.read_csv("data/train/train.csv")
    groups_df = pd.read_csv("data/train/train_groups.csv")
    df = pd.merge(train_df, groups_df, on="image_id").reset_index(drop=True)

    N_SPLITS = 5
    EPOCHS = 15
    BATCH_SIZE = 16
    LR = 3e-4

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_predictions = np.zeros(len(df))
    fold_maes = []

    print(f"Запуск {N_SPLITS}-Fold обучения (по {EPOCHS} эпох на фолд)...")

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["group_id"])):
        print(f"\n{'=' * 25} ФОЛД {fold + 1}/{N_SPLITS} {'=' * 25}")

        train_data = df.iloc[train_idx].reset_index(drop=True)
        val_data = df.iloc[val_idx].reset_index(drop=True)

        train_loader = DataLoader(
            CargoDataset(train_data, "data/train/images", get_transforms(True)),
            batch_size=BATCH_SIZE,
            shuffle=True,
            worker_init_fn=lambda _: np.random.seed(42 + fold),
        )
        val_loader = DataLoader(
            CargoDataset(val_data, "data/train/images", get_transforms(False)),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

        model = CargoPredictor().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        criterion = nn.L1Loss()

        save_path = f"best_model_fold{fold}.pth"
        record_file = f"score_fold{fold}.txt"

        # Читаем исторический рекорд этого фолда (если файл уже есть от прошлых запусков)
        previous_best = float("inf")
        if os.path.exists(record_file):
            try:
                with open(record_file, "r") as f:
                    previous_best = float(f.read().strip())
            except Exception:
                previous_best = float("inf")

        current_run_best_mae = float("inf")
        best_val_preds = None

        for epoch in range(1, EPOCHS + 1):
            model.train()
            for imgs, targets in train_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                optimizer.zero_grad()
                loss = criterion(model(imgs), targets)
                loss.backward()
                optimizer.step()

            scheduler.step()

            model.eval()
            val_preds = []
            with torch.no_grad():
                for imgs, _ in val_loader:
                    preds = model(imgs.to(device)).cpu().numpy()
                    val_preds.extend(preds)

            val_preds = np.array(val_preds)
            val_targets = val_data["load_pct"].values
            mae = float(np.mean(np.abs(val_preds - val_targets)))

            # Сохраняем лучший результат внутри текущего прогона
            if mae < current_run_best_mae:
                current_run_best_mae = mae
                best_val_preds = val_preds

                # Проверяем, побит ли исторический рекорд фолда
                if current_run_best_mae < previous_best:
                    torch.save(model.state_dict(), save_path)
                    with open(record_file, "w") as f:
                        f.write(str(round(current_run_best_mae, 4)))
                    previous_best = current_run_best_mae

            if epoch % 5 == 0 or epoch == EPOCHS:
                acc_10 = np.mean(np.abs(val_preds - val_targets) <= 10.0) * 100.0
                print(f"Эпоха {epoch:02d}/{EPOCHS} | Val MAE: {mae:.2f}% | Точность (<=10 п.п.): {acc_10:.1f}%")

        print(f"-> Лучший результат фолда {fold + 1} в этом запуске: MAE = {current_run_best_mae:.2f}%")
        print(f"   Зафиксированный рекорд фолда: MAE = {previous_best:.2f}% (файл {save_path})")

        oof_predictions[val_idx] = best_val_preds
        fold_maes.append(current_run_best_mae)

    # Итоговый скор по всей выборке (Out-Of-Fold)
    total_mae = np.mean(np.abs(oof_predictions - df["load_pct"].values))
    total_acc_10 = np.mean(np.abs(oof_predictions - df["load_pct"].values) <= 10.0) * 100.0

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ OOF СКОР (по всем 716 фото):")
    print(f"Средняя ошибка (MAE): {total_mae:.2f} п.п.")
    print(f"Доля прогнозов с ошибкой <= 10 п.п.: {total_acc_10:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
