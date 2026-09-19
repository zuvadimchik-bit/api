import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CargoDataset, CargoPredictor, get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv("data/test/test.csv")

# Собираем все доступные веса моделей
fold_paths = [f"best_model_fold{i}.pth" for i in range(5) if os.path.exists(f"best_model_fold{i}.pth")]
if not fold_paths:
    # Запасной вариант, если есть только одиночный файл
    fold_paths = ["best_model.pth"]

print(f"Найдено моделей для ансамбля: {len(fold_paths)} ({fold_paths})")

models = []
for p in fold_paths:
    m = CargoPredictor()
    m.load_state_dict(torch.load(p, map_location=device))
    m.to(device)
    m.eval()
    models.append(m)

test_loader = DataLoader(
    CargoDataset(test_df, "data/test/images", get_transforms(False), is_test=True),
    batch_size=16, shuffle=False
)

results = []
print("Генерация предсказаний с ансамблем и TTA (зеркалирование)...")

with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device)
        # Зеркальное отражение по горизонтали для TTA
        imgs_flipped = torch.flip(imgs, dims=[3])

        batch_preds = []
        for model in models:
            # Обычный прогон
            p_orig = model(imgs).cpu().numpy()
            # Прогон с зеркальным отражением
            p_flip = model(imgs_flipped).cpu().numpy()
            # Усреднение внутри одной модели (TTA)
            batch_preds.append((p_orig + p_flip) / 2.0)

        # Усреднение по всем моделям ансамбля
        ensemble_preds = np.mean(batch_preds, axis=0)

        for img_id, raw_p in zip(ids, ensemble_preds):
            p = float(raw_p)
            # Пост-процессинг экстремальных значений кузова
            if p >= 96.5:
                p = 100.0
            elif p <= 3.5:
                p = 0.0
            else:
                p = np.clip(p, 0.0, 100.0)

            results.append({"image_id": img_id, "load_pct": round(p, 1)})

sub = pd.DataFrame(results)
sub.to_csv("submission.csv", index=False)

print(f"Готово! Сформирован submission.csv ({len(sub)} строк).")
print("Распределение предсказаний ансамбля:")
print(sub["load_pct"].describe())
