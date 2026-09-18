import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import CargoDataset, CargoPredictor, get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_df = pd.read_csv("data/test/test.csv")

model = CargoPredictor()
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

test_loader = DataLoader(CargoDataset(test_df, "data/test/images", get_transforms(False), is_test=True), batch_size=16, shuffle=False)

records = []
with torch.no_grad():
    for imgs, ids in test_loader:
        preds = model(imgs.to(device)).cpu().numpy()
        for img_id, p in zip(ids, preds):
            val = float(np.clip(p, 0.0, 100.0))
            records.append({"image_id": img_id, "load_pct": round(val, 1)})

sub = pd.DataFrame(records)
sub.to_csv("submission.csv", index=False)
print("Готово! Файл submission.csv создан и готов к отправке.")