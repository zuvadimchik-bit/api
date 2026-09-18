import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import timm

def get_transforms(is_train: bool = True):
    if is_train:
        return T.Compose([
            T.Resize((288, 288)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.25, contrast=0.25),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    return T.Compose([
        T.Resize((288, 288)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class CargoDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row['image_id']
        path = f"{self.img_dir}/{img_id}.jpg"
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (288, 288), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return image, img_id
        return image, torch.tensor(row['load_pct'], dtype=torch.float32)

class CargoPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.num_features, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return (self.head(self.backbone(x)) * 100.0).squeeze(-1)
