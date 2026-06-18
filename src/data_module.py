import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


class PlantDiseaseDataModule(pl.LightningDataModule):

    def __init__(self, data_dir, batch_size=32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def setup(self, stage=None):
        full_dataset = datasets.ImageFolder(
            root=self.data_dir,
            transform=self.transform
        )

        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size

        self.train_ds, self.val_ds = random_split(
            full_dataset,
            [train_size, val_size]
        )

        self.classes = full_dataset.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )