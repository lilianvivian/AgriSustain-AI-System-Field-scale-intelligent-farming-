import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.data_module import PlantDiseaseDataModule
from src.model import PlantDiseaseModel


# -----------------------------
# CONFIG
# -----------------------------
DATA_DIR = "/teamspace/studios/this_studio/.cache/kagglehub/datasets/emmarex/plantdisease/versions/1/PlantVillage"
BATCH_SIZE = 32
EPOCHS = 10


def main():

    # -----------------------------
    # DATA MODULE
    # -----------------------------
    data_module = PlantDiseaseDataModule(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE
    )

    data_module.setup()

    num_classes = len(data_module.classes)
    print(f"Number of classes: {num_classes}")

    # -----------------------------
    # MODEL
    # -----------------------------
    model = PlantDiseaseModel(num_classes=num_classes)

    # -----------------------------
    # CHECKPOINTING (ENTERPRISE FEATURE)
    # -----------------------------
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="agrisustain-{epoch:02d}-{val_acc:.3f}"
    )

    # -----------------------------
    # TRAINER (ENTERPRISE CORE)
    # -----------------------------
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        devices="auto",
        callbacks=[checkpoint_callback],
        log_every_n_steps=10
    )

    # -----------------------------
    # TRAIN
    # -----------------------------
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()