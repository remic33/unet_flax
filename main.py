"""Main unet project file. Run model for training & inference"""
from absl import app

from src.data_pipe import load_data
from src.train import Train


def main(argv):
    images_dir = "data/images/image"
    mask_dir = "data/images/label"
    val_percent = 0.2
    batch_size = 6
    train_loader, val_loader = load_data(val_percent=val_percent, images_dir= images_dir, mask_dir= mask_dir, batch_size= batch_size)
    unet = Train()
    unet.run(train_loader=train_loader, val_loader=val_loader)


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
