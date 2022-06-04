# Note:
# to reproduce this example: 
# I am using torch==1.11.0, pytorch-lightning=1.6.3
# for MONAI, it was installed with  pip install -q "monai-weekly[nibabel]"
# Currently, there is bug related to protobuf version: 
# https://github.com/PyTorchLightning/pytorch-lightning/issues/13159
# so, I have to do the following:
# pip install "protobuf<4.21.0"


import pytorch_lightning
from monai.utils import set_determinism
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import CacheDataset, PersistentDataset, list_data_collate

from typing import Dict, List, Sequence, Tuple, Union
import numpy as np
from monai.data import ImageReader
from monai.utils import ensure_tuple, require_pkg
from monai.config import PathLike
from monai.data.image_reader import _stack_images
from aicsimageio import AICSImage

from aicsimageio.writers import OmeTiffWriter
import torch
import tempfile
import os


@require_pkg(pkg_name="aicsimageio")
class bio_reader(ImageReader):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            img_.append(AICSImage(f"{name}"))

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            data = img_obj.get_image_data(**self.kwargs)
            img_array.append(data)

        return _stack_images(img_array, {}), {}

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True


def creat_random_data(num_images: int = 10):
    # Generate some image data
    tempdir = tempfile.mkdtemp()
    img_list = []
    seg_list = []
    for img_idx in range(num_images):
        seg_ = np.random.randint(0, 2, size=(64, 256, 128)).astype(np.uint8)
        # # version 1: random
        # img_ = np.random.randint(20, 275, size=(64, 256, 128)).astype(np.uint16)

        # # version 2: with pixel-to-pixel correspondence to easily debug dataloader
        # img_ = seg_.copy().astype(np.uint16)
        # img_[img_ > 0] = 100

        # version 3: different size image
        img_ = np.random.randint(20, 275, size=(32, 64, 64)).astype(np.uint16)
        seg_[0, 0:32, 0:32] = 0

        img_name = os.path.join(tempdir, "img_" + str(img_idx) + ".tiff")
        seg_name = os.path.join(tempdir, "seg_" + str(img_idx) + ".tiff")
        OmeTiffWriter.save(img_, img_name, dim_order="ZYX")
        OmeTiffWriter.save(seg_, seg_name, dim_order="ZYX")
        img_list.append(img_name)
        seg_list.append(seg_name)

    return img_list, seg_list, tempdir


class myDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # create a random dataset
        train_images, train_labels, tmpdir = creat_random_data(num_images=12)
        print(f"temporay directory created at {tmpdir}")

        train_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(train_images, train_labels)
        ]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"], reader=bio_reader, dimension_order_out= "ZYX", C=0, T=0),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area

                # RandCropByPosNegLabeld(
                #    keys=["image", "label"],
                #    label_key="label",
                #    spatial_size=(32, 64, 64),
                #    pos=1,
                #    neg=1,
                #    num_samples=4,
                #    image_key="image",
                #    image_threshold=0,
                #),
                RandSpatialCropSamplesd(
                    keys=["image", "label"],
                    random_size=False,
                    num_samples=4,
                    roi_size=(32, 64, 64),
                ),
                EnsureTyped(keys=["image", "label"]),
            ]
        )

        # try two different ways of caching datasets
        """
        self.train_ds = SmartCacheDataset(
            data=train_files, transform=train_transforms,
            replace_rate=0.2, num_replace_workers=4,
            cache_rate=0.5, num_init_workers=4
        )
        """

        """
        self.train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=0.5, num_workers=4,
        )
        """

        self.train_ds = PersistentDataset(
            data=train_files[:8], cache_dir="./tmp/", transform=train_transforms
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=2,
            num_workers=2,
            collate_fn=list_data_collate,
            persistent_workers=False
        )
        return train_loader


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.post_pred = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([EnsureType("tensor", device="cpu"), AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.best_val_dice = 0
        self.best_val_epoch = 0

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        sample_id = np.random.randint(1000, 9000)
        OmeTiffWriter.save(images.detach().cpu().numpy().astype(np.float), f"image_{sample_id}.tiff", dim_order="TCZYX")
        OmeTiffWriter.save(labels.detach().cpu().numpy().astype(np.float), f"label_{sample_id}.tiff", dim_order="TCZYX")
        return {"loss": loss}


if __name__ == '__main__':

    # initialise the LightningModule
    net = Net()
    dm = myDataModule()
    dm.prepare_data()

    # initialise Lightning's trainer (basic pytorch lightning).
    trainer = pytorch_lightning.Trainer(
        gpus=2,
        precision=16,
        enable_checkpointing=True,
        logger=False,
        max_epochs=5
    )

    # train
    trainer.fit(net, datamodule=dm)
