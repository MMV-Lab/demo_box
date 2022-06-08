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
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    ScaleIntensityRanged,
    EnsureTyped,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.data import PersistentDataset, list_data_collate

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
    A_list = []
    B_list = []
    for img_idx in range(num_images):
        # image for domain A
        domainA_ = np.random.randint(0, 255, size=(64, 256, 128)).astype(np.uint16)

        # different size image for domain B
        domainB_ = np.random.randint(0, 2, size=(64, 256, 128)).astype(np.uint8)

        # save the images
        A_name = os.path.join(tempdir, "imgA_" + str(img_idx) + ".tiff")
        B_name = os.path.join(tempdir, "imgB_" + str(img_idx) + ".tiff")
        OmeTiffWriter.save(domainA_, A_name, dim_order="ZYX")
        OmeTiffWriter.save(domainB_, B_name, dim_order="ZYX")
        A_list.append(A_name)
        B_list.append(B_name)

    return A_list, B_list, tempdir


class myDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self):
        super().__init__()

    def prepare_data(self):
        # create a random dataset
        train_A, train_B, tmpdir = creat_random_data(num_images=12)
        print(f"temporay directory created at {tmpdir}")

        train_files = [
            {"imageA": image_A, "imageB": label_B}
            for image_A, label_B in zip(train_A, train_B)
        ]

        # set deterministic training for reproducibility
        set_determinism(seed=0)

        # define the data transforms
        train_transforms = Compose(
            [
                LoadImaged(
                    keys=["imageA", "imageB"],
                    reader=bio_reader,
                    dimension_order_out="ZYX",
                    C=0,
                    T=0,
                ),
                AddChanneld(keys=["imageA", "imageB"]),
                # version 1
                RandSpatialCropd(
                    keys=["imageA"],
                    random_size=False,
                    roi_size=(64, 64, 64),
                ),
                RandSpatialCropd(
                    keys=["imageB"],
                    random_size=False,
                    roi_size=(64, 64, 64),
                ),
                # # version 2:
                # RandSpatialCropSamplesd(
                #     keys=["imageA", "imageB"],
                #     random_size=False,
                #     num_samples=4,
                #     roi_size=(64, 64, 64),
                # ),
                EnsureTyped(keys=["imageA", "imageB"]),
            ]
        )

        self.train_ds = PersistentDataset(
            data=train_files, cache_dir="./tmp/", transform=train_transforms
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=2,
            num_workers=2,
            collate_fn=list_data_collate,
            persistent_workers=False,
        )
        return train_loader


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self._model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128),
            strides=(2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)

    def forward(self, x):
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self._model.parameters(), 1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        imageA, imageB = batch["imageA"], batch["imageB"]
        output = self.forward(imageA)
        loss = self.loss_function(output, imageB)
        return {"loss": loss}


if __name__ == "__main__":

    # initialise the LightningModule
    net = Net()
    dm = myDataModule()
    dm.prepare_data()

    # initialise Lightning's trainer (basic pytorch lightning).
    trainer = pytorch_lightning.Trainer(
        gpus=1, precision=16, enable_checkpointing=True, logger=False, max_epochs=5
    )

    # train
    trainer.fit(net, datamodule=dm)
