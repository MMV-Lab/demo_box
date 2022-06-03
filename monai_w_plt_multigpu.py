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
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.data import SmartCacheDataset, CacheDataset, list_data_collate
import torch
import tempfile
import nibabel as nib
import numpy as np
import os
from ray_lightning import RayShardedPlugin


def creat_random_data(num_images: int = 10):
    # Generate some image data
    tempdir = tempfile.mkdtemp()
    img_list = []
    seg_list = []
    for img_idx in range(num_images):
        img_ = nib.Nifti1Image(np.random.randint(0, 128, size=(256, 256, 256)), np.eye(4))
        seg_ = nib.Nifti1Image(np.random.randint(0, 2, size=(256, 256, 256)), np.eye(4))
        img_name = os.path.join(tempdir, "img_" + str(img_idx) + ".nii.gz")
        seg_name = os.path.join(tempdir, "seg_" + str(img_idx) + ".nii.gz")
        nib.save(img_, img_name)
        nib.save(seg_, seg_name)
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
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(
                    keys=["image", "label"],
                    pixdim=(1.5, 1.5, 2.0),
                    mode=("bilinear", "nearest"),
                ),
                ScaleIntensityRanged(
                    keys=["image"], a_min=-57, a_max=164,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                # randomly crop out patch samples from
                # big image based on pos / neg ratio
                # the image centers of negative samples
                # must be in valid image area
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(96, 96, 96),
                    pos=1,
                    neg=1,
                    num_samples=4,
                    image_key="image",
                    image_threshold=0,
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

        self.train_ds = CacheDataset(
            data=train_files, transform=train_transforms,
            cache_rate=0.5, num_workers=4,
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
        return {"loss": loss}


if __name__ == '__main__':
    # check multiprocessing context
    print(torch.multiprocessing.get_start_method())

    # initialise the LightningModule
    net = Net()
    dm = myDataModule()
    dm.prepare_data()

    """
    # Create your PyTorch Lightning model here.
    plugin = RayShardedPlugin(num_workers=3, num_cpus_per_worker=2, use_gpu=True)

    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        # precision=16,
        max_epochs=600,
        enable_checkpointing=True,
        logger=False,
        plugins=[plugin]
    )
    """
    # initialise Lightning's trainer.
    trainer = pytorch_lightning.Trainer(
        gpu=2,
        precision=16,
        max_epochs=600,
        enable_checkpointing=True,
        logger=False,
        plugins=[plugin]
    )
    
    # train
    trainer.fit(net, datamodule=dm)