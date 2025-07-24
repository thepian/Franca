import os
import random
from typing import List, Optional

import pytorch_lightning as pl
import webdataset as wds
from torchvision.transforms import Compose

from rasa.data.imagenet import logger


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        batch_size: int,
        data_dir: str,
        class_names: List[str],
        train_transforms: Compose,
        num_images: int,
        val_transforms=None,
        size_val_set: int = 10,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.class_names = class_names
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.num_images = num_images
        self.size_val_set = size_val_set
        self.im_train = None
        self.shards = self.make_shards()
        assert all("n" in os.path.basename(shard) for shard in self.shards)
        logger.info(f"Using {len(self.shards)} class tar files for training.")

    def make_shards(self) -> List[str]:
        expected = {cls + ".tar" for cls in self.class_names}
        all_files = os.listdir(self.data_dir)
        selected_shards = [os.path.join(self.data_dir, f) for f in all_files if f in expected]
        if not selected_shards:
            raise ValueError("No matching tar files found for given class names.")

        # logger.info(f"Selected shards:\n" + "\n".join(selected_shards[:10]) + ("..." if len(selected_shards) > 10 else ""))
        random.shuffle(selected_shards)
        return selected_shards

    def __len__(self):
        return self.num_images

    def get_train_dataset_size(self):
        return self.num_images

    def get_val_dataset_size(self):
        return 0

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = (
                wds.WebDataset(self.shards, resampled=False)
                .decode("pil")
                .to_tuple("jpeg", handler=wds.handlers.warn_and_continue)
                .map_tuple(self.train_transforms)
                .map(lambda x: x[0])
                .batched(self.batch_size, partial=False)
                # .with_length(self.num_images)
            )
            self.im_train = dataset
            # self.im_train = SizedIterableDataset(dataset, self.num_images)
            # makes sure that the webDataset Defines the length
            logger.info(f"WebDataset set up with {len(self.shards)} shards.")
        else:
            raise NotImplementedError("No val/test set implemented.")

    def train_dataloader(self):
        return wds.WebLoader(self.im_train, batch_size=None, num_workers=self.num_workers)
