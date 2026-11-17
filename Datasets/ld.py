"""LD gear/shaft anomaly dataset, with explicit box-derived masks.

Original val is exposed as TEST. There is no independent VAL split. The data
format and label semantics are documented in docs/ld_dataset.md.
"""
from enum import Enum
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


CLASSNAMES = ("gear_teeth", "shaft_cam26", "shaft_cam5", "shaft_cam9")


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, source, classname="ALL", resize=256, imagesize=224,
                 split=DatasetSplit.TRAIN, clip_transformer=None, k_shot=0,
                 random_seed=42, divide_num=1, divide_iter=0, **kwargs):
        self.source = str(source)
        self.split = split if isinstance(split, DatasetSplit) else DatasetSplit(split)
        if self.split is DatasetSplit.VAL:
            raise ValueError("LD original val is TEST; there is no independent VAL split")
        self.classnames_to_use = list(CLASSNAMES) if classname == "ALL" else [classname]
        if not set(self.classnames_to_use) <= set(CLASSNAMES):
            raise ValueError(f"Unknown LD category: {classname}; available: {CLASSNAMES}")
        self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        rng = random.Random(random_seed)
        if divide_num > 1:
            if not 0 <= divide_iter < divide_num:
                raise ValueError("divide_iter must be in [0, divide_num)")
            entries = list(self.data_to_iterate)
            rng.shuffle(entries)
            self.data_to_iterate = entries[divide_iter::divide_num]
        if k_shot > 0 and k_shot < len(self.data_to_iterate):
            self.data_to_iterate = rng.sample(self.data_to_iterate, k_shot)
        self.transform_img = clip_transformer if clip_transformer is not None else transforms.Compose([
            transforms.Resize((resize, resize)), transforms.CenterCrop(imagesize), transforms.ToTensor(),
            transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225]),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.CenterCrop(imagesize),
        ])
        self.imagesize = (3, imagesize, imagesize)

    def get_image_data(self):
        images, entries = {}, []
        for category in sorted(self.classnames_to_use):
            base = Path(self.source) / category
            split_dir = base / self.split.value
            if not split_dir.is_dir():
                raise FileNotFoundError(f"LD dataset split is missing: {split_dir}")
            images[category] = {}
            for state in ("good", "defect"):
                paths = sorted((split_dir / state).glob("*.jpg"))
                images[category][state] = [str(p) for p in paths]
                for image_path in paths:
                    mask_path = None
                    if state == "defect":
                        mask_dir = base / "ground_truth"
                        if self.split is DatasetSplit.TRAIN:
                            mask_dir /= "train"
                        mask_path = mask_dir / "defect" / (image_path.stem + "_mask.png")
                        if not mask_path.is_file():
                            raise FileNotFoundError(f"LD anomaly mask is missing: {mask_path}")
                    entries.append([category, state, str(image_path), str(mask_path) if mask_path else None])
            if not any(images[category].values()):
                raise ValueError(f"Empty LD dataset split: {split_dir}")
        return images, entries

    def __len__(self):
        return len(self.data_to_iterate)

    def __getitem__(self, index):
        _, state, image_path, mask_path = self.data_to_iterate[index]
        with Image.open(image_path) as image:
            native_size = image.size
            tensor = self.transform_img(image.convert("RGB"))
        if mask_path:
            with Image.open(mask_path) as image:
                if image.size != native_size:
                    raise ValueError(f"LD image/mask geometry differs: {image_path}")
                mask = np.array(self.transform_mask(image.convert("L")), copy=True)
            mask = torch.from_numpy(mask > 127).unsqueeze(0).float()
        else:
            mask = torch.zeros((1, *tensor.shape[-2:]), dtype=torch.float32)
        if tuple(mask.shape[-2:]) != tuple(tensor.shape[-2:]):
            raise ValueError("LD image and mask transforms produce different spatial sizes")
        return dict(image=tensor, mask=mask, is_anomaly=int(state != "good"), image_path=image_path)
