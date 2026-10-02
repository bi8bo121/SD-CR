import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

BASE_DEGRADATIONS = [
    "VI_Blur",
    "VI_Haze",
    "VI_Low_light",
    "VI_Noise",
    "VI_Over_exposure"
]

INTENSITIES = ["slight", "moderate", "average", "extreme"]

DEGRADATION_TYPE_MAP = {
    f"{base}_{intensity}": i * 4 + j
    for i, base in enumerate(BASE_DEGRADATIONS)
    for j, intensity in enumerate(INTENSITIES)
}


class DDLDataset(Dataset):
    def __init__(
        self,
        base_type: str,
        root_dir: str = "datasets/DDL",
        split: str = "train",
        target_size: Tuple[int, int] = (320, 320),
        augment: bool = True,
        resize_all_splits: bool = True
    ):
        self.root_dir = Path(root_dir)
        self.base_type = base_type
        self.split = split
        self.target_size = target_size
        self.augment = augment and (split == "train")
        self.resize_all_splits = resize_all_splits

        self.samples = self._build_samples_list()

        if len(self.samples) == 0:
            logger.warning(f"No samples found for base type {base_type} ({split})")

    def _get_files_by_stem(self, directory: Path):
        files_map = {}
        if not directory.exists():
            return files_map

        for f in directory.iterdir():
            if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
                files_map[f.stem] = f
        return files_map

    def _build_samples_list(self) -> List[Dict]:
        samples = []

        for intensity in INTENSITIES:
            full_deg_type = f"{self.base_type}_{intensity}"
            deg_dir = self.root_dir / self.base_type / full_deg_type / self.split

            if not deg_dir.exists():
                alt_dir = self.root_dir / self.base_type / intensity / self.split
                if alt_dir.exists():
                    deg_dir = alt_dir
                else:
                    continue

            if self.split == "train":
                infrared_dir = deg_dir / "Infrared"
                visible_dir = deg_dir / "Visible"
                visible_gt_dir = deg_dir / "Visible_gt"

                if not all([infrared_dir.exists(), visible_dir.exists(), visible_gt_dir.exists()]):
                    continue

                ir_files = self._get_files_by_stem(infrared_dir)
                vis_files = self._get_files_by_stem(visible_dir)
                gt_files = self._get_files_by_stem(visible_gt_dir)

                for stem, ir_path in sorted(ir_files.items()):
                    if stem in vis_files and stem in gt_files:
                        samples.append({
                            'infrared': str(ir_path),
                            'visible': str(vis_files[stem]),
                            'visible_gt': str(gt_files[stem]),
                            'has_gt': True,
                            'degradation_type': full_deg_type,
                            'degradation_type_id': DEGRADATION_TYPE_MAP.get(full_deg_type, -1)
                        })

            elif self.split in ["val", "test"]:
                infrared_dir = deg_dir / "Infrared"
                visible_dir = deg_dir / "Visible"

                if not all([infrared_dir.exists(), visible_dir.exists()]):
                    continue

                ir_files = self._get_files_by_stem(infrared_dir)
                vis_files = self._get_files_by_stem(visible_dir)

                for stem, ir_path in sorted(ir_files.items()):
                    if stem in vis_files:
                        samples.append({
                            'infrared': str(ir_path),
                            'visible': str(vis_files[stem]),
                            'visible_gt': None,
                            'has_gt': False,
                            'degradation_type': full_deg_type,
                            'degradation_type_id': DEGRADATION_TYPE_MAP.get(full_deg_type, -1)
                        })

        return samples

    def _resize_if_needed(self, img: Image.Image) -> Image.Image:
        if self.resize_all_splits:
            if img.size != (self.target_size[1], self.target_size[0]):
                img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        else:
            if self.split == "train" and img.size != (self.target_size[1], self.target_size[0]):
                img = img.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        return img

    def _load_image_vis(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('RGB')
        img = self._resize_if_needed(img)
        return TF.to_tensor(img)

    def _load_image_ir(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert('L')
        img = self._resize_if_needed(img)
        return TF.to_tensor(img)

    def _random_crop(self, *images) -> Tuple:
        patch_size = 256
        h, w = images[0].shape[-2:]
        if h < patch_size or w < patch_size:
            return images
        y = np.random.randint(0, h - patch_size + 1)
        x = np.random.randint(0, w - patch_size + 1)
        return tuple(img[:, y:y + patch_size, x:x + patch_size] for img in images)

    def _horizontal_flip(self, *images) -> Tuple:
        if np.random.rand() > 0.5:
            return tuple(torch.flip(img, dims=[-1]) for img in images)
        return images

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        infrared = self._load_image_ir(sample['infrared'])
        visible = self._load_image_vis(sample['visible'])

        deg_type = sample['degradation_type']
        deg_id = sample['degradation_type_id']

        if sample['has_gt']:
            visible_gt = self._load_image_vis(sample['visible_gt'])

            if self.augment:
                infrared, visible, visible_gt = self._random_crop(infrared, visible, visible_gt)
                infrared, visible, visible_gt = self._horizontal_flip(infrared, visible, visible_gt)

            return {
                'infrared_hq': infrared,
                'visible_lq': visible,
                'visible_hq': visible_gt,
                'degradation_type': deg_type,
                'degradation_type_id': torch.tensor(deg_id, dtype=torch.long)
            }
        else:
            return {
                'infrared_hq': infrared,
                'visible_lq': visible,
                'visible_hq': visible,
                'degradation_type': deg_type,
                'degradation_type_id': torch.tensor(deg_id, dtype=torch.long)
            }


class MultiDegradationBatchSampler:
    BASE_DEGRADATION_TYPES = [
        "VI_Blur",
        "VI_Haze",
        "VI_Low_light",
        "VI_Noise",
        "VI_Over_exposure"
    ]

    def __init__(
        self,
        root_dir: str = "datasets/DDL",
        batch_size: int = 10,
        split: str = "train",
        target_size: Tuple[int, int] = (320, 320),
        augment: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        type_batch_map: Dict[str, int] = None
    ):
        self.batch_size = batch_size
        self.split = split
        self.datasets = {}
        self.loaders = {}

        for base_type in self.BASE_DEGRADATION_TYPES:
            dataset = DDLDataset(
                base_type=base_type,
                root_dir=root_dir,
                split=split,
                target_size=target_size,
                augment=augment and (split == "train"),

                resize_all_splits=(split == "train")
            )
            if len(dataset) > 0:
                self.datasets[base_type] = dataset

        self.valid_degradation_types = list(self.datasets.keys())
        self.num_types = len(self.valid_degradation_types)

        if self.num_types == 0:
            raise ValueError(f"No valid degradation types found for split='{split}'!")

        if split == "train":

            if type_batch_map is None:
                if batch_size % self.num_types != 0:
                    raise ValueError(f"batch_size={batch_size} must be divisible by num_types={self.num_types}")
                per_type = batch_size // self.num_types
                self.type_batch_map = {t: per_type for t in self.valid_degradation_types}
            else:
                self.type_batch_map = {t: type_batch_map.get(t, 0) for t in self.valid_degradation_types}
                if sum(self.type_batch_map.values()) != batch_size:
                    raise ValueError(f"sum(type_batch_map.values()) must equal batch_size={batch_size}")
                if any(v <= 0 for v in self.type_batch_map.values()):
                    raise ValueError("Every valid type in type_batch_map must have positive batch size")
        else:

            self.type_batch_map = {t: 1 for t in self.valid_degradation_types}

        for base_type in self.valid_degradation_types:
            bs_i = self.type_batch_map[base_type]
            self.loaders[base_type] = DataLoader(
                self.datasets[base_type],
                batch_size=bs_i,  
                shuffle=(split == "train"),
                num_workers=num_workers if split == "train" else 0,
                pin_memory=pin_memory,
                drop_last=(split == "train")
            )

        if split == "train":
            self.steps_per_epoch = max(
                len(self.datasets[t]) // self.type_batch_map[t]
                for t in self.valid_degradation_types
            )
        else:
            self.steps_per_epoch = min(len(loader) for loader in self.loaders.values())

        logger.info(f"Valid base types ({split}): {self.valid_degradation_types}")
        logger.info(f"Type batch map ({split}): {self.type_batch_map}")
        logger.info(f"Steps per epoch ({split}): {self.steps_per_epoch}")

    def __iter__(self):
        iters = {k: iter(v) for k, v in self.loaders.items()}

        for _ in range(self.steps_per_epoch):
            combined_batch = {}

            for base_type in self.valid_degradation_types:
                try:
                    batch = next(iters[base_type])
                except StopIteration:
                    iters[base_type] = iter(self.loaders[base_type])
                    batch = next(iters[base_type])

                if not combined_batch:
                    combined_batch = {key: [] for key in batch.keys()}

                for key, value in batch.items():
                    combined_batch[key].append(value)

            final_batch = {}
            for key, values in combined_batch.items():
                if isinstance(values[0], torch.Tensor):

                    if self.split == "train":
                        final_batch[key] = torch.cat(values, dim=0)
                    else:

                        pass
                else:
                    if self.split == "train":
                        merged = []
                        for v in values:
                            if isinstance(v, list):
                                merged.extend(v)
                            else:
                                merged.append(v)
                        final_batch[key] = merged

            if self.split == "train":
                yield final_batch
            else:

                for i, base_type in enumerate(self.valid_degradation_types):
                    one = {}

                    for key, vals in combined_batch.items():
                        v = vals[i]
                        if isinstance(v, torch.Tensor):

                            one[key] = v
                        else:

                            one[key] = v
                    yield one

    def __len__(self):
        if self.split == "train":
            return self.steps_per_epoch

        return self.steps_per_epoch * len(self.valid_degradation_types)


def create_dataloaders(
    root_dir: str = "datasets/DDL",
    batch_size: int = 10,
    val_batch_size: int = 1, 
    num_workers: int = 4,
    target_size: Tuple[int, int] = (320, 320),
    train_type_batch_map: Dict[str, int] = None
):
    logger.info("\n" + "=" * 80)
    logger.info("Creating Unified Dataloaders (5 degradations, Rain removed)")
    logger.info("=" * 80)
    logger.info("Train: /DDL/*/train/ (with GT)")
    logger.info("Val:   /DDL/*/test/ (no GT, no resize, batch=1)")
    logger.info("Test:  /DDL/*/test/ (no GT, no resize, batch=1)")
    logger.info(f"Train batch size: {batch_size}")
    logger.info(f"Val/Test batch size (requested): {val_batch_size}")
    logger.info("=" * 80 + "\n")

    train_sampler = MultiDegradationBatchSampler(
        root_dir=root_dir,
        batch_size=batch_size,
        split="train",
        target_size=target_size,
        augment=True,
        num_workers=num_workers,
        pin_memory=True,
        type_batch_map=train_type_batch_map
    )

    val_sampler = MultiDegradationBatchSampler(
        root_dir=root_dir,
        split="test",
        target_size=target_size,
        batch_size=1,  
        augment=False,
        num_workers=0,
        pin_memory=True,
        type_batch_map=None
    )

    test_sampler = MultiDegradationBatchSampler(
        root_dir=root_dir,
        batch_size=1,   
        split="test",
        target_size=target_size,
        augment=False,
        num_workers=0,
        pin_memory=True,
        type_batch_map=None
    )

    class IterableLoader:
        def __init__(self, sampler):
            self.sampler = sampler

        def __iter__(self):
            return iter(self.sampler)

        def __len__(self):
            return len(self.sampler)

    return IterableLoader(train_sampler), IterableLoader(val_sampler), IterableLoader(test_sampler)
