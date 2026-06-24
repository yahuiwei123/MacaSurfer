import json
import os
from typing import Tuple, List, Dict, Any, Optional
from monai.data import PersistentDataset, CacheDataset
from torch.utils.data import DataLoader
# from monai.data import DataLoader


def load_splits(json_path: str) -> List[Dict[str, List[Dict[str, Any]]]]:
    """
    Load cross-validation splits from a JSON file.

    The JSON should be a list of dicts, each with keys 'training' and 'val',
    mapping to lists of data examples (e.g., dicts with 'image' and 'label' paths).
    """
    with open(json_path, 'r') as f:
        splits = json.load(f)
    return splits


def get_datasets(
    images_dir: str,
    labels_dir: str,
    json_path: str,
    fold: int,
    train_transforms: Any,
    val_transforms: Any,
    mode: str = 'persist',
) -> Tuple[PersistentDataset, PersistentDataset]:
    """
    Create training and validation MONAI datasets for a given fold.

    Args:
        json_path: Path to splits_final.json
        fold: index of fold (0-based)
        train_transforms: MONAI transform or Compose for training
        val_transforms: MONAI transform or Compose for validation
        cache_rate: fraction of training data to cache in memory
        num_workers: number of worker processes for caching

    Returns:
        train_ds: CacheDataset with training data
        val_ds: Dataset or CacheDataset with validation data
    """
    splits = load_splits(json_path)
    if fold == 'all':
        # add all fold to lists
        all_train_list = []
        for i in range(len(splits)):
            split = splits[i]
            train_list = split.get('train', [])
            all_train_list.extend(train_list)
        
        train_list = all_train_list
        val_list = all_train_list
    else:
        fold = int(fold)
        if fold < 0 or fold >= len(splits):
            raise ValueError(f"Fold index out of range. Should be in [0, {len(splits)-1}]")

        split = splits[fold]
        train_list = split.get('train', [])
        val_list = split.get('val', [])
    
    # Remove replicative subjects
    train_list = list(set(train_list))
    val_list = list(set(val_list))
    
    # Add base directory
    train_list = [{'image': os.path.join(images_dir, f"{name}_0000.nii.gz"), 'label': os.path.join(labels_dir, f"{name}.nii.gz")} for name in train_list]
    val_list = [{'image': os.path.join(images_dir, f"{name}_0000.nii.gz"), 'label': os.path.join(labels_dir, f"{name}.nii.gz")} for name in val_list]
    
    # Create datasets
    if mode == 'persist':
        train_ds = PersistentDataset(
            data=train_list,
            transform=train_transforms,
            cache_dir='data/cache_dir',
        )
        val_ds = PersistentDataset(
            data=val_list,
            transform=val_transforms,
            cache_dir='data/cache_dir',
        )
    elif mode == 'cache':
        train_ds = CacheDataset(
            data=train_list,
            transform=train_transforms,
            num_workers=32,
            cache_rate=0.5,
            as_contiguous=True
        )
        val_ds = CacheDataset(
            data=val_list,
            transform=val_transforms,
            num_workers=1,
            cache_rate=0.5,
            as_contiguous=True
        )
    else:
        raise ValueError(f"Uncorrect dataset type {mode} specify!")

    return train_ds, val_ds


def get_dataloaders(
    train_ds: PersistentDataset,
    val_ds: PersistentDataset,
    batch_size: int,
    train_shuffle: bool = True,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Wrap datasets in DataLoaders.

    Args:
        train_ds: MONAI CacheDataset for training
        val_ds: MONAI Dataset for validation
        batch_size: batch size for both loaders
        train_shuffle: whether to shuffle training data
        num_workers: number of workers for DataLoader

    Returns:
        train_loader, val_loader
    """
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader, val_loader
