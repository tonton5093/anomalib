"""Custom COCO Dataset.

This script creates a custom dataset from a coco.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence
from pathlib import Path

import albumentations as A  # noqa: N812
from pandas import DataFrame

from anomalib.data.base import AnomalibDataModule, AnomalibDataset
from anomalib.data.utils import (
    DirType,
    COCOType,
    InputNormalizationMethod,
    LabelName,
    Split,
    TestSplitMode,
    ValSplitMode,
    get_transforms,
)
from anomalib.data.utils.path import (
    _prepare_files_labels,
    _prepare_files_labels_from_coco,
    _prepare_files_labels_and_make_mask_from_coco,
    _resolve_path,
)
from anomalib.utils.types import TaskType


def make_coco_dataset(
    normal_coco: str | Path | Sequence[str | Path],
    normal_dir: str | Path | Sequence[str | Path] | None = None,
    root: str | Path | None = None,
    abnormal_coco: str | Path | Sequence[str | Path] | None = None,
    normal_test_coco: str | Path | Sequence[str | Path] | None = None,
    abnormal_dir: str | Path | Sequence[str | Path] | None = None,
    normal_test_dir: str | Path | Sequence[str | Path] | None = None,
    mask_dir: str | Path | Sequence[str | Path] | None = None,
    split: str | Split | None = None,
    extensions: tuple[str, ...] | None = None,
) -> DataFrame:
    """Make COCO Dataset.

    Args:
        normal_coco (str | Path | Sequence): Path to the COCO.json containing normal images.
        normal_dir (str | Path | Sequence): Path to the directory containing normal images.
            Defaults to ``None``.
        root (str | Path | None): Path to the root directory of the dataset.
            Defaults to ``None``.
        abnormal_coco (str | Path | Sequence | None, optional): Path to the COCO.json containing abnormal images.
            Defaults to ``None``.
        normal_test_coco (str | Path | Sequence | None, optional): Path to the COCO.json containing normal images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing normal images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing the mask annotations.
            Defaults to ``None``.
        split (str | Split | None, optional): Dataset split (ie., Split.FULL, Split.TRAIN or Split.TEST).
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Returns:
        DataFrame: an output dataframe containing samples for the requested split (ie., train or test).

    Examples:
        Assume that we would like to use this ``make_coco_dataset`` to create a dataset from a coco.
        We could then create the dataset as follows,

        .. code-block:: python

            coco_df = make_coco_dataset(
                normal_coco=dataset_root / "good.json",
                abnormal_coco=dataset_root / "crack.json",
                split="train",
            )
            coco_df.head()

        .. code-block:: bash

                      image_path           label  label_index mask_path        split
            0  ./toy/good/00.jpg  DirType.NORMAL            0            Split.TRAIN
            1  ./toy/good/01.jpg  DirType.NORMAL            0            Split.TRAIN
            2  ./toy/good/02.jpg  DirType.NORMAL            0            Split.TRAIN
            3  ./toy/good/03.jpg  DirType.NORMAL            0            Split.TRAIN
            4  ./toy/good/04.jpg  DirType.NORMAL            0            Split.TRAIN
    """

    def _resolve_path_and_convert_to_list(path: str | Path | Sequence[str | Path] | None) -> list[Path]:
        """Convert path to list of paths.

        Args:
            path (str | Path | Sequence | None): Path to replace with Sequence[str | Path].

        Examples:
            >>> _resolve_path_and_convert_to_list("dir")
            [Path("path/to/dir")]
            >>> _resolve_path_and_convert_to_list(["dir1", "dir2"])
            [Path("path/to/dir1"), Path("path/to/dir2")]

        Returns:
            list[Path]: The result of path replaced by Sequence[str | Path].
        """
        if isinstance(path, Sequence) and not isinstance(path, str):
            return [_resolve_path(dir_path, root) for dir_path in path]
        return [_resolve_path(path, root)] if path is not None else []

    # All paths are changed to the List[Path] type and used.
    normal_coco = _resolve_path_and_convert_to_list(normal_coco)
    abnormal_coco = _resolve_path_and_convert_to_list(abnormal_coco)
    normal_test_coco = _resolve_path_and_convert_to_list(abnormal_coco)
    normal_dir = _resolve_path_and_convert_to_list(normal_dir)
    abnormal_dir = _resolve_path_and_convert_to_list(abnormal_dir)
    normal_test_dir = _resolve_path_and_convert_to_list(normal_test_dir)
    mask_dir = _resolve_path_and_convert_to_list(mask_dir)
    assert len(normal_dir) > 0 or len(normal_coco) > 0, "A coco location must be provided in normal_dir or normal_coco."

    filenames = []
    labels = []
    dirs = {}
    cocos ={}

    if normal_dir:
        dirs[DirType.NORMAL] = normal_dir

    if abnormal_dir:
        dirs[DirType.ABNORMAL] = abnormal_dir

    if normal_test_dir:
        dirs[DirType.NORMAL_TEST] = normal_test_dir

    if mask_dir:
        dirs[DirType.MASK] = mask_dir

    for dir_type, paths in dirs.items():
        for path in paths:
            filename, label = _prepare_files_labels(path, dir_type, extensions)
            filenames += filename
            labels += label

    if normal_coco:
        cocos[COCOType.NORMAL] = normal_coco

    if abnormal_coco:
        cocos[COCOType.ABNORMAL] = abnormal_coco

    if normal_test_coco:
        cocos[COCOType.NORMAL_TEST] = normal_test_coco

    for coco_type, paths in cocos.items():
        for path in paths:
            if coco_type == COCOType.ABNORMAL:
                filename, label = _prepare_files_labels_and_make_mask_from_coco(path, coco_type, extensions)
            else:
                filename, label = _prepare_files_labels_from_coco(path, coco_type, extensions)
            filenames += filename
            labels += label

    samples = DataFrame({"image_path": filenames, "label": labels})
    samples = samples.sort_values(by="image_path", ignore_index=True)

    # Create label index for normal (0) and abnormal (1) images.
    samples.loc[
        (samples.label == DirType.NORMAL) | (samples.label == DirType.NORMAL_TEST),
        "label_index",
    ] = LabelName.NORMAL
    samples.loc[(samples.label == DirType.ABNORMAL) | (samples.label == COCOType.ABNORMAL)
    , "label_index"] = LabelName.ABNORMAL
    samples.label_index = samples.label_index.astype("Int64")

    # If a path to mask is provided, add it to the sample dataframe.

    if len(mask_dir) > 0 and len(abnormal_dir) > 0:
        samples.loc[samples.label == DirType.ABNORMAL, "mask_path"] = samples.loc[
            samples.label == DirType.MASK
        ].image_path.to_numpy()
        samples["mask_path"] = samples["mask_path"].fillna("")
        samples = samples.astype({"mask_path": "str"})

    if COCOType.ABNORMAL in list(samples["label"].values):
        samples.loc[samples.label == COCOType.ABNORMAL, "mask_path"] = samples.loc[
            samples.label == COCOType.ABNORMAL
        ].image_path.apply(lambda x: x.as_posix().replace('/Image2/', '/Mask2/'))
        samples["mask_path"] = samples["mask_path"].fillna("")
        samples = samples.astype({"mask_path": "str"})
    
    if (len(mask_dir) > 0 and len(abnormal_dir) > 0) or COCOType.ABNORMAL in list(samples["label"].values):
        # make sure all every rgb image has a corresponding mask image.
        assert (
            samples.loc[samples.label_index == LabelName.ABNORMAL]
            .apply(lambda x: Path(x.image_path).stem in Path(x.mask_path).stem, axis=1)
            .all()
        ), "Mismatch between anomalous images and mask images. Make sure the mask files \
            coco follow the same naming convention as the anomalous images in the dataset \
            (e.g. image: '000.png', mask: '000.png')."
    else:
        samples["mask_path"] = ""

    # remove all the rows with temporal image samples that have already been assigned
    samples = samples.loc[
        (samples.label == DirType.NORMAL) | (samples.label == DirType.ABNORMAL) | (samples.label == DirType.NORMAL_TEST) |
        (samples.label == COCOType.NORMAL) | (samples.label == COCOType.ABNORMAL) | (samples.label == COCOType.NORMAL_TEST)
    ]

    # Ensure the pathlib objects are converted to str.
    # This is because torch dataloader doesn't like pathlib.
    samples = samples.astype({"image_path": "str"})

    # Create train/test split.
    # By default, all the normal samples are assigned as train.
    #   and all the abnormal samples are test.
    samples.loc[(samples.label == DirType.NORMAL) | (samples.label == COCOType.NORMAL), "split"] = Split.TRAIN
    samples.loc[(samples.label == DirType.ABNORMAL) | (samples.label == DirType.NORMAL_TEST) |
                (samples.label == COCOType.ABNORMAL) | (samples.label == COCOType.NORMAL_TEST), "split"] = Split.TEST

    # Get the data frame for the split.
    if split:
        samples = samples[samples.split == split]
        samples = samples.reset_index(drop=True)

    return samples


class COCODataset(AnomalibDataset):
    """COCO dataset.

    This class is used to create a dataset from a coco. The class utilizes the Torch Dataset class.

    Args:
        task (TaskType): Task type. (``classification``, ``detection`` or ``segmentation``).
        transform (A.Compose): Albumentations Compose object describing the transforms that are applied to the inputs.
        normal_coco (str | Path | Sequence): Path to the COCO.json containing normal images.normal_coco (str | Path | Sequence): Path to the COCO.json containing normal images.
        normal_dir (str | Path | Sequence): Path to the directory containing normal images.
            Defaults to ``None``.
        root (str | Path | None): Root coco of the dataset.
            Defaults to ``None``.
        abnormal_coco (str | Path | Sequence | None, optional): Path to the COCO.json containing abnormal images.
            Defaults to ``None``.
        normal_test_coco (str | Path | Sequence | None, optional): Path to the COCO.json containing normal images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.
        abnormal_dir (str | Path | Sequence | None, optional): Path to the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing
            normal images for the test dataset.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing
            the mask annotations.
            Defaults to ``None``.
        split (str | Split | None): Fixed subset split that follows from coco structure on file system.
            Choose from [Split.FULL, Split.TRAIN, Split.TEST]
            Defaults to ``None``.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the directory.
            Defaults to ``None``.

    Raises:
        ValueError: When task is set to classification and `mask_dir` is provided. When `mask_dir` is
            provided, `task` should be set to `segmentation`.

    Examples:
        Assume that we would like to use this ``COCODataset`` to create a dataset from a coco for a classification
        task. We could first create the transforms,

        >>> from anomalib.data.utils import InputNormalizationMethod, get_transforms
        >>> transform = get_transforms(image_size=256, normalization=InputNormalizationMethod.NONE)

        We could then create the dataset as follows,

        .. code-block:: python

            coco_dataset_classification_train = COCODataset(
                normal_coco=dataset_root / "good.json",
                abnormal_coco=dataset_root / "crack.json",
                split="train",
                transform=transform,
                task=TaskType.CLASSIFICATION,
            )

    """

    def __init__(
        self,
        task: TaskType,
        transform: A.Compose,
        normal_coco: str | Path | Sequence[str | Path],
        normal_dir: str | Path | Sequence[str | Path] | None = None,
        root: str | Path | None = None,
        abnormal_coco: str | Path | Sequence[str | Path] | None = None,
        normal_test_coco: str | Path | Sequence[str | Path] | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        split: str | Split | None = None,
        extensions: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(task, transform)

        self.split = split
        self.root = root
        self.normal_coco = normal_coco
        self.abnormal_coco = abnormal_coco
        self.normal_test_coco = normal_test_coco
        self.normal_dir = normal_dir
        self.abnormal_dir = abnormal_dir
        self.normal_test_dir = normal_test_dir
        self.mask_dir = mask_dir
        self.extensions = extensions

    def _setup(self) -> None:
        """Assign samples."""
        self.samples = make_coco_dataset(
            root=self.root,
            normal_coco=self.normal_coco,
            abnormal_coco=self.abnormal_coco,
            normal_test_coco=self.normal_test_coco,
            normal_dir=self.normal_dir,
            abnormal_dir=self.abnormal_dir,
            normal_test_dir=self.normal_test_dir,
            mask_dir=self.mask_dir,
            split=self.split,
            extensions=self.extensions,
        )


class COCO(AnomalibDataModule):
    """COCO DataModule.

    Args:
        normal_coco (str | Path | Sequence): Path to the COCO.json containing normal images.
        normal_dir (str | Path | Sequence): Name of the directory containing normal images.
            Defaults to ``None``. 
        root (str | Path | None): Path to the root coco containing normal and abnormal dirs.
            Defaults to ``None``.
        abnormal_coco (str | Path | Sequence | None, optional): Path to the COCO.json containing abnormal images.
            Defaults to ``None``.
        normal_test_coco (str | Path | Sequence | None, optional): Path to the COCO.json containing normal images for
            the test dataset. Normal test images will be a split of `normal_dir` if `None`.
            Defaults to ``None``.        
        abnormal_dir (str | Path | None | Sequence): Name of the directory containing abnormal images.
            Defaults to ``None``.
        normal_test_dir (str | Path | Sequence | None, optional): Path to the directory containing
            normal images for the test dataset.
            Defaults to ``None``.
        mask_dir (str | Path | Sequence | None, optional): Path to the directory containing
            the mask annotations.
            Defaults to ``None``.
        normal_split_ratio (float, optional): Ratio to split normal training images and add to the
            test set in case test set doesn't contain any normal images.
            Defaults to 0.2.
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.
            Defaults to ``None``.
        image_size (int | tuple[int, int] | None, optional): Size of the input image.
            Defaults to ``(256, 256)``.
        center_crop (int | tuple[int, int] | None, optional): When provided, the images will be center-cropped
            to the provided dimensions.
            Defaults to ``None``.
        normalization (str | InputNormalizationMethod): Normalization method to apply to the input images.
            Defaults to ``InputNormalizationMethod.IMAGENET``.
        train_batch_size (int, optional): Training batch size.
            Defaults to ``32``.
        eval_batch_size (int, optional): Validation, test and predict batch size.
            Defaults to ``32``.
        num_workers (int, optional): Number of workers.
            Defaults to ``8``.
        task (TaskType, optional): Task type. Could be ``classification``, ``detection`` or ``segmentation``.
            Defaults to ``segmentation``.
        transform_config_train (str | A.Compose | None, optional): Config for pre-processing
            during training.
            Defaults to ``None``.
        transform_config_val (str | A.Compose | None, optional): Config for pre-processing
            during validation.
            Defaults to ``None``.
        test_split_mode (TestSplitMode): Setting that determines how the testing subset is obtained.
            Defaults to ``TestSplitMode.FROM_DIR``.
        test_split_ratio (float): Fraction of images from the train set that will be reserved for testing.
            Defaults to ``0.2``.
        val_split_mode (ValSplitMode): Setting that determines how the validation subset is obtained.
            Defaults to ``ValSplitMode.FROM_TEST``.
        val_split_ratio (float): Fraction of train or test images that will be reserved for validation.
            Defaults to ``0.5``.
        seed (int | None, optional): Seed used during random subset splitting.
            Defaults to ``None``.

    Examples:
        The following code demonstrates how to use the ``COCO`` datamodule. Assume that the dataset is structured
        as follows:

        .. code-block:: bash

            $ tree sample_dataset
            sample_dataset
            ├── colour
            │   ├── 00.jpg
            │   ├── ...
            │   └── x.jpg
            ├── crack
            │   ├── 00.jpg
            │   ├── ...
            │   └── y.jpg
            ├── good
            │   ├── ...
            │   └── z.jpg
            ├── LICENSE
            └── mask
                ├── colour
                │   ├── ...
                │   └── x.jpg
                └── crack
                    ├── ...
                    └── y.jpg

        .. code-block:: python

            coco_datamodule = COCO(
                root=dataset_root,
                normal_coco=dataset_root / "good.json",
                abnormal_coco=dataset_root / "crack.json",
                normal_dir="good",
                abnormal_dir="crack",
                task=TaskType.SEGMENTATION,
                mask_dir=dataset_root / "mask" / "crack",
                image_size=256,
                normalization=InputNormalizationMethod.NONE,
            )
            coco_datamodule.setup()

        To access the training images,

        .. code-block:: python

            >> i, data = next(enumerate(coco_datamodule.train_dataloader()))
            >> print(data.keys(), data["image"].shape)

        To access the test images,

        .. code-block:: python

            >> i, data = next(enumerate(coco_datamodule.test_dataloader()))
            >> print(data.keys(), data["image"].shape)
    """

    def __init__(
        self,
        normal_coco: str | Path | Sequence[str | Path],
        normal_dir: str | Path | Sequence[str | Path] | None = None,
        root: str | Path | None = None,
        abnormal_coco: str | Path | Sequence[str | Path] | None = None,
        normal_test_coco: str | Path | Sequence[str | Path] | None = None,
        abnormal_dir: str | Path | Sequence[str | Path] | None = None,
        normal_test_dir: str | Path | Sequence[str | Path] | None = None,
        mask_dir: str | Path | Sequence[str | Path] | None = None,
        normal_split_ratio: float = 0.2,
        extensions: tuple[str] | None = None,
        image_size: int | tuple[int, int] = (256, 256),
        center_crop: int | tuple[int, int] | None = None,
        normalization: str | InputNormalizationMethod = InputNormalizationMethod.IMAGENET,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        task: TaskType = TaskType.SEGMENTATION,
        transform_config_train: str | A.Compose | None = None,
        transform_config_eval: str | A.Compose | None = None,
        test_split_mode: TestSplitMode = TestSplitMode.FROM_DIR,
        test_split_ratio: float = 0.2,
        val_split_mode: ValSplitMode = ValSplitMode.FROM_TEST,
        val_split_ratio: float = 0.5,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            test_split_mode=test_split_mode,
            test_split_ratio=test_split_ratio,
            val_split_mode=val_split_mode,
            val_split_ratio=val_split_ratio,
            seed=seed,
        )
 
        if task == TaskType.SEGMENTATION and test_split_mode == TestSplitMode.FROM_DIR:
            if mask_dir is None and abnormal_coco is None:
                msg = (
                    f"Segmentation task requires mask directory or COCO Datasets if test_split_mode is {test_split_mode}. "
                    "You could set test_split_mode to {TestSplitMode.NONE} or provide a mask directory."
                )
                raise ValueError(
                    msg,
                )

        self.normal_split_ratio = normal_split_ratio
        transform_train = get_transforms(
            config=transform_config_train,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )
        transform_eval = get_transforms(
            config=transform_config_eval,
            image_size=image_size,
            center_crop=center_crop,
            normalization=InputNormalizationMethod(normalization),
        )

        self.train_data = COCODataset(
            task=task,
            transform=transform_train,
            split=Split.TRAIN,
            root=root,
            normal_coco=normal_coco,
            abnormal_coco=abnormal_coco,
            normal_test_coco=normal_test_coco,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )

        self.test_data = COCODataset(
            task=task,
            transform=transform_eval,
            split=Split.TEST,
            root=root,
            normal_coco=normal_coco,
            abnormal_coco=abnormal_coco,
            normal_test_coco=normal_test_coco,
            normal_dir=normal_dir,
            abnormal_dir=abnormal_dir,
            normal_test_dir=normal_test_dir,
            mask_dir=mask_dir,
            extensions=extensions,
        )
