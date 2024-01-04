"""Path Utils."""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re

import cv2
from enum import Enum
import numpy as np
from pathlib import Path
from pycocotools.coco import COCO

from torchvision.datasets.folder import IMG_EXTENSIONS


class DirType(str, Enum):
    """Dir type names."""

    NORMAL = "normal"
    ABNORMAL = "abnormal"
    NORMAL_TEST = "normal_test"
    NORMAL_DEPTH = "normal_depth"
    ABNORMAL_DEPTH = "abnormal_depth"
    NORMAL_TEST_DEPTH = "normal_test_depth"
    MASK = "mask_dir"


class COCOType(str, Enum):
    """COCO type names."""

    NORMAL = "normal"
    ABNORMAL = "coco_abnormal"
    NORMAL_TEST = "normal_test"
    NORMAL_DEPTH = "normal_depth"
    ABNORMAL_DEPTH = "abnormal_depth"
    NORMAL_TEST_DEPTH = "normal_test_depth"
    MASK = "mask_dir"


def _check_and_convert_path(path: str | Path) -> Path:
    """Check an input path, and convert to Pathlib object.

    Args:
        path (str | Path): Input path.

    Returns:
        Path: Output path converted to pathlib object.
    """
    if not isinstance(path, Path):
        path = Path(path)
    return path


def _prepare_files_labels(
    path: str | Path,
    path_type: str,
    extensions: tuple[str, ...] | None = None,
) -> tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (str | Path): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = [
        f
        for f in path.glob("**/*")
        if f.suffix in extensions and not f.is_dir() and not any(part.startswith(".") for part in f.parts)
    ]
    if not filenames:
        msg = f"Found 0 {path_type} images in {path}"
        raise RuntimeError(msg)

    labels = [path_type] * len(filenames)

    return filenames, labels


def _prepare_files_labels_from_coco(
    path: str | Path,
    path_type: str,
    extensions: tuple[str, ...] | None = None,
) -> tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (str | Path): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = []

    coco = COCO(path)
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        filenames.append(Path(image_info['file_name']))

    if not filenames:
        msg = f"Found 0 {path_type} images in {path}"
        raise RuntimeError(msg)

    labels = [path_type] * len(filenames)

    return filenames, labels


def _prepare_files_labels_and_make_mask_from_coco(
    path: str | Path,
    path_type: str,
    extensions: tuple[str, ...] | None = None,
) -> tuple[list, list]:
    """Return a list of filenames and list corresponding labels.

    Args:
        path (str | Path): Path to the directory containing images.
        path_type (str): Type of images in the provided path ("normal", "abnormal", "normal_test")
        extensions (tuple[str, ...] | None, optional): Type of the image extensions to read from the
            directory.

    Returns:
        List, List: Filenames of the images provided in the paths, labels of the images provided in the paths
    """
    path = _check_and_convert_path(path)
    if extensions is None:
        extensions = IMG_EXTENSIONS

    if isinstance(extensions, str):
        extensions = (extensions,)

    filenames = []

    coco = COCO(path)
    image_ids = coco.getImgIds()

    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        height = image_info['height']
        width = image_info['width']
        file_name = image_info['file_name']
        annotation_ids = coco.getAnnIds(imgIds=image_id, iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)
        annotation_img = np.zeros((height, width), dtype=np.uint8)
        for annotation in annotations:
            annotation_img = np.maximum(annotation_img, coco.annToMask(annotation) * 255)
        match = re.search(r'Image(\d+)', file_name)

        '''
        .../Image1/00.jpg --> .../Mask1/00.jpg
        .../Image2/00.jpg --> .../Mask2/00.jpg
        '''
        if match:
            image_mode = match.group(1)
            mask_path = file_name.replace(f'Image{image_mode}', f'Mask{image_mode}')
        else:
            msg = f"Unknown image mode"
            raise ValueError(msg)

        if not os.path.exists(mask_path):
            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            cv2.imwrite(mask_path, annotation_img)
        if len(annotation_ids) > 0:
            filenames.append(Path(image_info['file_name']))

    if not filenames:
        msg = f"Found 0 {path_type} images in {path}"
        raise RuntimeError(msg)

    labels = [path_type] * len(filenames)

    return filenames, labels


def _resolve_path(folder: str | Path, root: str | Path | None = None) -> Path:
    """Combine root and folder and returns the absolute path.

    This allows users to pass either a root directory and relative paths, or absolute paths to each of the
    image sources. This function makes sure that the samples dataframe always contains absolute paths.

    Args:
        folder (str | Path | None): Folder location containing image or mask data.
        root (str | Path | None): Root directory for the dataset.
    """
    folder = Path(folder)
    if folder.is_absolute():
        path = folder
    # path is relative.
    elif root is None:
        # no root provided; return absolute path
        path = folder.resolve()
    else:
        # root provided; prepend root and return absolute path
        path = (Path(root) / folder).resolve()
    return path
