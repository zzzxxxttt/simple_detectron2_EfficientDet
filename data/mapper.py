# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import cv2
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Boxes, BoxMode, Instances

from .images import random_crop, crop_image, color_jittering_, lighting_, random_crop_v2, center_crop_v2

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapper"]


class DatasetMapper:
  """
  A callable which takes a dataset dict in Detectron2 Dataset format,
  and map it into a format used by the model.

  This is the default callable to be used to map your dataset dict into training data.
  You may need to follow it to implement your own one for customized logic,
  such as a different way to read or transform images.
  See :doc:`/tutorials/data_loading` for details.

  The callable currently does the following:

  1. Read the image from "file_name"
  2. Applies cropping/geometric transforms to the image and annotations
  3. Prepare data and annotations to Tensor and :class:`Instances`
  """

  def __init__(self, cfg, is_train=True):

    # fmt: off
    self.img_format = cfg.INPUT.FORMAT
    self.mask_on = cfg.MODEL.MASK_ON
    self.mask_format = cfg.INPUT.MASK_FORMAT
    self.keypoint_on = cfg.MODEL.KEYPOINT_ON
    self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

    assert isinstance(cfg.INPUT.OUTPUT_SIZE, int), 'currently only square output size supported!'
    self.output_size = {'h': cfg.INPUT.OUTPUT_SIZE, 'w': cfg.INPUT.OUTPUT_SIZE}
    self.rand_range = np.arange(cfg.INPUT.RAND_CROP_RANGE[0], cfg.INPUT.RAND_CROP_RANGE[1], 0.1)
    self.padding = 0
    # fmt: on

    self.is_train = is_train

  def __call__(self, dataset_dict):
    """
    Args:
        dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

    Returns:
        dict: a format that builtin models in detectron2 accept
    """
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # USER: Write your own image loading if it's not from a file
    image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
    utils.check_image_size(dataset_dict, image)

    classes = np.array([anno['category_id'] for anno in dataset_dict["annotations"]
                        if anno.get("iscrowd", 0) == 0], dtype=np.float32)
    bboxes = np.array([anno['bbox'] for anno in dataset_dict["annotations"]
                       if anno.get("iscrowd", 0) == 0], dtype=np.float32)

    if len(bboxes) == 0:
      # normally this should not be reached
      bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
      classes = np.array([0])

    bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

    sorted_inds = np.argsort(classes, axis=0)
    bboxes = bboxes[sorted_inds]
    classes = classes[sorted_inds]

    # random crop (for training) or center crop (for validation)
    if self.is_train:
      # image, bboxes = random_crop(image,
      #                             bboxes,
      #                             random_scales=self.rand_range,
      #                             new_size=self.output_size,
      #                             padding=self.padding)
      image, bboxes, offset_x, offset_y, real_w, real_h = \
        random_crop_v2(image, bboxes, random_scales=self.rand_range, new_size=self.output_size)
    else:
      # image, border, offset = crop_image(image,
      #                                    center=[image.shape[0] // 2, image.shape[1] // 2],
      #                                    new_size=[max(image.shape[0:2]), max(image.shape[0:2])])
      # bboxes[:, 0::2] += border[2]
      # bboxes[:, 1::2] += border[0]
      image, bboxes, offset_x, offset_y, real_w, real_h = \
        center_crop_v2(image, bboxes, new_size=self.output_size)

    # resize image and bbox
    # height, width = image.shape[:2]
    # image = cv2.resize(image, (self.output_size['w'], self.output_size['h']))
    # bboxes[:, 0::2] *= self.output_size['w'] / width
    # bboxes[:, 1::2] *= self.output_size['h'] / height

    # discard non-valid bboxes
    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, self.output_size['w'] - 1)
    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, self.output_size['h'] - 1)
    keep_inds = np.logical_and((bboxes[:, 2] - bboxes[:, 0]) > 0,
                               (bboxes[:, 3] - bboxes[:, 1]) > 0)
    bboxes = bboxes[keep_inds]
    classes = classes[keep_inds]

    # randomly flip image and bboxes
    if self.is_train and np.random.uniform() > 0.5:
      image[:] = image[:, ::-1, :]
      bboxes[:, [0, 2]] = image.shape[1] - bboxes[:, [2, 0]] - 1

    # ----------------------------- debug -----------------------------------------
    # import matplotlib.pyplot as plt
    # from matplotlib.patches import Rectangle
    #
    # # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # # plt.show()
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # for lab, bbox in zip(classes, bboxes):
    #   plt.gca().add_patch(Rectangle(bbox[:2], bbox[2] - bbox[0], bbox[3] - bbox[1],
    #                                 linewidth=1, edgecolor='r', facecolor='none'))
    #   # plt.text(bbox[0], bbox[1], self.class_name[lab + 1],
    #   #          bbox=dict(facecolor='b', alpha=0.5), fontsize=7, color='w')
    # plt.show()
    # -----------------------------------------------------------------------------

    # image = image.astype(np.float32) / 255.

    # note do we need color and lighting augmentation ?
    # # randomly change color and lighting
    # if self.is_train:
    #   color_jittering_(self.data_rng, image)
    #   lighting_(self.data_rng, image, 0.1, self.eig_val, self.eig_vec)
    #
    # image -= self.mean
    # image /= self.std
    # image = image.transpose((2, 0, 1))  # [H, W, C] to [C, H, W]

    dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

    dataset_dict["offset_x"] = offset_x
    dataset_dict["offset_y"] = offset_y
    dataset_dict["real_w"] = real_w
    dataset_dict["real_h"] = real_h

    if not self.is_train:
      # USER: Modify this if you want to keep them for some reason.
      dataset_dict.pop("annotations", None)
      dataset_dict.pop("sem_seg_file_name", None)
      return dataset_dict

    if "annotations" in dataset_dict:
      # USER: Modify this if you want to keep them for some reason.
      for anno in dataset_dict["annotations"]:
        if not self.mask_on:
          anno.pop("segmentation", None)
        if not self.keypoint_on:
          anno.pop("keypoints", None)

      target = Instances(image.shape[:2])
      boxes = target.gt_boxes = Boxes(bboxes)
      boxes.clip(image.shape[:2])

      classes = torch.tensor(classes, dtype=torch.int64)
      target.gt_classes = classes
      dataset_dict["instances"] = target
    return dataset_dict
