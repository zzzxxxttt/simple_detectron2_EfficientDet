import math
import logging
from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.layers import get_norm
from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.utils.logger import log_first_n
from detectron2.utils.events import get_event_storage

from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.postprocessing import detector_postprocess


@META_ARCH_REGISTRY.register()
class EfficientDetRetinaNet(RetinaNet):
  def __init__(self, cfg):
    super(RetinaNet, self).__init__()

    self.device = torch.device(cfg.MODEL.DEVICE)

    # fmt: off
    self.num_classes              = cfg.MODEL.RETINANET.NUM_CLASSES
    self.in_features              = cfg.MODEL.RETINANET.IN_FEATURES
    # Loss parameters:
    self.focal_loss_alpha         = cfg.MODEL.RETINANET.FOCAL_LOSS_ALPHA
    self.focal_loss_gamma         = cfg.MODEL.RETINANET.FOCAL_LOSS_GAMMA
    self.smooth_l1_loss_beta      = cfg.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA
    # Inference parameters:
    self.score_threshold          = cfg.MODEL.RETINANET.SCORE_THRESH_TEST
    self.topk_candidates          = cfg.MODEL.RETINANET.TOPK_CANDIDATES_TEST
    self.nms_threshold            = cfg.MODEL.RETINANET.NMS_THRESH_TEST
    self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
    # Vis parameters
    self.vis_period               = cfg.VIS_PERIOD
    self.input_format             = cfg.INPUT.FORMAT
    # fmt: on

    self.backbone = build_backbone(cfg)

    backbone_shape = self.backbone.output_shape()
    feature_shapes = [backbone_shape[f] for f in self.in_features]
    self.head = RetinaNetHead(cfg, feature_shapes)
    self.anchor_generator = build_anchor_generator(cfg, feature_shapes)

    # Matching and loss
    self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)
    self.matcher = Matcher(
      cfg.MODEL.RETINANET.IOU_THRESHOLDS,
      cfg.MODEL.RETINANET.IOU_LABELS,
      allow_low_quality_matches=True,
    )

    assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
    num_channels = len(cfg.MODEL.PIXEL_MEAN)
    pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
    pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
    self.normalizer = lambda x: (x - pixel_mean) / pixel_std
    self.to(self.device)

    """
    In Detectron1, loss is normalized by number of foreground samples in the batch.
    When batch size is 1 per GPU, #foreground has a large variance and
    using it lead to lower performance. Here we maintain an EMA of #foreground to
    stabilize the normalizer.
    """
    self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
    self.loss_normalizer_momentum = 0.9

  def forward(self, batched_inputs):
    """
    Args:
        batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
            Each item in the list contains the inputs for one image.
            For now, each item in the list is a dict that contains:

            * image: Tensor, image in (C, H, W) format.
            * instances: Instances

            Other information that's included in the original dicts, such as:

            * "height", "width" (int): the output resolution of the model, used in inference.
              See :meth:`postprocess` for details.
    Returns:
        dict[str: Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
    """
    images = self.preprocess_image(batched_inputs)
    if "instances" in batched_inputs[0]:
      gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
    elif "targets" in batched_inputs[0]:
      log_first_n(
        logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
      )
      gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
    else:
      gt_instances = None

    features = self.backbone(images.tensor)
    features = [features[f] for f in self.in_features]
    box_cls, box_delta = self.head(features)
    anchors = self.anchor_generator(features)

    if self.training:
      gt_classes, gt_anchors_reg_deltas = self.get_ground_truth(anchors, gt_instances)
      losses = self.losses(gt_classes, gt_anchors_reg_deltas, box_cls, box_delta)

      if self.vis_period > 0:
        storage = get_event_storage()
        if storage.iter % self.vis_period == 0:
          results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
          self.visualize_training(batched_inputs, results)

      return losses
    else:
      results = self.inference(box_cls, box_delta, anchors, images.image_sizes)
      processed_results = []
      for results_per_image, input_per_image, image_size in zip(
          results, batched_inputs, images.image_sizes
      ):
        offset_x = input_per_image.get("offset_x", 0)
        offset_y = input_per_image.get("offset_y", 0)
        real_w = input_per_image.get("real_w", image_size[1])
        real_h = input_per_image.get("real_h", image_size[0])

        results_per_image.pred_boxes.tensor[:, 0::2] -= offset_x
        results_per_image.pred_boxes.tensor[:, 1::2] -= offset_y
        results_per_image.pred_boxes.clip((real_h, real_w))

        results_per_image._image_size = (real_h, real_w)

        height = input_per_image.get("height", image_size[0])
        width = input_per_image.get("width", image_size[1])
        r = detector_postprocess(results_per_image, height, width)
        processed_results.append({"instances": r})
      return processed_results


class ResHead(nn.Module):
  def __init__(self, depthwise, pointwise, out_channels, norm):
    super().__init__()
    self.depthwise = depthwise
    self.pointwise = pointwise
    self.norm = get_norm(norm, out_channels) if norm != '' else nn.Sequential()

  def forward(self, x):
    out = self.depthwise(x)
    out = self.pointwise(out)
    out = self.norm(out)
    out = F.relu(out)
    return out + x


class RetinaNetHead(nn.Module):
  """
  The head used in RetinaNet for object classification and box regression.
  It has two subnets for the two tasks, with a common structure but separate parameters.
  """

  def __init__(self, cfg, input_shape: List[ShapeSpec]):
    super().__init__()
    # fmt: off
    in_channels       = input_shape[0].channels
    num_classes       = cfg.MODEL.RETINANET.NUM_CLASSES
    prior_prob        = cfg.MODEL.RETINANET.PRIOR_PROB
    num_anchors       = build_anchor_generator(cfg, input_shape).num_cell_anchors

    norm              = cfg.MODEL.RETINANET.NORM
    num_convs         = cfg.MODEL.RETINANET.NUM_CONVS
    in_features       = cfg.MODEL.RETINANET.IN_FEATURES
    # fmt: on

    assert (len(set(num_anchors)) == 1), \
      "Using different number of anchors between levels is not currently supported!"
    num_anchors = num_anchors[0]

    cls_depthwise_convs = []
    cls_pointwise_convs = []
    bbox_depthwise_convs = []
    bbox_pointwise_convs = []
    for _ in range(num_convs):
      cls_depthwise_convs.append(nn.Conv2d(in_channels, in_channels,
                                           kernel_size=3, stride=1, padding=1,
                                           groups=in_channels, bias=False))
      cls_pointwise_convs.append(nn.Conv2d(in_channels, in_channels,
                                           kernel_size=1, stride=1, padding=0,
                                           bias=norm == ''))
      bbox_depthwise_convs.append(nn.Conv2d(in_channels, in_channels,
                                            kernel_size=3, stride=1, padding=1,
                                            groups=in_channels, bias=False))
      bbox_pointwise_convs.append(nn.Conv2d(in_channels, in_channels,
                                            kernel_size=1, stride=1, padding=0,
                                            bias=norm == ''))

    self.cls_subnets = nn.ModuleList()
    self.bbox_subnets = nn.ModuleList()
    for _ in in_features:
      cls_subnet = []
      bbox_subnet = []
      for cls_depthwise, cls_pointwise, bbox_depthwise, bbox_pointwise in \
          zip(cls_depthwise_convs, cls_pointwise_convs, bbox_depthwise_convs, bbox_pointwise_convs):
        cls_subnet.append(ResHead(cls_depthwise, cls_pointwise, in_channels, norm))
        bbox_subnet.append(ResHead(bbox_depthwise, bbox_pointwise, in_channels, norm))
      self.cls_subnets.append(nn.Sequential(*cls_subnet))
      self.bbox_subnets.append(nn.Sequential(*bbox_subnet))

    self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes,
                               kernel_size=3, stride=1, padding=1)
    self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4,
                               kernel_size=3, stride=1, padding=1)

    # Initialization
    for modules in [self.cls_subnets, self.bbox_subnets, self.cls_score, self.bbox_pred]:
      for layer in modules.modules():
        if isinstance(layer, nn.Conv2d):
          torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
          if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)

    # Use prior in model initialization to improve stability
    bias_value = -math.log((1 - prior_prob) / prior_prob)
    torch.nn.init.constant_(self.cls_score.bias, bias_value)

  def forward(self, features):
    """
    Arguments:
        features (list[Tensor]): FPN feature map tensors in high to low resolution.
            Each tensor in the list correspond to different feature levels.

    Returns:
        logits (list[Tensor]): #lvl tensors, each has shape (N, AxK, Hi, Wi).
            The tensor predicts the classification probability
            at each spatial position for each of the A anchors and K object
            classes.
        bbox_reg (list[Tensor]): #lvl tensors, each has shape (N, Ax4, Hi, Wi).
            The tensor predicts 4-vector (dx,dy,dw,dh) box
            regression values for every anchor. These values are the
            relative offset between the anchor and the ground truth box.
    """
    logits = []
    bbox_reg = []
    for feature, cls_subnet, bbox_subnet in zip(features, self.cls_subnets, self.bbox_subnets):
      logits.append(self.cls_score(cls_subnet(feature)))
      bbox_reg.append(self.bbox_pred(bbox_subnet(feature)))
    return logits, bbox_reg
