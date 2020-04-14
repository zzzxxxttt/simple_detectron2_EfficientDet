import os

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import BACKBONE_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.layers import ShapeSpec, FrozenBatchNorm2d, get_norm
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool

from utils import round_filters, round_repeats, drop_connect, \
    get_same_padding_conv2d, Swish, MemoryEfficientSwish, decode, GlobalParams

params_dict = \
    {
        # Coefficients:   width,depth,res,dropout
        'model-b0': (1.0, 1.0, 224, 0.2),
        'model-b1': (1.0, 1.1, 240, 0.2),
        'model-b2': (1.1, 1.2, 260, 0.3),
        'model-b3': (1.2, 1.4, 300, 0.3),
        'model-b4': (1.4, 1.8, 380, 0.4),
        'model-b5': (1.6, 2.2, 456, 0.4),
        'model-b6': (1.8, 2.6, 528, 0.5),
        'model-b7': (2.0, 3.1, 600, 0.5),
    }


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, norm):
        super().__init__()
        self.in_channels = block_args.input_filters
        self.out_channels = block_args.output_filters
        self.stride = block_args.stride if isinstance(block_args.stride, int) else block_args[0]

        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=norm == '')
            # self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._bn0 = get_norm(norm, oup)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(oup, oup, groups=oup, kernel_size=k, stride=s, bias=norm == '')
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn1 = get_norm(norm, oup)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(oup, final_oup, kernel_size=1, bias=norm == '')
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._bn2 = get_norm(norm, final_oup)
        self._swish = MemoryEfficientSwish()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        out = inputs
        if self._block_args.expand_ratio != 1:
            out = self._swish(self._bn0(self._expand_conv(inputs)))
        out = self._swish(self._bn1(self._depthwise_conv(out)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(out, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            out = torch.sigmoid(x_squeezed) * out

        out = self._bn2(self._project_conv(out))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                out = drop_connect(out, p=drop_connect_rate, training=self.training)
            out = out + inputs  # skip connection
        return out

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(Backbone):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('model-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, norm='', model_version=4):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self.g_cfg = global_params
        self.b_cfgs = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self.g_cfg.batch_norm_momentum
        bn_eps = self.g_cfg.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self.g_cfg)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        # self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn0 = get_norm(norm, out_channels)

        # Build blocks
        self._blocks = nn.ModuleList([])

        idx = 0
        out_inds = []
        out_channels = []
        for b_cfg in self.b_cfgs:
            # Update block input and output filters based on depth multiplier.
            b_cfg = b_cfg._replace(input_filters=round_filters(b_cfg.input_filters, self.g_cfg),
                                   output_filters=round_filters(b_cfg.output_filters, self.g_cfg),
                                   num_repeat=round_repeats(b_cfg.num_repeat, self.g_cfg))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(b_cfg, self.g_cfg, norm))
            print(b_cfg.stride)
            if b_cfg.stride[0] != 1:
                out_inds.append(idx - 1)
                out_channels.append(b_cfg.input_filters)
            idx += 1
            if b_cfg.num_repeat > 1:
                b_cfg = b_cfg._replace(input_filters=b_cfg.output_filters, stride=1)
            for _ in range(b_cfg.num_repeat - 1):
                self._blocks.append(MBConvBlock(b_cfg, self.g_cfg, norm))
                idx += 1
        out_inds.append(idx - 1)
        out_channels.append(b_cfg.output_filters)
        self.out_block_inds_all_stage = out_inds
        self.out_block_inds = out_inds[1:]
        self._out_feature_channels = out_channels[1:]
        # Head
        # in_channels = block_args.output_filters  # output of final block
        # out_channels = round_filters(1280, self._global_params)
        # self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        #
        # # Final linear layer
        # self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self._dropout = nn.Dropout(self._global_params.dropout_rate)
        # self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def freeze_at(self, stage):
        stage = min(stage, len(self.out_block_inds_all_stage))
        if stage < 0:
            return  # skip freeze, used when train from scratch
        # stage == 0: freeze stem
        for p in self._conv_stem.parameters():
            p.requires_grad = False
        FrozenBatchNorm2d.convert_frozen_batchnorm(self._bn0)
        # stage >= 1: freeze blocks
        if stage >= 1:
            block_idx = self.out_block_inds_all_stage[stage - 1]
            for i, block in enumerate(self._blocks):
                if i > block_idx:
                    break
                block.freeze()

    def forward(self, x):
        # Stem
        out = self._swish(self._bn0(self._conv_stem(x)))
        features = []
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.g_cfg.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            out = block(out, drop_connect_rate=drop_connect_rate)
            if idx in self.out_block_inds:
                features.append(out)
        outputs = {f"stride-{2 ** i}": x for i, x in enumerate(features, 2)}
        return outputs

    def output_shape(self):
        return {f"stride-{2 ** i}":
                    ShapeSpec(channels=self._out_feature_channels[i - 2], stride=2 ** i)
                for i, _ in enumerate(self.out_block_inds, 2)}

    @property
    def out_feature_strides(self):
        """deprecated"""
        return {f: v.stride for f, v in self.output_shape().items()}

    @property
    def out_feature_channels(self):
        """deprecated"""
        return {f: v.channels for f, v in self.output_shape().items()}


@BACKBONE_REGISTRY.register()
def build_efficientnet_backbone(cfg, shape):
    version = cfg.MODEL.EfficientNet.VERSION
    assert isinstance(version, int) and 0 <= version <= 7
    w, d, s, p = params_dict["model-b%d" % version]
    # note: all models have drop connect rate = 0.2, really ?
    blocks_args = decode(['r1_k3_s11_e1_i32_o16_se0.25',
                          'r2_k3_s22_e6_i16_o24_se0.25',
                          'r2_k5_s22_e6_i24_o40_se0.25',
                          'r3_k3_s22_e6_i40_o80_se0.25',
                          'r3_k5_s11_e6_i80_o112_se0.25',
                          'r4_k5_s22_e6_i112_o192_se0.25',
                          'r1_k3_s11_e6_i192_o320_se0.25'])

    global_params = GlobalParams(batch_norm_momentum=0.99,
                                 batch_norm_epsilon=1e-3,
                                 dropout_rate=p,
                                 drop_connect_rate=0.2,
                                 num_classes=1000,
                                 width_coefficient=w,
                                 depth_coefficient=d,
                                 depth_divisor=8,
                                 min_depth=None,
                                 image_size=s)

    model = EfficientNet(blocks_args, global_params,
                         norm=cfg.MODEL.EfficientNet.NORM, model_version=version)
    if cfg.MODEL.WEIGHTS and os.path.exists(cfg.MODEL.WEIGHTS):
        state_dict = torch.load(cfg.MODEL.WEIGHTS, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
        model.freeze_at(cfg.MODEL.EfficientNet.FREEZE_AT)
    return model


@BACKBONE_REGISTRY.register()
def build_efficientnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_efficientnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(bottom_up=bottom_up,
                   in_features=in_features,
                   out_channels=out_channels,
                   norm=cfg.MODEL.FPN.NORM,
                   top_block=LastLevelMaxPool(),
                   fuse_type=cfg.MODEL.FPN.FUSE_TYPE)
    return backbone


if __name__ == '__main__':
    class cfg:
        def __init__(self):
            class model:
                def __init__(self):
                    class efficientnet:
                        def __init__(self):
                            self.VERSION = 1
                            self.NORM = 'SyncBN'
                            self.FREEZE_AT = 2

                    self.EfficientNet = efficientnet()
                    self.WEIGHTS = None

            self.MODEL = model()


    CFG = cfg()
    model = build_efficientnet_backbone(CFG, None)


    def hook(self, input, output):
        print(output.data.cpu().numpy().shape)
        # pass


    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.register_forward_hook(hook)

    with torch.no_grad():
        y = model(torch.randn(1, 3, 512, 512))
    pass
