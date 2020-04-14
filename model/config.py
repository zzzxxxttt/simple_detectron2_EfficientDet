from detectron2.config import CfgNode as CN


def add_efficientnet_config(cfg):
    _C = cfg
    _C.MODEL.EfficientNet = CN()
    _C.MODEL.EfficientNet.VERSION = 0
    _C.MODEL.EfficientNet.NORM = 'SyncBN'
    _C.MODEL.EfficientNet.FREEZE_AT = 2

    _C.MODEL.FPN.IN_FEATURE_P6P7 = 'res5'
    _C.MODEL.FPN.REPEAT = 3

    _C.MODEL.RETINANET.NORM = 'SyncBN'

    _C.INPUT.OUTPUT_SIZE = 512
    _C.INPUT.RAND_CROP_RANGE = [0.1, 2.0]
