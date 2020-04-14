__version__ = "0.5.1"
from .backbone import EfficientNet
from .utils import GlobalParams, BlockArgs

from .config import add_efficientnet_config
from .backbone import build_efficientnet_backbone, build_efficientnet_fpn_backbone
from .bifpn import build_retinanet_resnet_bifpn_backbone, build_retinanet_efficientnet_bifpn_backbone
from .retinanet import EfficientDetRetinaNet