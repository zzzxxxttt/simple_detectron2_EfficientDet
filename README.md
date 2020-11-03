# A simple Detectron2 EfficientDet (working in progress)

**update 2020/11/3: There are still some hidden bugs in the code, but due to lack of GPU, I can't continue the experiments** 

This repository is a simple detectron2 based implementation of [EfficientDet](https://arxiv.org/pdf/1911.09070.pdf)

+ The backbone part is ported from [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
+ The BiFPN implementation is based on the [official implementation](https://github.com/google/automl/tree/master/efficientdet)
+ The detection framework is based on [Detectron2](https://github.com/facebookresearch/detectron2)

## Requirements:
- python>=3.5
- detectron2

## Getting Started
1. [Install detectron2](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)

2. Download [COCO dataset](http://cocodataset.org/#download) and put ```annotations```, ```train2017```, ```val2017``` (or create symlinks) into ```DETECTRON2_PATH/datasets/coco```

3. Clone this repo:
    ```
    git clone https://github.com/zzzxxxttt/simple_detectron2_efficientdet /path/to/efficientdet
    ```

4. Download the [pretrained EfficientNet weights](https://github.com/lukemelas/EfficientNet-PyTorch). For example, you downloaded the EfficientNet-B0 weights and name it as b0.pth, run the following codes in python console:
   
   ```python
   >>> import torch
   >>> ckpt = torch.load('b0.pth', map_location = 'cpu')
   >>> ckpt = {'model': ckpt, 'matching_heuristics': True}
   >>> torch.save(ckpt, 'b0_detectron2.pth')
   ```
   
5. Start training: 
    ```
    python train_net.py --config-file configs/EfficientDet_D0.yaml MODEL.WEIGHTS /path/to/checkpoint_file
    ```

## Results

|      Model      | mAP (val, 100 epochs) | mAP (val, 300 epochs) | paper mAP (val, 300 epochs) |
| :-------------: | :-----------------: | :------------------: | :-----------------------------: |
| EfficientDet-D0 |        31.9%        |        32.4%         |              33.5%              |

