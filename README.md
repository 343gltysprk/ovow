# From Open Vocabulary to Open World

<p align="center">
    <img src="assets/main.png" alt="main" width=60%>
</p>

## Environment
- Python 3.11.9 toch 2.3.1 CUDA 12.2
- Install [Yolo World](https://github.com/AILab-CVC/YOLO-World)
  - Requires: mmcv, mmcv-lite, mmdet, mmengine, mmyolo, numpy, opencv-python, openmim, supervision, tokenizers, torch, torchvision, transformers, wheel
- Prepare datasets:
  - Download [COCO](https://cocodataset.org/#download) and [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/).
  - Convert annotation format using `coco_to_voc.py`.
  - Move all images to `datasets/JPEGImages` and annotations to `datasets/Annotations`.

## Getting Started
- Training open world object detector:
  ```
  sh train.sh
  ```
- To evaluate the model:
  ```
  sh test_owod.sh
  ```
- Model training starts from pretrained [Yolo World checkpoint](https://huggingface.co/wondervictor/YOLO-World/blob/main/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth)
