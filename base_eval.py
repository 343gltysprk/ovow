import os
import itertools
import weakref
from typing import Any, Dict, List, Set
import logging
from collections import OrderedDict

import torch
from fvcore.nn.precise_bn import get_bn_modules

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, create_ddp_model, \
    AMPTrainer, SimpleTrainer, hooks
from detectron2.evaluation import COCOEvaluator, LVISEvaluator, verify_results
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.modeling import build_model

from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from core.eval_utils import Trainer

import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.dataset import BaseDataset
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmyolo.registry import RUNNERS
from torchvision.ops import nms
import PIL.Image
import cv2
import supervision as sv
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms
import matplotlib.pyplot as plt
import supervision as sv
import os.path as osp
from tqdm import tqdm


class Register:
    def __init__(self, dataset_root, split, cfg):
        self.dataset_root = dataset_root
        self.super_split = split.split('/')[0]
        self.cfg = cfg

        self.PREDEFINED_SPLITS_DATASET = {
            "my_train": split,
            "my_val": os.path.join(self.super_split, 'test')
        }

    def register_dataset(self):
        """
        purpose: register all splits of datasets with PREDEFINED_SPLITS_DATASET
        """
        for name, split in self.PREDEFINED_SPLITS_DATASET.items():
            register_pascal_voc(name, self.dataset_root, self.super_split, split, self.cfg)






def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg




if __name__ == "__main__":
    parser0 = default_argument_parser()
    parser0.add_argument("--task", default="")
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    data_register = Register('./datasets/', args.task, cfg)
    data_register.register_dataset()

    
    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    if task_name == "nu-OWODB":
        class_names = list(inital_prompts()['nu-prompt'])
    else:
        class_names = list(inital_prompts()[task_name])

    # yolo world load config
    config_file = os.path.join("./configs", task_name,split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    cfgY.load_from = "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain-5daf1395.pth"
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS
    class_names = class_names[:unknown_index]
    class_names.append('object')
    class_names = [class_names]
    

    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(class_names)
    runner.model.eval()
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    evaluator = Trainer.build_evaluator(cfg,"my_val")
    evaluator.reset()


    for i in tqdm(test_loader):
        data_batch = runner.model.data_preprocessor(i)
        with torch.no_grad():
            outputs = runner.model.predict(data_batch['inputs'],data_batch['data_samples'])
        preds = []

        for j in outputs:
            pred_instances = j.pred_instances
            # nms
            keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=0.5)
            pred_instances = pred_instances[keep_idxs]
            preds.append(pred_instances)
        
        evaluator.process_mm(i['data_samples'],preds,unknown_index)

    evaluator.evaluate()




