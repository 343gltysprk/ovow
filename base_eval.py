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


class Trainer(DefaultTrainer):
    """ Extension of the Trainer class adapted """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()  # call grandfather's `__init__` while avoid father's `__init()`
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # EMA
        kwargs = {
            'trainer': weakref.proxy(self),
        }
        kwargs.update(may_get_ema_checkpointer(cfg, model))
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            **kwargs,
            # trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        # setup EMA
        may_build_model_ema(cfg, model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        if 'lvis' in dataset_name:
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        else:
            return PascalVOCDetectionEvaluator(dataset_name, cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_optimizer(cls, cfg, model):
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for key, value in model.named_parameters(recurse=True):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY
            if "backbone" in key:
                lr = lr * cfg.SOLVER.BACKBONE_MULTIPLIER
            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def ema_test(cls, cfg, model, evaluators=None):
        # model with ema weights
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL_EMA.ENABLED:
            logger.info("Run evaluation with EMA.")
            with apply_model_ema_and_restore(model):
                results = cls.test(cfg, model, evaluators=evaluators)
        else:
            results = cls.test(cfg, model, evaluators=evaluators)
        return results

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            EMAHook(self.cfg, self.model) if cfg.MODEL_EMA.ENABLED else None,  # EMA hook
            hooks.LRScheduler(),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(hooks.PeriodicCheckpointer(self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD))

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        # Do evaluation after checkpointer, because then if it fails,
        # we can use the saved checkpoint to debug.
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            # Here the default print/log frequency of each writer is used.
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret


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
    #nunames = list(inital_prompts()[task_name])
    #nunames = nunames[:unknown_index]
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
            score_filter = (pred_instances.scores.float() > 0.05) | (pred_instances.labels == unknown_index)
            pred_instances = pred_instances[score_filter]
            preds.append(pred_instances)
        
        evaluator.process_mm(i['data_samples'],preds,unknown_index)

    evaluator.evaluate()




