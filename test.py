import os
import itertools
import weakref
from typing import Any, Dict, List, Set

import torch
import torch.optim as optim
from fvcore.nn.precise_bn import get_bn_modules


from core import DatasetMapper, add_config
from core.util.model_ema import add_model_ema_configs, may_build_model_ema, may_get_ema_checkpointer, EMAHook, \
    apply_model_ema_and_restore, EMADetectionCheckpointer
from core.pascal_voc import register_pascal_voc, inital_prompts
from core.pascal_voc_evaluation import PascalVOCDetectionEvaluator
from core.customyoloworld import CustomYoloWorld, load_ckpt
from core.eval_utils import Trainer

from mmengine.config import Config
from mmengine.runner import Runner
from torchvision.ops import nms, batched_nms
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup
from detectron2.config import get_cfg
import supervision as sv
import os.path as osp
import cv2


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
    parser0.add_argument("--ckpt", default="model.pth")
    args = parser0.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)

    data_register = Register('./datasets/', args.task, cfg)
    data_register.register_dataset()

    
    task_name = args.task.split('/')[0]
    split_name = args.task.split('/')[1]
    class_names = list(inital_prompts()[task_name])

    # model's config
    config_file = os.path.join("./configs", task_name,split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS


    class_names = class_names[:unknown_index]
    classnames = [class_names]

    

    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.eval()

    train_loader = Runner.build_dataloader(cfgY.trlder)
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    evaluator = Trainer.build_evaluator(cfg,"my_val")
    evaluator.reset()

    trainable = []
    model = CustomYoloWorld(runner.model,unknown_index)
    with torch.no_grad():
        model = load_ckpt(model, args.ckpt,cfg.TEST.PREV_INTRODUCED_CLS,cfg.TEST.CUR_INTRODUCED_CLS,eval=True)
        model = model.cuda()
        model.add_generic_text(class_names,generic_prompt='object',alpha=0.4)


    model.eval()

    for i in tqdm(test_loader):
        data_batch = model.parent.data_preprocessor(i)
        with torch.no_grad():
            outputs = model.predict(data_batch['inputs'],data_batch['data_samples'])
        preds = []

        for j in outputs:
            pred_instances = j.pred_instances
            pred_instances.ood_score = -pred_instances.cosinescores.max(dim=1).values

            for k in range(len(pred_instances.ood_score)):
                if pred_instances.ood_score[k] > cfgY.ood_threshold:
                    pred_instances.labels[k] = unknown_index

            keep_idxs = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=0.5)
            pred_instances = pred_instances[keep_idxs]
            preds.append(pred_instances)

        evaluator.process_mm(i['data_samples'],preds,unknown_index,use_ood_score = True)


    evaluator.evaluate()


