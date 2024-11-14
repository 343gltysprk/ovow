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




# Function to save the model
def save_model(model, epoch, save_dir='checkpoints', file_name="model"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f"{file_name}_{epoch}.pth")
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


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
    if task_name == "nu-OWODB":
        class_names = list(inital_prompts()['nu-prompt'])
    else:
        class_names = list(inital_prompts()[task_name])

    # model's config
    config_file = os.path.join("./configs", task_name, split_name + ".py")
    cfgY = Config.fromfile(config_file)
    cfgY.work_dir = "."
    if cfg.TEST.PREV_INTRODUCED_CLS == 0:
        cfgY.load_from = args.ckpt
    
    unknown_index = cfg.TEST.PREV_INTRODUCED_CLS + cfg.TEST.CUR_INTRODUCED_CLS


    class_names = class_names[:unknown_index]
    classnames = [class_names]

    

    runner = Runner.from_cfg(cfgY)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.model.reparameterize(classnames)
    runner.model.train()

    train_loader = Runner.build_dataloader(cfgY.trlder)
    test_loader = Runner.build_dataloader(cfgY.test_dataloader)

    evaluator = Trainer.build_evaluator(cfg,"my_val")
    evaluator.reset()

    trainable = ['embeddings']
    model = CustomYoloWorld(runner.model,unknown_index)
    with torch.no_grad():
        model = load_ckpt(model, args.ckpt,cfg.TEST.PREV_INTRODUCED_CLS,cfg.TEST.CUR_INTRODUCED_CLS)
    model = model.cuda()
    for name, param in model.named_parameters():
        #if name in trainable or 'projectors' in name:
        if name in trainable:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)
    model.enable_projector_grad(cfg.TEST.PREV_INTRODUCED_CLS)
    optimizer = optim.AdamW(model.parameters(), lr=cfgY.base_lr, weight_decay=cfgY.weight_decay)

    model.train()
    for epoch in range(cfgY.max_epochs):
        print(f"Epoch: {epoch}")
        step = 0
        for i in train_loader:
            optimizer.zero_grad()
            data_batch = model.parent.data_preprocessor(i)
            loss1,loss2 = model.head_loss(data_batch['inputs'],data_batch['data_samples'])
            loss = loss1['loss_cls'] + loss1['loss_dfl'] + loss1['loss_bbox'] + loss2
            loss.backward()
            if step%20 == 0:
                print('cls loss: ', loss1['loss_cls'].item(), 'dfl loss: ', loss1['loss_dfl'].item(),'bbox loss: ', loss1['loss_bbox'].item(),'contrastive loss: ', loss2.item())
            optimizer.step()
            step+=1
        
        if epoch%5 == 0:
            save_model(model, epoch, save_dir=args.task)
        #each epoch save the latest model
        save_model(model, 'latest', save_dir=args.task)
    #save the final model
    save_model(model, 'final', save_dir=args.task)

