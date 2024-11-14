import torch
import torch.nn as nn
from torch import Tensor
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
import copy
from typing import List, Optional, Tuple, Union, Sequence
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from mmdet.structures import SampleList
from mmdet.utils import InstanceList
from mmdet.models.utils import (multi_apply, unpack_gt_instances,
                                filter_scores_and_topk)
from mmyolo.models.utils import gt_instances_preprocess
from mmengine.dist import get_dist_info
import torch.nn.functional as F



def load_ckpt(model, checkpoint_path, prev_classes, current_classes, eval = False):
    state_dict = torch.load(checkpoint_path)
    partial_state_dict = {k: v for k, v in state_dict.items() if not 'embeddings' in k}
    if eval:
        partial_state_dict = {k: v for k, v in state_dict.items() if not 'embeddings' in k}
        model.load_state_dict(partial_state_dict, strict=False)
        if state_dict.get('frozen_embeddings') is not None:
            model.frozen_embeddings = nn.Parameter(state_dict['frozen_embeddings'])
        else:
            model.frozen_embeddings = None
        if state_dict.get('embeddings') is not None:
            model.embeddings = nn.Parameter(state_dict['embeddings'])
        else:
            model.embeddings = None
        return model

    model.load_state_dict(partial_state_dict, strict=False)
    if prev_classes == 0:
        return model
    else:
        part_a = state_dict.get('frozen_embeddings')
        part_b = state_dict.get('embeddings')
        if part_a is not None and part_b is not None:
            freeze = torch.cat([part_a, part_b], dim=1)
        elif part_a is None:
            freeze = part_b
        else:
            freeze = part_a
        length = freeze.shape[1]
        model.frozen_embeddings = nn.Parameter(freeze)
        model.embeddings = nn.Parameter(model.text_feats[:, length:, :])
    return model

class ProjectionHead(nn.Module):
    def __init__(self, dim_in=[384,768,768], proj_dim=[384,768,768]):
        super(ProjectionHead, self).__init__()

        self.anchor0 = nn.Conv2d(proj_dim[0], 1, kernel_size=1, bias=False)
        self.anchor1 = nn.Conv2d(proj_dim[1], 1, kernel_size=1, bias=False)
        self.anchor2 = nn.Conv2d(proj_dim[2], 1, kernel_size=1, bias=False)


        self.proj0 = nn.Sequential(
            nn.Conv2d(dim_in[0], dim_in[0], kernel_size=1),
            nn.SyncBatchNorm(dim_in[0]),
            nn.ReLU(),
            nn.Conv2d(dim_in[0], proj_dim[0], kernel_size=1)
        )
        self.proj1 = nn.Sequential(
            nn.Conv2d(dim_in[1], dim_in[1], kernel_size=1),
            nn.SyncBatchNorm(dim_in[1]),
            nn.ReLU(),
            nn.Conv2d(dim_in[1], proj_dim[1], kernel_size=1)
        )
        self.proj2 = nn.Sequential(
            nn.Conv2d(dim_in[2], dim_in[2], kernel_size=1),
            nn.SyncBatchNorm(dim_in[2]),
            nn.ReLU(),
            nn.Conv2d(dim_in[2], proj_dim[2], kernel_size=1)
        )

    def forward(self, x):
        x0 = self.proj0(x[0])
        x0 = self.anchor0(x0)
        x1 = self.proj1(x[1])
        x1 = self.anchor1(x1)
        x2 = self.proj2(x[2])
        x2 = self.anchor2(x2)
        return x0, x1, x2
    
    
# Define the model
class CustomYoloWorld(nn.Module):
    def __init__(self, yolo_world_model,unknown_index,temperature=1):
        super(CustomYoloWorld, self).__init__()
        self.parent = yolo_world_model
        self.bbox_head = yolo_world_model.bbox_head
        self.unknown_index = unknown_index
        self.tmp_labels = None
        self.frozen_embeddings = None
        self.initalize_text_embedding()
        self.temperature = temperature

    def update_unknown_index(self, unknown_index):
        self.unknown_index = unknown_index

    def initalize_text_embedding(self):
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()
            self.embeddings = nn.Parameter(self.text_feats)
            self.projectors = nn.ModuleList([ProjectionHead() for i in range(self.unknown_index)])

    def add_generic_text(self,class_names,generic_prompt='object',alpha=0.05):
        if len(class_names) <= self.unknown_index:
            class_names.append(generic_prompt)
        classnames = [class_names]
        self.parent.reparameterize(classnames)
        with torch.no_grad():
            self.texts = copy.deepcopy(self.parent.texts)
            self.text_feats = self.parent.text_feats.clone()

            part_a = self.frozen_embeddings
            part_b = self.embeddings
            if part_a is not None and part_b is not None:
                freeze = torch.cat([part_a, part_b], dim=1)
            elif part_a is None:
                freeze = part_b
            else:
                freeze = part_a
                
            generic_embedding = self.text_feats[:, self.unknown_index, :]


            if alpha!=0:
                normalized_embedding = F.normalize(freeze, p=2, dim=2)
                normalized_embedding = normalized_embedding.mean(dim=1)
                generic_embedding = generic_embedding - alpha * normalized_embedding



            freeze = torch.cat([freeze, generic_embedding.unsqueeze(0)], dim=1)
            self.frozen_embeddings = nn.Parameter(freeze)
            self.embeddings = None
            #self.text_feats[:, :self.unknown_index, :] = self.frozen_embeddings
            #self.embeddings = nn.Parameter(self.text_feats)

    def head_loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples."""
        self.bbox_head.num_classes = self.text_feats[0].shape[0]
        self.bbox_head.assigner.num_classes = self.text_feats[0].shape[0]
        img_feats, txt_feats, flatten_scores_list = self.extract_feat(batch_inputs,
                                                 batch_data_samples)
        '''
        flatten_img_feats = [
            img_feat.permute(0, 2, 3, 1).reshape(len(batch_data_samples), -1,
                                                 self.bbox_head.num_classes)
            for img_feat in img_feats
        ]
        flatten_img_feats = torch.cat(flatten_img_feats, dim=1)
        '''
        self.tmp_labels = None
        head_losses = self.bbox_head_loss(img_feats, txt_feats, batch_data_samples)
        contrastive_loss = self.contrastive_loss(flatten_scores_list)
        return head_losses, contrastive_loss
    
    def bbox_head_loss(self, img_feats: Tuple[Tensor], txt_feats: Tensor,
             batch_data_samples: Union[list, dict]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network."""

        batch_gt_instances, batch_gt_instances_ignore, batch_img_metas = unpack_gt_instances(batch_data_samples)
        
        
        #self.bbox_head.training = True
        outs = self.bbox_head(img_feats, txt_feats)
        losses = self.dev_loss_by_feat(outs[0],outs[1],outs[2],batch_gt_instances, batch_img_metas)
        return losses


    def extract_feat(
            self, batch_inputs: Tensor,
            batch_data_samples: SampleList) -> Tuple[Tuple[Tensor], Tensor, Tensor]:
        """Extract features."""
        if self.frozen_embeddings != None and self.embeddings != None:
            txt_feats = torch.cat([self.frozen_embeddings, self.embeddings], dim=1)
        elif self.embeddings != None:
            txt_feats = self.embeddings
        else:
            txt_feats = self.frozen_embeddings

        img_feats = self.parent.backbone.forward_image(batch_inputs)
        txt_feats = txt_feats.repeat(img_feats[0].shape[0], 1, 1)   

        if self.parent.with_neck:
            if self.parent.mm_neck:
                img_feats = self.parent.neck(img_feats, txt_feats)
            else:
                img_feats = self.parent.neck(img_feats)

        flatten_scores_list = []
        for projector in self.projectors:
            scores = projector(img_feats)
            flatten_scores = [
            score.permute(0, 2, 3, 1).reshape(img_feats[0].shape[0], -1)
            for score in scores
            ]
            flatten_scores = torch.cat(flatten_scores, dim=1)
            flatten_scores_list.append(flatten_scores)

        flatten_scores_list = torch.stack(flatten_scores_list, dim=2)    

        return img_feats, txt_feats, flatten_scores_list
    
    def enable_projector_grad(self,index):
        for i in range(index, self.unknown_index):
            for param in self.projectors[i].parameters():
                param.requires_grad = True

    
    def contrastive_loss(self,
            flatten_scores_list: Tensor) -> Tensor:
        """Calculate the contrastive loss."""
        #mask = torch.zeros_like(flatten_scores_list)
        #mask = mask.to(torch.bool)
        #mask.scatter_(2, self.tmp_labels.unsqueeze(2), 1)
        b, n, c = flatten_scores_list.shape
        flatten_scores_list = torch.div(flatten_scores_list,
                                        self.temperature)
        contrastive_losses = []
        for i in range(c):
            mask = (self.tmp_labels == i)
            positive = flatten_scores_list[:,:,i][mask]
            if not positive.numel() > 0:
                continue
            negative = flatten_scores_list[:,:,i][~mask]
            positive_exp = torch.exp(positive)
            negative = torch.exp(negative)
            log_prob = positive - torch.log(positive_exp.sum() + negative.sum())
            contrastive_losses.append(-log_prob.sum()/mask.sum())

        return sum(contrastive_losses) / len(contrastive_losses)


    def dev_loss_by_feat(
            self,
            cls_scores: Sequence[Tensor],
            bbox_preds: Sequence[Tensor],
            bbox_dist_preds: Sequence[Tensor],
            batch_gt_instances: Sequence[InstanceData],
            batch_img_metas: Sequence[dict]) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (Sequence[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (Sequence[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            bbox_dist_preds (Sequence[Tensor]): Box distribution logits for
                each scale level with shape (bs, reg_max + 1, H*W, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.
        Returns:
            dict[str, Tensor]: A dictionary of losses.
        """
        num_imgs = len(batch_img_metas)

        current_featmap_sizes = [
            cls_score.shape[2:] for cls_score in cls_scores
        ]
        # If the shape does not equal, generate new one
        if current_featmap_sizes != self.bbox_head.featmap_sizes_train:
            self.bbox_head.featmap_sizes_train = current_featmap_sizes

            mlvl_priors_with_stride = self.bbox_head.prior_generator.grid_priors(
                self.bbox_head.featmap_sizes_train,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device,
                with_stride=True)

            self.bbox_head.num_level_priors = [len(n) for n in mlvl_priors_with_stride]
            self.bbox_head.flatten_priors_train = torch.cat(mlvl_priors_with_stride,
                                                  dim=0)
            self.bbox_head.stride_tensor = self.bbox_head.flatten_priors_train[..., [2]]

        # gt info
        gt_info = gt_instances_preprocess(batch_gt_instances, num_imgs)
        gt_labels = gt_info[:, :, :1]
        gt_bboxes = gt_info[:, :, 1:]  # xyxy
        pad_bbox_flag = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pred info
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.bbox_head.num_classes)
            for cls_pred in cls_scores
        ]
        flatten_pred_bboxes = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        # (bs, n, 4 * reg_max)
        flatten_pred_dists = [
            bbox_pred_org.reshape(num_imgs, -1, self.bbox_head.head_module.reg_max * 4)
            for bbox_pred_org in bbox_dist_preds
        ]

        flatten_dist_preds = torch.cat(flatten_pred_dists, dim=1)
        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_pred_bboxes = torch.cat(flatten_pred_bboxes, dim=1)
        flatten_pred_bboxes = self.bbox_head.bbox_coder.decode(
            self.bbox_head.flatten_priors_train[..., :2], flatten_pred_bboxes,
            self.bbox_head.stride_tensor[..., 0])

        assigned_result = self.bbox_head.assigner(
            (flatten_pred_bboxes.detach()).type(gt_bboxes.dtype),
            flatten_cls_preds.detach().sigmoid(), self.bbox_head.flatten_priors_train,
            gt_labels, gt_bboxes, pad_bbox_flag)

        assigned_bboxes = assigned_result['assigned_bboxes']
        assigned_scores = assigned_result['assigned_scores']
        fg_mask_pre_prior = assigned_result['fg_mask_pre_prior']


        # gt labels
        max_values, max_indices = assigned_scores.max(dim=2)
        max_indices[max_values <= 0] = -1
        self.tmp_labels = max_indices

        assigned_scores_sum = assigned_scores.sum().clamp(min=1)

        loss_cls = self.bbox_head.loss_cls(flatten_cls_preds, assigned_scores).sum()
        loss_cls /= assigned_scores_sum

        # rescale bbox
        assigned_bboxes /= self.bbox_head.stride_tensor
        flatten_pred_bboxes /= self.bbox_head.stride_tensor

        # select positive samples mask
        num_pos = fg_mask_pre_prior.sum()
        if num_pos > 0:
            # when num_pos > 0, assigned_scores_sum will >0, so the loss_bbox
            # will not report an error
            # iou loss
            prior_bbox_mask = fg_mask_pre_prior.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(
                flatten_pred_bboxes, prior_bbox_mask).reshape([-1, 4])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, prior_bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(assigned_scores.sum(-1),
                                              fg_mask_pre_prior).unsqueeze(-1)
            loss_bbox = self.bbox_head.loss_bbox(
                pred_bboxes_pos, assigned_bboxes_pos,
                weight=bbox_weight) / assigned_scores_sum

            # dfl loss
            pred_dist_pos = flatten_dist_preds[fg_mask_pre_prior]
            assigned_ltrb = self.bbox_head.bbox_coder.encode(
                self.bbox_head.flatten_priors_train[..., :2] / self.bbox_head.stride_tensor,
                assigned_bboxes,
                max_dis=self.bbox_head.head_module.reg_max - 1,
                eps=0.01)
            assigned_ltrb_pos = torch.masked_select(
                assigned_ltrb, prior_bbox_mask).reshape([-1, 4])
            loss_dfl = self.bbox_head.loss_dfl(pred_dist_pos.reshape(
                -1, self.bbox_head.head_module.reg_max),
                                     assigned_ltrb_pos.reshape(-1),
                                     weight=bbox_weight.expand(-1,
                                                               4).reshape(-1),
                                     avg_factor=assigned_scores_sum)
        else:
            loss_bbox = flatten_pred_bboxes.sum() * 0
            loss_dfl = flatten_pred_bboxes.sum() * 0
        if self.bbox_head.world_size == -1:
            _, world_size = get_dist_info()
        else:
            world_size = self.bbox_head.world_size
        return dict(loss_cls=loss_cls * num_imgs * world_size,
                    loss_bbox=loss_bbox * num_imgs * world_size,
                    loss_dfl=loss_dfl * num_imgs * world_size)


    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """

        img_feats, txt_feats, flatten_scores_list = self.extract_feat(batch_inputs,
                                                 batch_data_samples)

        # self.bbox_head.num_classes = self.num_test_classes
        self.parent.bbox_head.num_classes = txt_feats[0].shape[0]
        results_list = self.bbox_head_pred(img_feats,
                                              txt_feats,
                                              batch_data_samples,
                                              flatten_scores_list,
                                              rescale=rescale)
        batch_data_samples = self.parent.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples
    
    
    def bbox_head_pred(self, img_feats: Tuple[Tensor],
                txt_feats: Tensor,
                batch_data_samples: SampleList,
                flatten_scores_list: Tensor,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.
        """
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self.bbox_head(img_feats, txt_feats)
        predictions = self.bbox_head_predict_by_feat(*outs,
                                                     flatten_scores_list=flatten_scores_list,
                                           batch_img_metas=batch_img_metas,
                                           rescale=rescale)
        return predictions

    def bbox_head_predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        objectnesses: Optional[List[Tensor]] = None,
                        batch_img_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        flatten_scores_list: Optional[Tensor] = None,
                        rescale: bool = True,
                        with_nms: bool = True) -> List[InstanceData]:
        """Transform a batch of output features extracted by the head into
        bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            batch_img_metas (list[dict], Optional): Batch image meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.
            with_nms (bool): If True, do nms before return boxes.
                Defaults to True.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert len(cls_scores) == len(bbox_preds)
        if objectnesses is None:
            with_objectnesses = False
        else:
            with_objectnesses = True
            assert len(cls_scores) == len(objectnesses)

        cfg = self.bbox_head.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        multi_label = cfg.multi_label
        multi_label &= self.bbox_head.num_classes > 1
        cfg.multi_label = multi_label

        num_imgs = len(batch_img_metas)
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        # If the shape does not change, use the previous mlvl_priors
        if featmap_sizes != self.bbox_head.featmap_sizes:
            self.bbox_head.mlvl_priors = self.bbox_head.prior_generator.grid_priors(
                featmap_sizes,
                dtype=cls_scores[0].dtype,
                device=cls_scores[0].device)
            self.bbox_head.featmap_sizes = featmap_sizes
        flatten_priors = torch.cat(self.bbox_head.mlvl_priors)

        mlvl_strides = [
            flatten_priors.new_full(
                (featmap_size.numel() * self.bbox_head.num_base_priors, ), stride) for
            featmap_size, stride in zip(featmap_sizes, self.bbox_head.featmap_strides)
        ]
        flatten_stride = torch.cat(mlvl_strides)

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                  self.bbox_head.num_classes)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]

        logits_batch = torch.cat(flatten_cls_scores, dim=1)

        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_decoded_bboxes = self.bbox_head.bbox_coder.decode(
            flatten_priors[None], flatten_bbox_preds, flatten_stride)

        if with_objectnesses:
            flatten_objectness = [
                objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
                for objectness in objectnesses
            ]
            flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()
        else:
            flatten_objectness = [None for _ in range(num_imgs)]


        if flatten_scores_list is None:
            flatten_scores_list = logits_batch
        
        # 8400
        # print(flatten_cls_scores.shape)
        results_list = []
        for (bboxes, scores, objectness,
             img_meta, logits, cosine) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                              flatten_objectness, batch_img_metas, logits_batch, flatten_scores_list):
            ori_shape = img_meta['ori_shape']
            scale_factor = img_meta['scale_factor']
            if 'pad_param' in img_meta:
                pad_param = img_meta['pad_param']
            else:
                pad_param = None

            score_thr = cfg.get('score_thr', -1)
            # yolox_style does not require the following operations
            if objectness is not None and score_thr > 0 and not cfg.get(
                    'yolox_style', False):
                conf_inds = objectness > score_thr
                bboxes = bboxes[conf_inds, :]
                scores = scores[conf_inds, :]
                objectness = objectness[conf_inds]

            if objectness is not None:
                # conf = obj_conf * cls_conf
                scores *= objectness[:, None]

            if scores.shape[0] == 0:
                empty_results = InstanceData()
                empty_results.bboxes = bboxes
                empty_results.scores = scores[:, 0]
                empty_results.labels = scores[:, 0].int()
                results_list.append(empty_results)
                continue

            nms_pre = cfg.get('nms_pre', 100000)

            if cfg.multi_label is False:
                scores, labels = scores.max(1, keepdim=True)
                scores, _, keep_idxs, results = filter_scores_and_topk(
                    scores,
                    score_thr,
                    nms_pre,
                    results=dict(labels=labels[:, 0]))
                labels = results['labels']
            else:
                scores, labels, keep_idxs, _ = filter_scores_and_topk(
                    scores, score_thr, nms_pre)

            '''
            Yolo world tends to produce low confidence score 
            (even if the detection is good)
            We rescale the score to make it easier to select threshold 
            (though it is unnecessary for threshold free evaluation)
            A monotonically increasing function 
            do not affect evaluation results
            scores = 1 - torch.exp(-5*scores)
            '''
            results = InstanceData(scores=scores,
                        labels=labels,
                        cosinescores=cosine[keep_idxs],
                        logits=logits[keep_idxs],
                        bboxes=bboxes[keep_idxs])

            if rescale:
                if pad_param is not None:
                    results.bboxes -= results.bboxes.new_tensor([
                        pad_param[2], pad_param[0], pad_param[2], pad_param[0]
                    ])
                results.bboxes /= results.bboxes.new_tensor(
                    scale_factor).repeat((1, 2))

            if cfg.get('yolox_style', False):
                # do not need max_per_img
                cfg.max_per_img = len(results)

            results = self.bbox_head._bbox_post_process(results=results,
                                              cfg=cfg,
                                              rescale=False,
                                              with_nms=with_nms,
                                              img_meta=img_meta)
            results.bboxes[:, 0::2].clamp_(0, ori_shape[1])
            results.bboxes[:, 1::2].clamp_(0, ori_shape[0])

            results_list.append(results)
        return results_list