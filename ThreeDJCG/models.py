from dataclasses import dataclass
from pathlib import Path

import os
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.scannet.model_util_scannet import ScannetDatasetConfig
from scripts.joint_scripts.caption_eval import get_model, get_eval_data, get_dataloader

# constants
DC = ScannetDatasetConfig()


@dataclass
class Args:
    cfg_file: str
    opts: str


class ThreeDJCG(nn.Module):
    def __init__(self):
        super().__init__()

        self.set_arg()
        # setting
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(self.args.gpu)
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # reproducibility
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(self.args.seed)

        # get eval data
        scanrefer_eval, eval_scene_list, scanrefer_eval_new = get_eval_data(self.args)

        # get dataloader
        dataset, dataloader = get_dataloader(self.args, scanrefer_eval, scanrefer_eval_new, eval_scene_list, DC)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.args, dataset, device)

    # def run_train(self):
    #     optimizer = optim.construct_optimizer(
    #         self.model, self.cfg
    #     )

    def set_arg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--folder", type=str, help="Folder containing the model")
        parser.add_argument("--dataset", type=str, help="Choose a dataset: ScanRefer or ReferIt3D", default="ScanRefer")
        parser.add_argument("--gpu", type=str, help="gpu", default="0")
        # parser.add_argument("--gpu", type=str, help="gpu", default=["0"], nargs="+")
        parser.add_argument("--batch_size", type=int, help="batch size", default=8)
        parser.add_argument("--seed", type=int, default=42, help="random seed")
        
        parser.add_argument("--lang_num_max", type=int, help="lang num max", default=8)
        parser.add_argument("--num_points", type=int, default=40000, help="Point Number [default: 40000]")
        parser.add_argument("--num_proposals", type=int, default=256, help="Proposal number [default: 256]")
        parser.add_argument("--num_scenes", type=int, default=-1, help="Number of scenes [default: -1]")
        parser.add_argument("--num_locals", type=int, default=-1, help="Number of local objects [default: -1]")
        parser.add_argument("--num_graph_steps", type=int, default=0, help="Number of graph conv layer [default: 0]")
        
        parser.add_argument("--query_mode", type=str, default="corner", help="Mode for querying the local context, [choices: center, corner]")
        parser.add_argument("--graph_mode", type=str, default="edge_conv", help="Mode for querying the local context, [choices: graph_conv, edge_conv]")
        parser.add_argument("--graph_aggr", type=str, default="add", help="Mode for aggregating features, [choices: add, mean, max]")
        
        parser.add_argument("--min_iou", type=float, default=0.25, help="Min IoU threshold for evaluation")
        
        parser.add_argument("--no_height", action="store_true", help="Do NOT use height signal in input.")
        parser.add_argument("--no_lang_cls", action="store_true", help="Do NOT use language classifier.")
        parser.add_argument("--no_nms", action="store_true", help="do NOT use non-maximum suppression for post-processing.")
        
        parser.add_argument("--use_tf", action="store_true", help="Enable teacher forcing")
        parser.add_argument("--use_color", action="store_true", help="Use RGB color in input.")
        parser.add_argument("--use_normal", action="store_true", help="Use RGB color in input.")
        parser.add_argument("--use_multiview", action="store_true", help="Use multiview images.")
        parser.add_argument("--use_train", action="store_true", help="Use train split in evaluation.")
        parser.add_argument("--use_last", action="store_true", help="Use the last model")
        parser.add_argument("--use_topdown", action="store_true", help="Use top-down attention for captioning.")
        parser.add_argument("--use_relation", action="store_true", help="Use object-to-object relation in graph.")
        
        parser.add_argument("--eval_caption", action="store_true", help="evaluate the reference localization results")
        parser.add_argument("--eval_detection", action="store_true", help="evaluate the object detection results")
        parser.add_argument("--eval_pretrained", action="store_true", help="evaluate the pretrained object detection results")
        
        parser.add_argument("--force", action="store_true", help="generate the results by force")
        parser.add_argument("--save_interm", action="store_true", help="Save the intermediate results")
        self.args = parser.parse_args()

    @torch.no_grad()
    def inference(self, inputs, meta):
        cfg = self.cfg

        trajectories = human_poses = trajectory_boxes = skeleton_imgs = trajectory_box_masks = None
        if cfg.MODEL.USE_TRAJECTORIES:
            trajectories = meta['trajectories']
        if cfg.MODEL.USE_HUMAN_POSES:
            human_poses = meta['human_poses']
        if cfg.DETECTION.ENABLE_TOI_POOLING or cfg.MODEL.USE_TRAJECTORY_CONV:
            trajectory_boxes = meta['trajectory_boxes']
        if cfg.MODEL.USE_SPA_CONF:
            skeleton_imgs = meta['skeleton_imgs']
            trajectory_box_masks = meta['trajectory_box_masks']
        preds, action_labels, bbox_pair_ids, gt_bbox_pair_ids = self.model(
            inputs,
            meta["boxes"],
            meta['proposal_classes'],
            meta['proposal_lengths'],
            meta['action_labels'],
            meta['obj_classes'],
            meta['obj_classes_lengths'],
            trajectories=trajectories,
            human_poses=human_poses,
            trajectory_boxes=trajectory_boxes,
            skeleton_imgs=skeleton_imgs,
            trajectory_box_masks=trajectory_box_masks,
        )

        preds_score = F.sigmoid(preds).cpu()
        preds = preds_score >= 0.5 # Convert scores into 'True' or 'False'
        action_labels = action_labels.cpu()
        boxes = meta["boxes"].cpu()
        obj_classes = meta['obj_classes'].cpu()
        # obj_classes_lengths = meta['obj_classes_lengths'].cpu()
        bbox_pair_ids = bbox_pair_ids.cpu()
        gt_bbox_pair_ids = gt_bbox_pair_ids.cpu()
        # hopairs = hopairs # .cpu()
        proposal_scores = meta['proposal_scores'].cpu()
        gt_boxes = meta['gt_boxes'].cpu()
        proposal_classes = meta['proposal_classes'].cpu()

        return preds_score, preds, proposal_scores, proposal_classes


if __name__ == "__main__":
    ThreeDJCG()