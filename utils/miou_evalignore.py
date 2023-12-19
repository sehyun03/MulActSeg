from typing import Any, Dict
import numpy as np
import torch

from .miou import MeanIoU


class IoUIgnore(MeanIoU):
    def __init__(self,
                 num_classes: int,
                 ignore_label: int,
                 output_tensor: str = 'outputs',
                 target_tensor: str = 'targets',
                 name: str = 'iou') -> None:
        super().__init__(num_classes, ignore_label, output_tensor, target_tensor, name)
        self.total_seen = 0
        self.total_correct = 0
        self.total_positive = 0

    def _after_step(self, output_dict: Dict[str, Any]) -> None:

        outputs_all = output_dict[self.output_tensor]
        targets_all = output_dict[self.target_tensor]
        assert(type(outputs_all) == torch.Tensor)

        cdx = self.num_classes
        ldx = self.ignore_label
        self.total_seen += torch.sum(targets_all == ldx).item()
        self.total_correct += torch.sum(
            (targets_all == ldx) & (outputs_all == cdx)).item()
        self.total_positive += torch.sum(
            outputs_all == cdx).item()

    def _after_epoch(self, ignore_label_list=None) -> None:
        ignore_iou = 0

        if self.total_seen == 0:
            ignore_iou = 100.0
        else:
            cur_iou = self.total_correct / (self.total_seen + self.total_positive - self.total_correct)
            ignore_iou = cur_iou * 100

        return ignore_iou
    
    def _after_epoch_ipr(self):
        ignore_iou = 0
        ignore_prec = 0
        ignore_recall = 0

        if self.total_seen == 0:
            ignore_iou = 100.0
            ignore_prec = 100.0
            ignore_recall = 100.0
        else:
            cur_iou = self.total_correct / (self.total_seen + self.total_positive - self.total_correct)
            cur_prec = self.total_correct / self.total_positive
            cur_recall = self.total_correct / self.total_seen
            ignore_iou = cur_iou * 100
            ignore_prec = cur_prec * 100
            ignore_recall = cur_recall * 100

        return (ignore_iou, ignore_prec, ignore_recall)