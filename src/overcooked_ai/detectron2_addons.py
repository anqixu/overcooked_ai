import numpy as np
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.structures import Boxes, Instances
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


@ROI_HEADS_REGISTRY.register()
class StandardROIHeadsWithClassProbs(StandardROIHeads):
    """
    Use as:
    from ai.detectron2_addons import StandardROIHeadsWithClassProbs
    # load detectron2's cfg, as usual
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeadsWithClassProbs"
    # load detectron2 model using cfg, as usual
    """

    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.softmax = torch.nn.Softmax(dim=1)

    def _forward_box(self, features: dict[str, torch.Tensor], proposals: list[Instances]):
        """
        Code copied from StandardROIHeads._forwardbox().

        In inference mode, this version further saves class probabilities into each prediction instance as pred_instances[i].pred_class_probs.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, nms_inds = self.box_predictor.inference(predictions, proposals)
            pred_class_scores = predictions[0][nms_inds[0]][:, :-1]  # N instances x K classes
            pred_class_probs = self.softmax(pred_class_scores)
            pred_instances[0].set("pred_class_probs", pred_class_probs)

            return pred_instances


def predict_classes(
    model: GeneralizedRCNN, image_rgb: np.ndarray, boxes_xyxy: np.ndarray, device: str
) -> tuple[list[int], list[float]]:
    """Predict class label and prob for arbitrary bounding boxes within image."""

    # Convert np image into tensor
    image_tensor = torch.as_tensor(image_rgb.astype("float32").transpose(2, 0, 1))
    height, width = image_tensor.shape[:2]
    image_tensor.to(device)
    inputs = [
        {"image": image_tensor, "height": height, "width": width},
    ]

    boxes_per_image = [Boxes(torch.tensor(boxes_xyxy)).to(device)]  # only 1 image, so 1 Boxes obj
    num_boxes_per_image = [len(p) for p in boxes_per_image]

    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        features_for_roi_head = [features[f] for f in model.roi_heads.box_in_features]

        box_features_flat_combined = model.roi_heads.box_pooler(
            features_for_roi_head, boxes_per_image
        )
        box_fcn_features_flat_combined = model.roi_heads.box_head(box_features_flat_combined)
        scores_flat_combined, _ = model.roi_heads.box_predictor(box_fcn_features_flat_combined)
        probs_flat_combined = F.softmax(scores_flat_combined, dim=-1)
        pred_classes_excluding_bg_combined = torch.argmax(
            probs_flat_combined[:, :-1], dim=1
        )  # skip last col==bg
        prob_classes_combined = probs_flat_combined[
            torch.arange(len(probs_flat_combined)), pred_classes_excluding_bg_combined
        ]

        # pred_classes_per_image = pred_classes_excluding_bg_combined.cpu().split(num_boxes_per_image, dim=0)
        # prob_classes_per_image = prob_classes_combined.cpu().split(num_boxes_per_image, dim=0)
        # pred_classes = list(pred_classes_per_image[0].numpy())
        # prob_classes = list(prob_classes_per_image[0].numpy())

        pred_classes = list(pred_classes_excluding_bg_combined.cpu().numpy())
        prob_classes = list(prob_classes_combined.cpu().numpy())
        return pred_classes, prob_classes
