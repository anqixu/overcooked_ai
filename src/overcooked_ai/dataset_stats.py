import numpy as np

from overcooked_ai.dataset_types import BBoxAnnotation
from overcooked_ai.grid_homography import match_euclidean_dist_linear_sum_assignment_xycats


# TODO: add unit tests
def filter_annotations_by_min_score(
    annotations: list[BBoxAnnotation], min_score: float
) -> list[BBoxAnnotation]:
    """
    Filter annotations by minimum score.

    @param annotations: list of BBoxAnnotation
    @param min_score: minimum score

    @return list of BBoxAnnotation
    """

    return [anno for anno in annotations if anno.score is not None and anno.score >= min_score]


def filter_annotations_by_nms_thresh(
    annotations: list[BBoxAnnotation], nms_thresh: float, filter_by_category: bool
) -> list[BBoxAnnotation]:
    """
    Filter annotations by non-maximum suppression threshold on pairwise IOUs, removing low-scoring annotations.

    @param annotations: list of BBoxAnnotation
    @param nms_thresh: non-maximum suppression threshold
    @param filter_by_category: whether to filter within category, or across all categories

    NOTE: empirically, seems like detectron2 uses filter_by_category=True

    @return list of BBoxAnnotation
    """

    def _filter_group(
        annotations_group: list[BBoxAnnotation], nms_thresh: float
    ) -> list[BBoxAnnotation]:
        # Sort annos by descending score, and greedily drop other annos that have IOU exceeding threshold
        sorted_annos = sorted(annotations_group, key=lambda a: a.score, reverse=True)
        dropped = [False] * len(sorted_annos)
        result = []
        for anno_idx, anno in enumerate(sorted_annos):
            if dropped[anno_idx]:
                continue
            result.append(anno)
            for lower_scored_anno_idx in range(anno_idx + 1, len(sorted_annos)):
                if (
                    not dropped[lower_scored_anno_idx]
                    and anno.iou(sorted_annos[lower_scored_anno_idx]) >= nms_thresh
                ):
                    dropped[lower_scored_anno_idx] = True
        return result

    if filter_by_category:
        filtered_annotations = []
        all_category_ids = set(anno.category_id for anno in annotations)
        for category_id in all_category_ids:
            annos_for_category = [anno for anno in annotations if anno.category_id == category_id]
            filtered_annotations.extend(_filter_group(annos_for_category, nms_thresh))
        return filtered_annotations

    else:
        return _filter_group(annotations, nms_thresh)


def compute_num_tp_fp_fn(
    pred_annotations: list[BBoxAnnotation], gt_annotations: list[BBoxAnnotation]
) -> tuple[int, int, int]:
    """
    Compute number of true positives, false positives, and false negatives between predicted and ground truth annotations across categories.

    @param pred_annotations: list of BBoxAnnotation
    @param gt_annotations: list of BBoxAnnotation

    @return tuple of (num_true_positives, num_false_positives, num_false_negatives)
    """

    num_true_positives = 0
    num_false_positives = 0
    num_false_negatives = 0

    all_category_ids = set(anno.category_id for anno in pred_annotations + gt_annotations)

    for category_id in all_category_ids:
        pred_annos_for_category = [
            anno for anno in pred_annotations if anno.category_id == category_id
        ]
        gt_annos_for_category = [anno for anno in gt_annotations if anno.category_id == category_id]
        if len(pred_annos_for_category) == 0 or len(gt_annos_for_category) == 0:
            num_false_positives += len(pred_annos_for_category)
            num_false_negatives += len(gt_annos_for_category)
            continue

        # Associate pred_annos with gt_annos using linear sum assignment
        pred_xycats = np.array([anno.get_xycat() for anno in pred_annos_for_category])
        gt_xycats = np.array([anno.get_xycat() for anno in gt_annos_for_category])
        matched_pred_gt_idxs = match_euclidean_dist_linear_sum_assignment_xycats(
            pred_xycats,
            gt_xycats,
        )
        matched_pred_idxs = set(pred_idx for pred_idx, _ in matched_pred_gt_idxs)
        matched_gt_idxs = set(gt_idx for _, gt_idx in matched_pred_gt_idxs)
        unmatched_pred_idxs = set(range(len(pred_annos_for_category))) - matched_pred_idxs
        unmatched_gt_idxs = set(range(len(gt_annos_for_category))) - matched_gt_idxs

        num_true_positives += len(matched_pred_gt_idxs)
        num_false_positives += len(unmatched_pred_idxs)
        num_false_negatives += len(unmatched_gt_idxs)

    return num_true_positives, num_false_positives, num_false_negatives
