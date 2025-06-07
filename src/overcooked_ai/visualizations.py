from collections import namedtuple

import cv2
import numpy as np
import matplotlib.colors
from matplotlib import pyplot as plt

from overcooked_ai.dataset_types import BBoxAnnotation

# NOTE: to fix issue with VS Code Debugger crashing when importing cv2/matplotlib/PyQt5,
#       delete/rename `PATH/site-packages/cv2/qt/plugins` dir, then
#       `ln -s ../../PyQt5/Qt5/plugins plugins`


LabelFields = namedtuple(
    "LabelFields",
    (
        "bbox_tl_px",
        "bbox_br_px",
        "label_trk",
        "label_cls",
        "label_trk_text_px",
        "label_trk_tl_px",
        "label_trk_br_px",
        "label_cls_text_px",
        "label_cls_tl_px",
        "label_cls_br_px",
        "bg_color",
        "img_scale",
        "bbox_thickness",
        "font_face",
        "font_scale",
        "font_thickness",
        "label_margin_px",
    ),
)


def _gen_anno_label_fields(
    anno: BBoxAnnotation,
    idx: int,
    labels: list[str],
    show_grid_rc: bool = False,
    img_scale: float = 0.5,
    bbox_thickness: int = 2,
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.325,
    font_thickness: int = 1,
    label_margin_px: int = 1,
    cmap: matplotlib.colors.Colormap = plt.cm.tab10,  # type: ignore
    bg_cmap_scale: float = 0.7,
) -> LabelFields:
    label_trk = f"{idx}|"
    if anno.score is None:  # Assume ground-truth
        label_trk += "GT"
    elif anno.score < 0:  # KF prediction count
        label_trk += f"{int(anno.score)}"
    else:
        label_trk += f"{int(round(anno.score*100))}%"
    label_cls = f"{labels[anno.category_id]}"
    if show_grid_rc and anno.grid_row_idx is not None and anno.grid_col_idx is not None:
        label_cls += f"|{anno.grid_row_idx},{anno.grid_col_idx}"

    bbox_half_thickness = bbox_thickness // 2
    anno_left, anno_top, anno_right, anno_bottom = (
        int(round(anno.bbox.x_min_px * img_scale)),
        int(round(anno.bbox.y_min_px * img_scale)),
        int(round(anno.bbox.x_max_px * img_scale)),
        int(round(anno.bbox.y_max_px * img_scale)),
    )
    (label_trk_width, label_trk_height_wo_bl), label_trk_baseline = cv2.getTextSize(
        label_trk, font_face, font_scale, font_thickness
    )
    label_trk_height = label_trk_height_wo_bl + label_trk_baseline
    (label_cls_width, label_cls_height_wo_bl), label_cls_baseline = cv2.getTextSize(
        label_cls, font_face, font_scale, font_thickness
    )
    label_cls_height = label_cls_height_wo_bl + label_cls_baseline
    return LabelFields(
        bbox_tl_px=(anno_left, anno_top),
        bbox_br_px=(anno_right, anno_bottom),
        label_trk=label_trk,
        label_cls=label_cls,
        label_trk_text_px=(
            anno_left + label_margin_px - bbox_half_thickness,
            anno_top + label_trk_height_wo_bl + label_margin_px - bbox_half_thickness,
        ),
        label_trk_tl_px=(
            anno_left - bbox_half_thickness,
            anno_top - bbox_half_thickness,
        ),
        label_trk_br_px=(
            anno_left + label_trk_width + 2 * label_margin_px - bbox_half_thickness,
            anno_top + label_trk_height + 2 * label_margin_px - bbox_half_thickness,
        ),
        label_cls_text_px=(
            anno_left + label_margin_px - bbox_half_thickness,
            anno_bottom - label_cls_baseline + label_margin_px - bbox_half_thickness,
        ),
        label_cls_tl_px=(
            anno_left - bbox_half_thickness,
            anno_bottom - label_cls_height - bbox_half_thickness,
        ),
        label_cls_br_px=(
            anno_left + label_cls_width + 2 * label_margin_px - bbox_half_thickness,
            anno_bottom + 2 * label_margin_px - bbox_half_thickness,
        ),
        bg_color=tuple(int(v * 255 * bg_cmap_scale) for v in cmap(idx % 10)[:3]),
        img_scale=img_scale,
        bbox_thickness=bbox_thickness,
        font_face=font_face,
        font_scale=font_scale,
        font_thickness=font_thickness,
        label_margin_px=label_margin_px,
    )


def plot_track_annotations(
    img: np.ndarray,
    annotations: list[BBoxAnnotation],
    idxs: list[int],
    labels: list[str],
    show_grid_rc: bool = False,
    img_scale: float = 0.5,
    fg_color: tuple[int, int, int] = (255, 255, 255),
    bbox_alpha=0.8,
):
    """
    Sample usage:

        img = cv2.imread(os.path.join(SOURCE_DIR, entry.file_name))
        annotations = sorted(entry.annotations, key=lambda anno: (anno.top, anno.bottom, anno.left, anno.right))
        idxs = range(len(annotations))
        img_annos = plot_track_annotations(img, annotations, idxs, plot_labels)

        plt.figure(figsize=(24, 20))
        plt.imshow(cv2.cvtColor(img_annos, cv2.COLOR_BGR2RGB))
    """

    label_plot_fields = [
        _gen_anno_label_fields(anno, idx, labels, show_grid_rc, img_scale=img_scale)
        for idx, anno in zip(idxs, annotations)
    ]

    img_bg = cv2.resize(img, None, fx=img_scale, fy=img_scale)
    img_boxes = img_bg.copy()
    for f in label_plot_fields:
        cv2.rectangle(img_boxes, f.bbox_tl_px, f.bbox_br_px, fg_color, int(2 * f.bbox_thickness))
    for f in label_plot_fields:
        cv2.rectangle(img_boxes, f.bbox_tl_px, f.bbox_br_px, f.bg_color, f.bbox_thickness)
        cv2.rectangle(img_boxes, f.label_trk_tl_px, f.label_trk_br_px, f.bg_color, cv2.FILLED)
        cv2.rectangle(img_boxes, f.label_cls_tl_px, f.label_cls_br_px, f.bg_color, cv2.FILLED)
    img_annos = cv2.addWeighted(img_bg, 1 - bbox_alpha, img_boxes, bbox_alpha, 0.0)
    for f in label_plot_fields:
        cv2.putText(
            img_annos,
            f.label_trk,
            f.label_trk_text_px,
            f.font_face,
            f.font_scale,
            fg_color,
            f.font_thickness,
            cv2.LINE_8,
        )
        cv2.putText(
            img_annos,
            f.label_cls,
            f.label_cls_text_px,
            f.font_face,
            f.font_scale,
            fg_color,
            f.font_thickness,
            cv2.LINE_8,
        )

    return img_annos
