import numpy as np

from overcooked_ai.ai_types import ObjectId, IsTile
from overcooked_ai.dataset_types import BBoxAnnotation


# TODO: add unit tests

def filter_tile_annotations(annotations: list[BBoxAnnotation]) -> list[BBoxAnnotation]:
    """
    Filter annotations to only include static tile objects.

    @param annotations: list of BBoxAnnotation
    @return list of BBoxAnnotation
    """

    return [anno for anno in annotations if ObjectId(anno.category_id) in IsTile]


def convert_from_annotations_to_frame_coord_xycats(annotations: list[BBoxAnnotation]) -> np.ndarray:
    """
    Convert annotations to (x, y, category_id) entries in frame coordinates.

    @param annotations: list of BBoxAnnotation
    @return np.ndarray of shape (N, 3)
    """

    xycats = []
    for anno in annotations:
        x, y = anno.bbox.get_mid_xy()
        xycats.append((x, y, anno.category_id))
    return np.array(xycats, dtype=np.float64)
