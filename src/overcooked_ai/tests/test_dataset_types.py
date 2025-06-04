import pytest
from overcooked_ai.dataset_types import BBox, BBoxDetection, BoxMode


def test_bbox_mid_xy():
    bbox = BBox(0, -4, 2, 2)
    assert bbox.get_mid_xy() == (1.0, -1.0)

    bbox = BBox(-1.5, 2.5, 3.5, 4.5)
    assert bbox.get_mid_xy() == (1.0, 3.5)


def test_bbox_xyxy():
    bbox = BBox(0, -1.1, 2, 3.3)
    assert bbox.get_bbox_xyxy() == (0, -1.1, 2, 3.3)


def test_bbox_width_height():
    bbox = BBox(0, -1.5, 2, 2)
    assert bbox.width == 2
    assert bbox.height == 3.5


def test_bbox_intersection():
    bbox1 = BBox(0, 0, 2, 2)
    bbox2 = BBox(1, -1, 1.5, 3)
    intersection = bbox1.intersection(bbox2)
    assert intersection.get_bbox_xyxy() == (1, 0, 1.5, 2)


def test_bbox_contains():
    bbox = BBox(0, -5, 2, -3.5)
    assert bbox.contains(0, -5) == True
    assert bbox.contains(1, -4) == True

    assert bbox.contains(2, -3.5) == False
    assert bbox.contains(2, -5) == False
    assert bbox.contains(1, -3.5) == False


def test_bbox_detection_serialization_plain():
    bbox = BBox(0, 1.1, 2, 3.4)
    detection = BBoxDetection(bbox=bbox, category_id=1)

    dict_data = detection.to_dict()
    assert dict_data == {
        "bbox": [0, 1.1, 2, 3.4],
        "bbox_mode": int(BoxMode.XYXY_ABS),
        "category_id": 1,
    }

    new_detection = BBoxDetection.from_dict(dict_data)
    assert new_detection.bbox.get_bbox_xyxy() == detection.bbox.get_bbox_xyxy()
    assert new_detection.category_id == detection.category_id
    assert new_detection.grid_col_idx == detection.grid_col_idx == None
    assert new_detection.grid_row_idx == detection.grid_row_idx == None
    assert new_detection.score == detection.score == None
    assert new_detection.class_probs == detection.class_probs == None
    assert new_detection.track_id == detection.track_id == None


def test_bbox_detection_serialization_with_optional_fields():
    bbox = BBox(0, 1.1, 2, 3.4)
    detection = BBoxDetection(
        bbox=bbox,
        category_id=1,
        grid_col_idx=2,
        grid_row_idx=3,
        score=0.5,
        class_probs=[0.1, 0.2, 0.7],
        track_id=1,
    )

    dict_data = detection.to_dict()
    assert dict_data == {
        "bbox": [0, 1.1, 2, 3.4],
        "bbox_mode": int(BoxMode.XYXY_ABS),
        "category_id": 1,
        "grid_col_idx": 2,
        "grid_row_idx": 3,
        "score": 0.5,
        "class_probs": [0.1, 0.2, 0.7],
        "track_id": 1,
    }

    new_detection = BBoxDetection.from_dict(dict_data)
    assert new_detection.bbox.get_bbox_xyxy() == detection.bbox.get_bbox_xyxy()
    assert new_detection.category_id == detection.category_id
    assert new_detection.grid_col_idx == detection.grid_col_idx
    assert new_detection.grid_row_idx == detection.grid_row_idx
    assert new_detection.score == detection.score
    assert new_detection.class_probs == detection.class_probs
    assert new_detection.track_id == detection.track_id
