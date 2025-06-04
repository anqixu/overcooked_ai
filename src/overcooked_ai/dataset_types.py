from __future__ import annotations

import dataclasses

from detectron2.structures.boxes import BoxMode


@dataclasses.dataclass
class BBox:
    """POD for axis-aligned bounding box."""

    # NOTE: x_max_px and y_max_px are exclusive, adhering to detectron2's format
    #       See: https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/boxes.py#L109-L114
    x_min_px: float
    y_min_px: float
    x_max_px: float  # exclusive
    y_max_px: float  # exclusive

    def __init__(
        self,
        x_min_px: float,
        y_min_px: float,
        x_max_px: float,
        y_max_px: float,
    ) -> None:
        self.x_min_px = x_min_px
        self.y_min_px = y_min_px
        self.x_max_px = x_max_px
        self.y_max_px = y_max_px

        if self.x_min_px >= self.x_max_px:
            raise ValueError(f"x_min_px ({self.x_min_px}) must be less than x_max_px ({self.x_max_px})")
        if self.y_min_px >= self.y_max_px:
            raise ValueError(f"y_min_px ({self.y_min_px}) must be less than y_max_px ({self.y_max_px})")


    def get_mid_xy(self) -> tuple[float, float]:
        """Return center (x, y) in pixel coordinates."""
        mid_x = (self.x_min_px + self.x_max_px) / 2.0
        mid_y = (self.y_min_px + self.y_max_px) / 2.0
        return (mid_x, mid_y)

    def get_mid_hxy(self) -> tuple[float, float, float]:
        """Return center (x, y) in homogeneous pixel coordinates."""
        mid_x, mid_y = self.get_mid_xy()
        return (mid_x, mid_y, 1.0)

    def get_bbox_xyxy(self) -> tuple[float, float, float, float]:
        """Return (x_min, y_min, x_max, y_max) vector adhering to detectron2's format."""
        bbox_xyxy = (self.x_min_px, self.y_min_px, self.x_max_px, self.y_max_px)
        return bbox_xyxy

    @property
    def width(self) -> float:
        """Return width of bounding box in pixel coordinates."""
        return self.x_max_px - self.x_min_px

    @property
    def height(self) -> float:
        """Return height of bounding box in pixel coordinates."""
        return self.y_max_px - self.y_min_px

    @property
    def area(self) -> float:
        """Return area of bounding box in pixel coordinates."""
        return self.width * self.height

    def intersection(self, other: BBox) -> BBox:
        """Return intersection of two bounding boxes in pixel coordinates."""
        intersection_x_min = max(self.x_min_px, other.x_min_px)
        intersection_y_min = max(self.y_min_px, other.y_min_px)
        intersection_x_max = min(self.x_max_px, other.x_max_px)
        intersection_y_max = min(self.y_max_px, other.y_max_px)
        try:
            return BBox(intersection_x_min, intersection_y_min, intersection_x_max, intersection_y_max)
        except ValueError as err:
            raise ValueError(f"Failed to compute intersection of {self} and {other}: {err}")

    def get_iou(self, other: BBox) -> float:
        """Return intersection over union (IoU) of two bounding boxes."""
        intersection = self.intersection(other)
        return intersection.area / float(self.area + other.area - intersection.area)

    def contains(self, x: float, y: float) -> bool:
        """Return whether a point is inside the bounding box."""
        return self.x_min_px <= x < self.x_max_px and self.y_min_px <= y < self.y_max_px
    
    def __repr__(self) -> str:
        return f"BBox(XYXY=({self.x_min_px},{self.y_min_px})-({self.x_max_px},{self.y_max_px}))"


@dataclasses.dataclass
class BBoxDetection:
    bbox: BBox
    category_id: int

    # Labelled grid coordinates
    grid_row_idx: int | None = None
    grid_col_idx: int | None = None

    # Score given by object detection model
    score: float | None = None

    # Class probabilities given by object detection model
    class_probs: list[float] | None = None

    # ID used to track object across frames
    track_id: int | None = None

    @classmethod
    def from_dict(cls, data_dict: dict) -> BBoxDetection:
        """Parse from structure adhering to detectron2's format."""

        assert data_dict["bbox_mode"] == int(
            BoxMode.XYXY_ABS
        ), f"Expected bbox_mode {BoxMode.XYXY_ABS}, got unsupported value: {data_dict['bbox_mode']}"
        x_min_px, y_min_px, x_max_px, y_max_px = [float(v) for v in data_dict["bbox"]]

        category_id = int(data_dict["category_id"])

        grid_row_idx = data_dict.get("grid_row_idx", None)
        grid_row_idx = int(grid_row_idx) if grid_row_idx is not None else None
        grid_col_idx = data_dict.get("grid_col_idx", None)
        grid_col_idx = int(grid_col_idx) if grid_col_idx is not None else None

        score = data_dict.get("score", None)
        score = float(score) if score is not None else None

        class_probs = (
            [float(p) for p in data_dict.get("class_probs", [])]
            if "class_probs" in data_dict
            else None
        )

        track_id = data_dict.get("track_id", None)
        track_id = int(track_id) if track_id is not None else None

        return cls(
            BBox(x_min_px, y_min_px, x_max_px, y_max_px),
            category_id,
            grid_row_idx,
            grid_col_idx,
            score,
            class_probs,
            track_id,
        )

    def to_dict(self) -> dict[str, int | list[float] | float]:
        data_dict: dict[str, int | list[float] | float] = {
            "bbox": [
                self.bbox.x_min_px,
                self.bbox.y_min_px,
                self.bbox.x_max_px,
                self.bbox.y_max_px,
            ],
            "bbox_mode": int(BoxMode.XYXY_ABS),
            "category_id": self.category_id,
        }
        if self.grid_row_idx is not None and self.grid_col_idx is not None:
            data_dict["grid_row_idx"] = self.grid_row_idx
            data_dict["grid_col_idx"] = self.grid_col_idx
        if self.score is not None:
            data_dict["score"] = self.score
        if self.class_probs is not None:
            data_dict["class_probs"] = self.class_probs
        if self.track_id is not None:
            data_dict["track_id"] = self.track_id
        return data_dict

    def sort_key(self) -> tuple[float, int, int, float, float, float, float, int]:
        r = self.grid_row_idx if self.grid_row_idx is not None else 100000
        c = self.grid_col_idx if self.grid_col_idx is not None else 100000
        s = self.score if self.score is not None else 1.1
        return (
            -s,
            r,
            c,
            self.bbox.y_min_px,
            self.bbox.x_min_px,
            self.bbox.y_max_px,
            self.bbox.x_max_px,
            self.category_id,
        )

    def __repr__(self) -> str:
        fields: dict[str, str] = {
            "XYXY": f"({self.bbox.x_min_px:.1f},{self.bbox.y_min_px:.1f})-({self.bbox.x_max_px:.1f},{self.bbox.y_max_px:.1f})",
            "ID": str(self.category_id),
        }
        if self.grid_row_idx is not None and self.grid_col_idx is not None:
            fields["GridRC"] = f"({self.grid_row_idx},{self.grid_col_idx})"
        if self.score is not None:
            fields["score"] = f"{self.score:.2f}"
        if self.class_probs is not None:
            fields["class_probs"] = "[" + ",".join(f"{p:.1e}" for p in self.class_probs) + "]"
        if self.track_id is not None:
            fields["track_id"] = f"{self.track_id}"
        repr = ", ".join(f"{k}={v}" for k, v in fields.items())
        return f"BBoxDetection[{repr}]"
