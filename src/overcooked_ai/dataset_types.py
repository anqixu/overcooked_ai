from __future__ import annotations

from pathlib import Path

import dataclasses

import imagesize
import json
import numpy as np

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
            raise ValueError(
                f"x_min_px ({self.x_min_px}) must be less than x_max_px ({self.x_max_px})"
            )
        if self.y_min_px >= self.y_max_px:
            raise ValueError(
                f"y_min_px ({self.y_min_px}) must be less than y_max_px ({self.y_max_px})"
            )

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
            return BBox(
                intersection_x_min, intersection_y_min, intersection_x_max, intersection_y_max
            )
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
class BBoxAnnotation:
    """POD for bounding box and category ID, with optional fields for detection and Overcooked grid coordinates."""

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
    def from_dict(cls, data_dict: dict) -> BBoxAnnotation:
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

    def get_xycat(self) -> tuple[float, float, int]:
        return self.bbox.get_mid_xy() + (self.category_id,)

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
        return f"BBoxAnnotation[{repr}]"


@dataclasses.dataclass
class DetectionDatasetEntry:
    """POD for a detectron2 dataset entry, corresponding to a single image with a list of detections."""

    file_name: str
    height: int
    width: int
    image_id: str
    annotations: list[BBoxAnnotation] = dataclasses.field(default_factory=list)

    # Non-detectron2 fields
    H_grid_img_vector: list[float] | None = None

    # TODO: add unit tests for DetectionDatasetEntry

    @classmethod
    def from_dict(cls, entry_dict: dict) -> DetectionDatasetEntry:
        # NOTE: dictionary entry is "H_grid_img" for backwards compatibility
        H_grid_img_vector = entry_dict.get("H_grid_img", None)
        annotations = [
            BBoxAnnotation.from_dict(annotation_dict)
            for annotation_dict in entry_dict["annotations"]
        ]

        obj = cls(
            file_name=str(entry_dict["file_name"]),
            height=int(entry_dict["height"]),
            width=int(entry_dict["width"]),
            image_id=str(entry_dict["image_id"]),
            annotations=annotations,
            H_grid_img_vector=H_grid_img_vector,
        )

        return obj

    @classmethod
    def from_image_path(
        cls, image_path: Path, image_id: str | None = None
    ) -> DetectionDatasetEntry:
        if image_id is None:
            image_id = image_path.stem
        width_px, height_px = imagesize.get(image_path)
        return cls(
            file_name=str(image_path), height=int(height_px), width=int(width_px), image_id=image_id
        )

    def to_dict(self) -> dict:
        entry_dict: dict = {
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
            "image_id": self.image_id,
        }
        if self.H_grid_img_vector is not None:
            entry_dict["H_grid_img"] = self.H_grid_img_vector
        entry_dict["annotations"] = [anno.to_dict() for anno in self.annotations]
        return entry_dict

    def find_nearest_annotation_idx(self, target_anno: BBoxAnnotation) -> int | None:
        """Find index of entry.annotations nearest to target."""

        if target_anno is None:
            return None
        dist_idx_tuples = []
        target_x_mid, target_y_mid = target_anno.bbox.get_mid_xy()
        for anno_idx, anno in enumerate(self.annotations):
            if anno.category_id != target_anno.category_id:
                continue
            x_mid, y_mid = anno.bbox.get_mid_xy()
            dist_idx_tuples.append(
                ((x_mid - target_x_mid) ** 2 + (y_mid - target_y_mid) ** 2, anno_idx)
            )
        dist_idx_tuples.sort()
        if len(dist_idx_tuples) > 0:
            return dist_idx_tuples[0][1]
        else:
            return None


@dataclasses.dataclass
class DetectionDataset:
    """POD for a detectron2 dataset, corresponding to a directory of images with a JSON file listing the annotations."""

    # TODO: add unit tests for DetectionDataset

    dataset_path: Path = Path()
    entries: list[DetectionDatasetEntry] = dataclasses.field(default_factory=list)
    thing_classes: list[str] = dataclasses.field(default_factory=list)
    tainted: bool = False

    @classmethod
    def load_from_json(cls, dataset_path: Path) -> DetectionDataset:
        """
        Following Detectron2's JSON format:
        https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts
        """

        obj = cls()
        obj.dataset_path = dataset_path
        with open(dataset_path, "r") as fh:
            dataset = json.load(fh)
        obj.entries = [DetectionDatasetEntry.from_dict(entry) for entry in dataset["dataset_dict"]]
        obj.thing_classes = dataset["metadata"]["thing_classes"]
        return obj

    @classmethod
    def load_from_images_dir(cls, imgs_dir: Path, dataset_path: Path) -> DetectionDataset:
        obj = cls()
        obj.dataset_path = dataset_path

        img_paths = []
        for image_path in Path(imgs_dir).iterdir():
            if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                img_paths.append(image_path)
        img_paths.sort()

        obj.entries = [DetectionDatasetEntry.from_image_path(path) for path in img_paths]
        obj.tainted = True
        return obj

    def save_to_json(self, dest_path: Path | None = None) -> None:
        if dest_path is not None:
            self.dataset_path = dest_path

        for entry in self.entries:
            entry.annotations.sort(key=BBoxAnnotation.sort_key)

        with open(self.dataset_path, "w") as fh:
            json.dump(
                {
                    "metadata": {"thing_classes": self.thing_classes},
                    "dataset_dict": [entry.to_dict() for entry in self.entries],
                },
                fh,
                indent=2,
            )

        self.tainted = False
