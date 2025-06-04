import dataclasses

import numpy as np


@dataclasses.dataclass
class Point2D:
    x: float
    y: float


@dataclasses.dataclass
class GridImageHomography:
    H_grid_img: np.ndarray
