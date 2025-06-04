from typing import Sequence

import numpy as np

from overcooked_ai.ai_types import ObjectId
from overcooked_ai.game_maps import num_pixels_per_tile


def convert_from_world_tiles_to_xycats(world_tiles: Sequence[Sequence[ObjectId]]) -> np.ndarray:
    """
    Convert from world tiles to Nx3 matrix, where each row is (x, y, category_id), in grid coordinates.

    Let P = num_pixels_per_tile, maps:
    (grid_row_idx, grid_col_idx) to
    (grid_row_idx*P + P/2, grid_col_idx*P + P/2).
    """

    P = num_pixels_per_tile
    P_half = P / 2
    xycats = []

    for row_idx, row in enumerate(world_tiles):
        for col_idx, object_id in enumerate(row):
            if object_id in [ObjectId.TileFreeSpace]:
                continue
            xycats.append((col_idx * P + P_half, row_idx * P + P_half, int(object_id)))
    return np.array(xycats).astype(np.float32)
