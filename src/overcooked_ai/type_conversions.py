from typing import Sequence

import numpy as np

from overcooked_ai.ai_types import ObjectId
from overcooked_ai.game_maps import num_pixels_per_tile, world_1_1_tile_short_labels


def convert_from_tile_idx_to_value(row_or_col_idx: int) -> float:
    """
    Convert from row or col index to x or y value, based on internal `num_pixels_per_tile`.
    """
    return row_or_col_idx * num_pixels_per_tile + num_pixels_per_tile / 2


def convert_from_world_tiles_to_xycats(world_tiles: Sequence[Sequence[ObjectId]]) -> np.ndarray:
    """
    Convert from world tiles to Nx3 matrix, where each row is (x, y, category_id), in grid coordinates.

    Let P = num_pixels_per_tile, maps:
    (grid_row_idx, grid_col_idx) to
    (grid_row_idx*P + P/2, grid_col_idx*P + P/2).
    """

    xycats = []

    for row_idx, row in enumerate(world_tiles):
        for col_idx, object_id in enumerate(row):
            if object_id in [ObjectId.TileFreeSpace]:
                continue
            xycats.append(
                (
                    convert_from_tile_idx_to_value(col_idx),
                    convert_from_tile_idx_to_value(row_idx),
                    int(object_id),
                )
            )
    return np.array(xycats).astype(np.float32)


def get_world_1_1_corner_grid_xys() -> list[list[float]]:
    """
    Return the corner tiles of world 1-1 in grid coordinates.

    @return corner_grid_xys: (4 x 2) list of [x, y] grid coordinates
    """

    world_1_1_num_rows = len(world_1_1_tile_short_labels)
    world_1_1_num_cols = len(world_1_1_tile_short_labels[0])

    corner_col_row_idxs = [
        [0, 0],
        [0, world_1_1_num_rows - 1],
        [world_1_1_num_cols - 1, world_1_1_num_rows - 1],
        [world_1_1_num_cols - 1, 0]]

    corner_grid_xys = [[convert_from_tile_idx_to_value(v) for v in xy] for xy in corner_col_row_idxs]
    return corner_grid_xys