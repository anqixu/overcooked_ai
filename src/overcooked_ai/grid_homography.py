from typing import Sequence

import numpy as np
import scipy.linalg
import scipy.optimize


def normalize_xys(xys: np.ndarray, by_bbox: bool = False) -> np.ndarray:
    """
    Normalizes rows of Nx2 array.

    @param xys: Nx2 array of x, y coordinates
    @param by_bbox: bool, if True, each normalized column is bound to [-1, 1]; else, each normalized column has zero-mean and 1-stdev (a.k.a. "whiten")

    @return xys_normalized: Nx2 array of normalized x, y coordinates
    """

    assert xys.shape[0] > 0, "xys must have at least one row"
    assert xys.shape[1] == 2, "xys must have 2 columns"
    assert not np.any(np.isnan(xys)), "xys must not contain NaN values"
    assert not np.any(np.isinf(xys)), "xys must not contain infinite values"

    if by_bbox:
        min_xy = np.min(xys, axis=0)
        max_xy = np.max(xys, axis=0)
        magnitude_xy = max_xy - min_xy
        zero_range_axes = magnitude_xy == 0
        magnitude_xy[zero_range_axes] = 1
        xys_normalized = (
            2 * (xys - np.expand_dims(min_xy, axis=0)) / np.expand_dims(magnitude_xy, axis=0) - 1
        )
        xys_normalized[:, zero_range_axes] = 0
    else:
        mean_xy = np.mean(xys, axis=0)
        std_xy = np.std(xys, axis=0)
        std_xy[std_xy == 0] = 1
        xys_normalized = (xys - mean_xy) / std_xy

    return xys_normalized


def whiten_hxys(hxys: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Whiten hxys by shifting to xy-centroid + rescale to avg length of sqrt(2)

    @param hxys: (N x 3) homogeneous points of the form (x, y, w)

    @return hxys_whitened: (N x 3) whitened homogeneous points of the form (x, y, w)
    @return hxys_mean: (2,) mean of whitened homogeneous points of the form (x, y, w)
    @return hxys_gain: scalar gain of whitened homogeneous points of the form (x, y, w)
    """

    assert hxys.shape[0] > 0, "hxys must have at least one row"
    assert hxys.shape[1] == 3, "hxys must have 3 columns"
    if hxys.shape[0] == 1:
        return np.array([[0, 0, 1]]), hxys[0, :2], 1
    assert not np.any(hxys[:, 2] == 0), "hxys must not have any w=0 entries"
    assert not np.any(np.isnan(hxys)), "hxys must not contain NaN values"
    assert not np.any(np.isinf(hxys)), "hxys must not contain infinite values"

    # Compute non-homogeneous xy points
    xys = hxys[:, :2] / np.expand_dims(hxys[:, 2], axis=1)

    # Compute mean + gain
    xys_mean = np.mean(xys, axis=0)
    xys_centered = xys - xys_mean
    xy_rescale_denom = np.mean(np.linalg.norm(xys_centered, axis=1))
    xy_rescale = np.sqrt(2) / xy_rescale_denom if np.abs(xy_rescale_denom) > 1e-6 else 1

    # Whiten
    xys_whitened = xys_centered * xy_rescale
    hxys_whitened = np.hstack((xys_whitened, np.ones((xys.shape[0], 1))))
    return hxys_whitened, xys_mean, xy_rescale


def match_nearest_neighbors_xycats(
    xycats_source: np.ndarray, xycats_target: np.ndarray, max_dist: float = np.inf
) -> list[tuple[int, int]]:
    """
    Match each source entry to the nearest-neighbor target entry matching category_id, optionally bounded by max_dist.

    @param xycats_source: Nx3, row = x, y, category_id
    @param xycats_target: Mx3, row = x, y, category_id
    @param max_dist: float, optional maximum distance for a match

    @return matched_source_target_idxs: list of (source_idx, target_idx)
    """

    assert xycats_source.shape[0] > 0, "xycats_source must have at least one row"
    assert xycats_target.shape[0] > 0, "xycats_target must have at least one row"
    assert xycats_source.shape[1] == 3, "xycats_source and xycats_target must have 3 columns"
    assert xycats_target.shape[1] == 3, "xycats_source and xycats_target must have 3 columns"
    assert not np.any(np.isnan(xycats_source)), "xycats_source must not contain NaN values"
    assert not np.any(np.isnan(xycats_target)), "xycats_target must not contain NaN values"
    assert not np.any(np.isinf(xycats_source)), "xycats_source must not contain infinite values"
    assert not np.any(np.isinf(xycats_target)), "xycats_target must not contain infinite values"

    matched_source_target_idxs = []
    # For each unique category_id...
    for category_id in set(xycats_source[:, 2]):
        # Get indices of source + target points with matching category_id
        idxs_source_for_category = np.where(xycats_source[:, 2] == category_id)[0]
        idxs_target_for_category = np.where(xycats_target[:, 2] == category_id)[0]
        if len(idxs_target_for_category) <= 0:
            continue

        # Compute source (row) to target (column) distances via broadcast
        xys_source_for_category = np.expand_dims(
            xycats_source[idxs_source_for_category, :2], axis=1
        )
        xys_target_for_category = np.expand_dims(
            xycats_target[idxs_target_for_category, :2], axis=0
        )
        source_target_dists = np.linalg.norm(
            xys_source_for_category - xys_target_for_category, axis=2
        )

        for source_idx, target_idx in enumerate(np.argmin(source_target_dists, axis=1)):
            if source_target_dists[source_idx, target_idx] <= max_dist:
                matched_source_target_idxs.append(
                    (
                        int(idxs_source_for_category[source_idx]),
                        int(idxs_target_for_category[target_idx]),
                    )
                )

    return matched_source_target_idxs


def match_euclidean_dist_linear_sum_assignment_xycats(
    xycats_source: np.ndarray, xycats_target: np.ndarray, max_dist: float = np.inf
) -> list[tuple[int, int]]:
    """
    Match source entries to target entries with the same category_id, via linear sum assignment, and optionally bounded by max_dist.

    WARNING: contrary to match_nearest_neighbors_xycats, this function may not match every source entry to a target entry,
             since linear sum assignment on N source entries to M target entries will return min(M, N) matches.

    @param xycats_source: Nx3, row = x, y, category_id
    @param xycats_target: Mx3, row = x, y, category_id
    @param max_dist: float, optional maximum distance for a match

    @return matched_source_target_idxs: list of (source_idx, target_idx)
    """

    assert xycats_source.shape[0] > 0, "xycats_source must have at least one row"
    assert xycats_target.shape[0] > 0, "xycats_target must have at least one row"
    assert xycats_source.shape[1] == 3, "xycats_source and xycats_target must have 3 columns"
    assert xycats_target.shape[1] == 3, "xycats_source and xycats_target must have 3 columns"
    assert not np.any(np.isnan(xycats_source)), "xycats_source must not contain NaN values"
    assert not np.any(np.isnan(xycats_target)), "xycats_target must not contain NaN values"
    assert not np.any(np.isinf(xycats_source)), "xycats_source must not contain infinite values"
    assert not np.any(np.isinf(xycats_target)), "xycats_target must not contain infinite values"

    matched_source_target_idxs = []
    # For each unique category_id...
    for category_id in set(xycats_source[:, 2]):
        # Get indices of source + target points with matching category_id
        idxs_source_for_category = np.where(xycats_source[:, 2] == category_id)[0]
        idxs_target_for_category = np.where(xycats_target[:, 2] == category_id)[0]
        if len(idxs_target_for_category) <= 0:
            continue

        # Compute source (row) to target (column) distances via broadcast
        xys_source_for_category = np.expand_dims(
            xycats_source[idxs_source_for_category, :2], axis=1
        )
        xys_target_for_category = np.expand_dims(
            xycats_target[idxs_target_for_category, :2], axis=0
        )
        source_target_dists = np.linalg.norm(
            xys_source_for_category - xys_target_for_category, axis=2
        )

        source_idxs, target_idxs = scipy.optimize.linear_sum_assignment(source_target_dists)
        for source_idx, target_idx in zip(source_idxs, target_idxs):
            if source_target_dists[source_idx, target_idx] <= max_dist:
                matched_source_target_idxs.append(
                    (
                        int(idxs_source_for_category[source_idx]),
                        int(idxs_target_for_category[target_idx]),
                    )
                )

    return matched_source_target_idxs


def direct_linear_transform(hxys_target: np.ndarray, hxys_source: np.ndarray) -> np.ndarray:
    """
    Direct Linear Transform: returns homography that maps from Xs to Xs_prime.

    Follows Multi-View Geometry book eqn. (4.3).

    WARNING: does not work for entry of the form (x, y, w=0).

    @param hxys_target: (N x 3) homogeneous points in target space of the form (x, y, w)
    @param hxys_source: (N x 3) homogeneous points in source space of the form (x, y, w)

    @return H_target_source: (3 x 3) homography from source to target
    """

    assert hxys_target.shape[0] > 0, "hxys_target must have at least one row"
    assert hxys_source.shape[0] > 0, "hxys_source must have at least one row"
    assert (
        hxys_target.shape == hxys_source.shape
    ), "hxys_target and hxys_source must have the same shape"
    assert hxys_target.shape[0] >= 4, "hxys_target and hxys_source must have at least 4 rows"
    assert hxys_target.shape[1] == 3, "hxys_target and hxys_source must have 3 columns"
    assert not np.any(np.isnan(hxys_target)), "hxys_target must not contain NaN values"
    assert not np.any(np.isnan(hxys_source)), "hxys_source must not contain NaN values"
    assert not np.any(np.isinf(hxys_target)), "hxys_target must not contain infinite values"
    assert not np.any(np.isinf(hxys_source)), "hxys_source must not contain infinite values"
    assert not np.any(hxys_target[:, 2] == 0), "hxys_target must not have any w=0 entries"
    assert not np.any(hxys_source[:, 2] == 0), "hxys_source must not have any w=0 entries"

    # Whiten both entries: shift to xy centroid + rescale to avg length of sqrt(2)
    hxys_target_whitened, hxys_target_mean, hxys_target_rescale = whiten_hxys(hxys_target)
    hxys_source_whitened, hxys_source_mean, hxys_source_rescale = whiten_hxys(hxys_source)

    # Build A matrix
    num_correspondences = len(hxys_target_whitened)
    A = np.zeros((2 * num_correspondences, 9), dtype=np.float64)
    for idx, (hxys_target_whitened, hxys_source_whitened) in enumerate(
        zip(hxys_target_whitened, hxys_source_whitened)
    ):
        x_prime, y_prime, w_prime = hxys_target_whitened
        A[2 * idx, 3:6] = -w_prime * hxys_source_whitened
        A[2 * idx, 6:9] = y_prime * hxys_source_whitened
        A[2 * idx + 1, 0:3] = w_prime * hxys_source_whitened
        A[2 * idx + 1, 6:9] = -x_prime * hxys_source_whitened

    # Solve for h vector via SVD null-space
    _, Sigma, V = scipy.linalg.svd(A)
    h_vec_whitened_prime_orig = V[-1, :]
    H_whitened_prime_orig = (
        np.reshape(h_vec_whitened_prime_orig, (3, 3)) / h_vec_whitened_prime_orig[-1]
    )

    # Check if matrix is singular (but not near identity)
    if (
        np.linalg.norm(H_whitened_prime_orig - np.eye(3)) > 1e-6
        and np.linalg.cond(H_whitened_prime_orig) > 1e8
    ):
        raise np.linalg.LinAlgError(
            "Near-singular solution detected - points may be close to coplanar"
        )

    # Rescale to original space
    T_from_orig = np.array(
        (
            (hxys_source_rescale, 0, -hxys_source_mean[0] * hxys_source_rescale),
            (0, hxys_source_rescale, -hxys_source_mean[1] * hxys_source_rescale),
            (0, 0, 1),
        ),
        dtype=np.float64,
    )
    T_to_prime = np.array(
        (
            (1 / hxys_target_rescale, 0, hxys_target_mean[0]),
            (0, 1 / hxys_target_rescale, hxys_target_mean[1]),
            (0, 0, 1),
        ),
        dtype=np.float64,
    )
    H_prime_orig = T_to_prime @ H_whitened_prime_orig @ T_from_orig

    return H_prime_orig


def extract_matched_hxys(
    matched_a_b_idxs: Sequence[tuple[int, int]], xys_a: np.ndarray, xys_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract matched entries from xys_a and xys_b.
    Note that this function is designed to work with entries of the general form of (x, y, ...),
    and not just homogeneous coordinates.

    @param matched_a_b_idxs: list of (a_idx, b_idx)
    @param xys_a: Nx(2+), containing (x, y, ...)
    @param xys_b: Nx(2+), containing (x, y, ...)

    @return (matched_a_hxys, matched_b_hxys): in homogeneous coordinates
    """

    assert xys_a.shape[0] > 0, "xys_a must have at least one row"
    assert xys_b.shape[0] > 0, "xys_b must have at least one row"
    assert xys_a.shape[1] >= 2, "xys_a must have at least 2 columns"
    assert xys_b.shape[1] >= 2, "xys_b must have at least 2 columns"

    matched_a_hxys, matched_b_hxys = [], []
    for a_idx, b_idx in matched_a_b_idxs:
        assert (
            a_idx < xys_a.shape[0]
        ), f"a_idx {a_idx} is out of bounds for xys_a with shape {xys_a.shape}"
        assert (
            b_idx < xys_b.shape[0]
        ), f"b_idx {b_idx} is out of bounds for xys_b with shape {xys_b.shape}"
        matched_a_hxys.append(np.array([xys_a[a_idx, 0], xys_a[a_idx, 1], 1.0]))
        matched_b_hxys.append(np.array([xys_b[b_idx, 0], xys_b[b_idx, 1], 1.0]))
    matched_a_hxys = np.array(matched_a_hxys) if matched_a_hxys else np.empty((0, 3))
    matched_b_hxys = np.array(matched_b_hxys) if matched_b_hxys else np.empty((0, 3))
    return matched_a_hxys, matched_b_hxys


def apply_homography(H_dest_source: np.ndarray, hxys_source: np.ndarray) -> np.ndarray:
    """
    Apply homography to homogeneous points.

    @param H_dest_source: 3x3, not necessarily normalized
    @param hxys_source: Nx3, not necessarily homogeneous-normalized

    @return hxys_dest: Nx3, homogeneous-normalized
    """

    # Expand single-row hxys_source
    if len(hxys_source.shape) == 1:
        hxys_source = np.expand_dims(hxys_source, axis=0)

    assert H_dest_source.shape == (3, 3), "H_dest_source must be 3x3"
    assert hxys_source.shape[1] == 3, "hxys_source must have 3 columns"
    assert not np.any(np.isnan(hxys_source)), "hxys_source must not contain NaN values"
    assert not np.any(np.isinf(hxys_source)), "hxys_source must not contain infinite values"
    assert not np.any(hxys_source[:, 2] == 0), "hxys_source must not have any w=0 entries"

    # Apply homography and normalize, to floating-point precision
    hxys_dest = hxys_source.astype(np.float64) @ H_dest_source.astype(np.float64).transpose()
    hxys_dest /= np.expand_dims(hxys_dest[:, 2], axis=1)
    return hxys_dest


def compute_frame_grid_homography(
    frame_xycats: np.ndarray,
    grid_xycats: np.ndarray,
    consensus_max_grid_pxs: float = 125,
    max_iters: int = 10,
    final_match_by_linear_sum_assignment: bool = True,
) -> tuple[np.ndarray, Sequence[tuple[int, int]]]:
    """
    Compute homography that maps from frame points to grid points.

    Assumes frame points may have multiple near-matches, originating from object detection,
    whereas grid points are unique and non-overlapping.

    @param frame_xycats: Nx3, row = x, y, category_id
    @param grid_xycats: Nx3, row = x, y, category_id
    @param consensus_max_grid_pxs: maximum distance for a match
    @param max_iters: int, maximum number of iterations
    @param final_match_by_linear_sum_assignment: bool, if True, use final match by linear sum assignment

    @return (H_frame_grid, matched_frame_grid_idxs)
        H_frame_grid: 3x3
        matched_frame_grid_idxs: list[(frame_idx, grid_idx)]
    """

    assert frame_xycats.shape[0] > 0, "frame_xycats must have at least one row"
    assert grid_xycats.shape[0] > 0, "grid_xycats must have at least one row"
    assert frame_xycats.shape[1] == 3, "frame_xycats must have 3 columns"
    assert grid_xycats.shape[1] == 3, "grid_xycats must have 3 columns"
    assert not np.any(np.isnan(frame_xycats)), "frame_xycats must not contain NaN values"
    assert not np.any(np.isnan(grid_xycats)), "grid_xycats must not contain NaN values"
    assert not np.any(np.isinf(frame_xycats)), "frame_xycats must not contain infinite values"
    assert not np.any(np.isinf(grid_xycats)), "grid_xycats must not contain infinite values"

    # Normalize coordinates for frame points and grid points
    # NOTE: normalize by bounds rather than whiten, since points may be severely spatially imbalanced
    # TODO: consider instead of normalizing based on batch, which may be arbitrarily off,
    #       take an average homograph from grid to frame as prior (from labelled dataset),
    #       project frame pooints into grid space, solve nearest-neighbor match, solve homography,
    #       then iteratively refine.
    grid_normalized_xycats = grid_xycats.copy()
    grid_normalized_xycats[:, :2] = normalize_xys(grid_normalized_xycats[:, :2], by_bbox=True)
    frame_normalized_xycats = frame_xycats.copy()
    frame_normalized_xycats[:, :2] = normalize_xys(frame_normalized_xycats[:, :2], by_bbox=True)

    min_frame_xy = np.min(frame_normalized_xycats[:, :2], axis=0)
    max_frame_xy = np.max(frame_normalized_xycats[:, :2], axis=0)
    frame_xy_max_range = np.max(max_frame_xy - min_frame_xy)
    assert frame_xy_max_range > 0, "frame_xycats cannot have zero range"
    consensus_max_grid_norm_pxs = consensus_max_grid_pxs / frame_xy_max_range

    # Bootstrap correspondences via nearest-neighbor, then solve for homography
    # NOTE: alteratives for seeding initial correspondence:
    # - nearest-neighbor radius from frame space to grid space
    # - subsequence match in radius order from frame to grid(+180' repeat)
    # - nearest-neighbor xy from grid to frame
    # TODO: justify why using nearest-neighbor rather than linear sum assignment to match
    matched_frame_grid_idxs = match_nearest_neighbors_xycats(
        frame_normalized_xycats, grid_normalized_xycats
    )
    num_matches = len(matched_frame_grid_idxs)
    assert (
        num_matches >= 4
    ), f"Solving for homography requires minimum 4 correspondences, found {num_matches}"

    matched_frame_normalized_hxys, matched_grid_normalized_hxys = extract_matched_hxys(
        matched_frame_grid_idxs, frame_normalized_xycats, grid_normalized_xycats
    )
    H_frame_grid_nearest_neighbor_normalized = direct_linear_transform(
        matched_frame_normalized_hxys, matched_grid_normalized_hxys
    )

    # Iteratively refine homography by matching nearest-neighbor grid projections
    # NOTE: map from frame pts to grid pts, rather than the opposite, since expect potential frame pts but unique grid pts
    seen_sets_of_matched_frame_grid_idxs = {frozenset(matched_frame_grid_idxs)}
    for iter in range(max_iters):
        # Compute grid projections of frame points, with category ids
        projected_matched_frame_normalized_hxys = apply_homography(
            np.linalg.inv(H_frame_grid_nearest_neighbor_normalized), matched_frame_normalized_hxys
        )
        projected_matched_frame_normalized_xycats = (
            projected_matched_frame_normalized_hxys  # verbose code
        )
        projected_matched_frame_normalized_xycats[:, 2] = [
            frame_normalized_xycats[frame_idx, 2] for frame_idx, _ in matched_frame_grid_idxs
        ]

        # Match projected frame points to grid points, optionally bounded by consensus_max_grid_norm_pxs
        matches_of_matched_frame_grid_idxs = match_nearest_neighbors_xycats(
            projected_matched_frame_normalized_xycats,
            grid_normalized_xycats,
            consensus_max_grid_norm_pxs,
        )
        if (
            len(matches_of_matched_frame_grid_idxs) < 4
        ):  # Bad match, insufficient number of correspondences
            break

        # Instead of maintaining pairs of matched_frame and grid idxs, express as pairs of frame and grid idxs
        matched_frame_grid_idxs = [
            (matched_frame_grid_idxs[match_idx][0], grid_idx)
            for match_idx, grid_idx in matches_of_matched_frame_grid_idxs
        ]
        if frozenset(matched_frame_grid_idxs) in seen_sets_of_matched_frame_grid_idxs:
            break  # Seen this set of matches, converged
        seen_sets_of_matched_frame_grid_idxs.add(frozenset(matched_frame_grid_idxs))

        # Refine homography
        matched_frame_normalized_hxys, matched_grid_normalized_hxys = extract_matched_hxys(
            matched_frame_grid_idxs, frame_normalized_xycats, grid_normalized_xycats
        )
        H_frame_grid_nearest_neighbor_normalized = direct_linear_transform(
            matched_frame_normalized_hxys, matched_grid_normalized_hxys
        )

    if final_match_by_linear_sum_assignment:
        # Compute grid projections of frame points, with category ids
        projected_matched_frame_normalized_hxys = apply_homography(
            np.linalg.inv(H_frame_grid_nearest_neighbor_normalized), matched_frame_normalized_hxys
        )
        projected_matched_frame_normalized_xycats = (
            projected_matched_frame_normalized_hxys  # verbose code
        )
        projected_matched_frame_normalized_xycats[:, 2] = [
            frame_normalized_xycats[frame_idx, 2] for frame_idx, _ in matched_frame_grid_idxs
        ]

        # Match projected frame points to grid points, optionally bounded by consensus_max_grid_norm_pxs
        matches_of_matched_frame_grid_idxs = match_euclidean_dist_linear_sum_assignment_xycats(
            projected_matched_frame_normalized_xycats,
            grid_normalized_xycats,
            consensus_max_grid_norm_pxs,
        )

        # Only update matched_frame_grid_idxs if there are at least 4 matches
        if len(matches_of_matched_frame_grid_idxs) >= 4:
            # Instead of maintaining pairs of matched_frame and grid idxs, express as pairs of frame and grid idxs
            matched_frame_grid_idxs = [
                (matched_frame_grid_idxs[match_idx][0], grid_idx)
                for match_idx, grid_idx in matches_of_matched_frame_grid_idxs
            ]

    matched_frame_hxys, matched_grid_hxys = extract_matched_hxys(
        matched_frame_grid_idxs, frame_xycats, grid_xycats
    )
    H_frame_grid = direct_linear_transform(matched_frame_hxys, matched_grid_hxys)

    return H_frame_grid, matched_frame_grid_idxs
