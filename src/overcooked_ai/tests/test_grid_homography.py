import pytest
import numpy as np

from overcooked_ai.grid_homography import (
    whiten_hxys,
    direct_linear_transform,
    normalize_xys,
    match_nearest_neighbors_xycats,
    apply_homography,
    extract_matched_hxys,
    match_euclidean_dist_linear_sum_assignment_xycats,
)


def test_normalize_xys_zero_values():
    xys = np.array([[0, 0], [0, 0]])

    # Test whitening mode
    xys_norm = normalize_xys(xys, by_bbox=False)
    assert np.allclose(xys_norm, 0)

    # Test bbox mode
    xys_norm = normalize_xys(xys, by_bbox=True)
    assert np.allclose(xys_norm, 0)


def test_normalize_xys_unit_values():
    xys = np.array([[1, 1], [-1, -1]], dtype=np.float64)

    # Test whitening mode
    xys_normalized = normalize_xys(xys, by_bbox=False)
    assert np.allclose(np.mean(xys_normalized, axis=0), 0)
    assert np.allclose(np.std(xys_normalized, axis=0), 1)

    # Test bbox mode
    xys_normalized = normalize_xys(xys, by_bbox=True)
    assert np.all(xys_normalized >= -1) and np.all(xys_normalized <= 1)


def test_normalize_xys_general_values():
    xys = np.array([[-1, 2], [3, -4], [-5, 6]])

    # Test whitening mode
    xys_normalized = normalize_xys(xys, by_bbox=False)
    assert np.allclose(np.mean(xys_normalized, axis=0), 0)
    assert np.allclose(np.std(xys_normalized, axis=0), 1)

    # Test bbox mode
    xys_normalized = normalize_xys(xys, by_bbox=True)
    assert np.all(xys_normalized >= -1) and np.all(xys_normalized <= 1)
    assert np.allclose(np.min(xys_normalized, axis=0), -1)
    assert np.allclose(np.max(xys_normalized, axis=0), 1)


def test_normalize_xys_extreme_scale_difference():
    # Create data where x values are very large while y values are small
    xys = np.array([[1e100, 1], [3e100, 2], [2e100, 3]])

    # Test whitening mode
    xys_normalized = normalize_xys(xys, by_bbox=False)
    # Both dimensions should be normalized to mean 0, std 1 despite extreme scale difference
    assert np.allclose(np.mean(xys_normalized, axis=0), 0)
    assert np.allclose(np.std(xys_normalized, axis=0), 1)

    # Test bbox mode
    xys_normalized = normalize_xys(xys, by_bbox=True)
    # Both dimensions should be scaled to [-1, 1] range regardless of original scale
    assert np.all(xys_normalized >= -1) and np.all(xys_normalized <= 1)
    assert np.allclose(np.min(xys_normalized, axis=0), -1)
    assert np.allclose(np.max(xys_normalized, axis=0), 1)


def test_normalize_xys_single_column_variance():
    xys = np.array([[1, 2], [1, 4], [1, 6]])

    # Test whitening mode - first column has 0 variance
    xys_normalized = normalize_xys(xys, by_bbox=False)
    assert np.allclose(xys_normalized[:, 0], 0)
    assert np.allclose(np.mean(xys_normalized[:, 1]), 0)
    assert np.allclose(np.std(xys_normalized[:, 1]), 1)

    # Test bbox mode - first column has 0 range
    xys_normalized = normalize_xys(xys, by_bbox=True)
    assert np.allclose(xys_normalized[:, 0], 0)
    assert np.allclose(np.min(xys_normalized[:, 1]), -1)
    assert np.allclose(np.max(xys_normalized[:, 1]), 1)


def test_normalize_xys_empty_input():
    xys = np.array([])

    for by_bbox in [False, True]:
        with pytest.raises(AssertionError):
            normalize_xys(xys, by_bbox=by_bbox)


def test_normalize_xys_invalid_input():
    xys = np.array([[1, 2, np.inf], [4, 5, 6]])

    for by_bbox in [False, True]:
        with pytest.raises(AssertionError):
            normalize_xys(xys, by_bbox=by_bbox)

    xys = np.array([[1, 2, np.nan], [4, 5, 6]])

    for by_bbox in [False, True]:
        with pytest.raises(AssertionError):
            normalize_xys(xys, by_bbox=by_bbox)


def test_whiten_hxys_zero_values():
    hxys = np.array([[0, 0, 1], [0, 0, 1]])

    hxys_whitened, xys_mean, _ = whiten_hxys(hxys)

    assert np.allclose(xys_mean, [0, 0])

    assert np.allclose(hxys_whitened[:, :2], 0)

    assert np.allclose(hxys_whitened[:, 2], 1)


def test_whiten_hxys_unit_values():
    hxys = np.array([[1, 1, 1], [-1, -1, 1], [2, 2, 2], [-2, -2, 2]], dtype=np.float64)

    hxys_whitened, xys_mean, _ = whiten_hxys(hxys)

    assert np.allclose(xys_mean, [0, 0])

    dists = np.linalg.norm(hxys_whitened[:, :2], axis=1)
    assert np.allclose(np.mean(dists), np.sqrt(2))

    assert np.allclose(hxys_whitened[:, 2], 1)


def test_whiten_hxys_general_values():
    hxys = np.array([[1, -2, 1], [-3, 5, 1], [-4, 6, -2]])

    hxys_whitened, xys_mean, _ = whiten_hxys(hxys)

    # Mean should be average of x and y coordinates, weighted by homogeneous coordinate
    assert np.allclose(xys_mean, [0, 0])

    # After whitening, points should have mean distance sqrt(2) from origin
    dists = np.linalg.norm(hxys_whitened[:, :2], axis=1)
    assert np.allclose(np.mean(dists), np.sqrt(2))

    # Homogeneous coordinates should be normalized to 1
    assert np.allclose(hxys_whitened[:, 2], 1)


def test_whiten_hxys_different_scales():
    # Test with x values around 1 and y values around 1e100
    hxys = np.array([[1.5, 2e100, 1], [2.0, 3e100, 1], [0.5, 1e100, 1]])

    hxys_whitened, xys_mean, _ = whiten_hxys(hxys)

    # Mean should capture the different scales
    assert np.allclose(xys_mean, [1.333333, 2e100], rtol=1e-5)

    # After whitening, points should be normalized to similar scales
    dists = np.linalg.norm(hxys_whitened[:, :2], axis=1)
    assert np.allclose(np.mean(dists), np.sqrt(2))

    # Last column should still be ones
    assert np.allclose(hxys_whitened[:, 2], 1)

    # Check that relative positions are preserved
    # Original x order: 1.5, 2.0, 0.5
    # Original y order: 2e100, 3e100, 1e100
    x_order = np.argsort(hxys_whitened[:, 0])
    y_order = np.argsort(hxys_whitened[:, 1])
    assert np.array_equal(x_order, [2, 0, 1])
    assert np.array_equal(y_order, [2, 0, 1])


def test_whiten_hxys_empty_input():
    hxys = np.array([]).reshape(0, 3)
    with pytest.raises(AssertionError):
        whiten_hxys(hxys)


def test_whiten_hxys_invalid_input():
    hxys = np.array([[1, np.nan, 1], [2, 3, 1]])
    with pytest.raises(AssertionError):
        whiten_hxys(hxys)

    hxys = np.array([[1, 2, 1], [np.inf, 3, 1]])
    with pytest.raises(AssertionError):
        whiten_hxys(hxys)


def test_whiten_hxys_point_at_infinity():
    hxys = np.array([[1, 2, 1], [3, 4, 0]])
    with pytest.raises(AssertionError):
        whiten_hxys(hxys)


def test_whiten_hxys_single_entry():
    hxys = np.array([[1, 2, 1]])

    hxys_whitened, xys_mean, _ = whiten_hxys(hxys)

    assert np.allclose(xys_mean, hxys[0, :2])

    assert np.allclose(hxys_whitened[0, :2], [0, 0])

    assert np.allclose(hxys_whitened[0, 2], 1)


def test_match_nearest_neighbors_xycats_exact_match():
    source = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0]])
    target = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0], [4, 4, 0]])

    matches = match_nearest_neighbors_xycats(source, target)

    expected = {(0, 0), (1, 1), (2, 2)}
    assert set(matches) == expected


def test_match_nearest_neighbors_xycats_no_matches():
    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([[1, 1, 1], [2, 2, 1]])

    matches = match_nearest_neighbors_xycats(source, target)

    assert len(matches) == 0


def test_match_nearest_neighbors_xycats_partial_matches():
    source = np.array([[1, 1, 0], [2, 2, 0], [1, 1, 1], [2, 2, 1], [3, 3, 2]])
    target = np.array([[1.1, 0.9, 0], [2.1, 2.1, 1]])

    matches = match_nearest_neighbors_xycats(source, target)

    expected = {(0, 0), (1, 0), (2, 1), (3, 1)}
    assert set(matches) == expected


def test_match_nearest_neighbors_xycats_empty_input():
    source = np.array([]).reshape(0, 3)
    target = np.array([[1, 1, 0], [2, 2, 0]])
    with pytest.raises(AssertionError):
        match_nearest_neighbors_xycats(source, target)

    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([]).reshape(0, 3)
    with pytest.raises(AssertionError):
        match_nearest_neighbors_xycats(source, target)


def test_match_nearest_neighbors_xycats_invalid_input():
    source = np.array([[1, np.nan, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [2, 2, 0]])
    with pytest.raises(AssertionError):
        match_nearest_neighbors_xycats(source, target)

    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [np.nan, 2, 0]])
    with pytest.raises(AssertionError):
        match_nearest_neighbors_xycats(source, target)

    source = np.array([[1, np.inf, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [2, 2, 0]])
    with pytest.raises(AssertionError):
        match_nearest_neighbors_xycats(source, target)

    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [np.inf, 2, 0]])
    with pytest.raises(AssertionError):
        match_nearest_neighbors_xycats(source, target)


def test_match_nearest_neighbors_xycats_max_dist():
    source = np.array([[1, 1, 0], [2, 2, 0], [3, 3, 0]])
    target = np.array([[1.1, 0.9, 0], [5, 5, 0], [2.2, 1.9, 0], [3.2, 2.9, 0], [4, 4, 0]])

    # With large max_dist
    matches = match_nearest_neighbors_xycats(source, target, max_dist=10.0)
    expected = {(0, 0), (1, 2), (2, 3)}
    assert set(matches) == expected

    # With exact max_dist matching nearest neighbor
    matches = match_nearest_neighbors_xycats(
        source, target, max_dist=0.2236
    )  # sqrt((0.1^2 + 0.1^2))
    expected = {(0, 0)}
    assert set(matches) == expected

    # With small max_dist
    matches = match_nearest_neighbors_xycats(source, target, max_dist=0.15)
    expected = {(0, 0)}
    assert set(matches) == expected


def test_match_nearest_neighbors_xycats_equal_distances():
    # Two target points equidistant from source point
    source = np.array([[1, 1, 0]])
    target = np.array([[2, 1, 0], [1, 2, 0]])  # Both sqrt(1) = 1 unit away

    matches = match_nearest_neighbors_xycats(source, target)

    # Should match with first point in target list
    expected = {(0, 0)}
    assert set(matches) == expected

    # Test with multiple source points
    source = np.array([[0, 0, 0], [2, 2, 0]])
    target = np.array([[1, 0, 0], [0, 1, 0], [3, 2, 0], [2, 3, 0]])
    # First source point equidistant to first two target points
    # Second source point equidistant to last two target points

    matches = match_nearest_neighbors_xycats(source, target)

    # Should match with first available equidistant point in each case
    expected = {(0, 0), (1, 2)}
    assert set(matches) == expected


def test_match_nearest_neighbors_xycats_extra_categories():
    source = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0], [4, 4, 1], [5, 5, 2]])
    target = np.array([[1.1, 0.9, 0], [2.1, 2.1, 1], [3.1, 3.1, 0], [5, 5, 3]])

    matches = match_nearest_neighbors_xycats(source, target)

    expected = {(0, 0), (1, 1), (2, 2), (3, 1)}
    assert set(matches) == expected


def test_match_euclidean_dist_linear_sum_assignment_xycats_exact_match():
    source = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0]])
    target = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0], [4, 4, 0]])

    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    expected = {(0, 0), (1, 1), (2, 2)}
    assert set(matches) == expected


def test_match_euclidean_dist_linear_sum_assignment_xycats_no_matches():
    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([[1, 1, 1], [2, 2, 1]])

    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    assert len(matches) == 0


def test_match_euclidean_dist_linear_sum_assignment_xycats_extra_source_matches():
    source = np.array([[1, 1, 0], [2, 2, 0], [1, 1, 1], [2, 2, 1], [3, 3, 2]])
    target = np.array([[1.1, 0.9, 0], [2.1, 2.1, 1]])

    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    expected = {(0, 0), (3, 1)}
    assert set(matches) == expected


def test_match_euclidean_dist_linear_sum_assignment_xycats_extra_target_matches():
    source = np.array([[1, 1, 0], [2, 2, 0], [2, 2, 1], [3, 3, 2]])
    target = np.array([[1.1, 0.9, 0], [1, 1, 1], [2.1, 2.1, 1], [3, 3, 1], [2, 2, 2], [5, 5, 2]])

    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    expected = {(0, 0), (2, 2), (3, 4)}
    assert set(matches) == expected


def test_match_euclidean_dist_linear_sum_assignment_xycats_empty_input():
    # Test with empty source
    source = np.array([]).reshape(0, 3)
    target = np.array([[1, 1, 0], [2, 2, 0]])

    with pytest.raises(AssertionError):
        match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    # Test with empty target
    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([]).reshape(0, 3)

    with pytest.raises(AssertionError):
        match_euclidean_dist_linear_sum_assignment_xycats(source, target)


def test_match_euclidean_dist_linear_sum_assignment_xycats_invalid_input():
    # Test with NaN values in source
    source = np.array([[1, np.nan, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [2, 2, 0]])

    with pytest.raises(AssertionError):
        match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    # Test with NaN values in target
    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [np.nan, 2, 0]])

    with pytest.raises(AssertionError):
        match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    # Test with inf values in source
    source = np.array([[1, np.inf, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [2, 2, 0]])

    with pytest.raises(AssertionError):
        match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    # Test with inf values in target
    source = np.array([[1, 1, 0], [2, 2, 0]])
    target = np.array([[1, 1, 0], [np.inf, 2, 0]])

    with pytest.raises(AssertionError):
        match_euclidean_dist_linear_sum_assignment_xycats(source, target)


def test_match_euclidean_dist_linear_sum_assignment_xycats_max_dist():
    # Test with max_dist constraint
    source = np.array([[1, 1, 0], [2, 2, 0], [3, 3, 0]])
    target = np.array([[1.1, 0.9, 0], [5, 5, 0], [2.2, 1.9, 0], [3.2, 2.9, 0], [4, 4, 0]])

    # With large max_dist
    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target, max_dist=10.0)
    expected = {(0, 0), (1, 2), (2, 3)}
    assert set(matches) == expected

    # With exact max_dist matching nearest neighbor
    matches = match_euclidean_dist_linear_sum_assignment_xycats(
        source, target, max_dist=0.2236
    )  # sqrt((0.1^2 + 0.1^2))
    expected = {(0, 0)}
    assert set(matches) == expected

    # With small max_dist
    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target, max_dist=0.15)
    expected = {(0, 0)}
    assert set(matches) == expected


def test_match_euclidean_dist_linear_sum_assignment_xycats_single_source():
    source = np.array([[1, 1, 0]])
    target = np.array([[1.1, 0.9, 0], [2, 2, 0], [3, 3, 0]])

    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    expected = {(0, 0)}
    assert set(matches) == expected


def test_match_euclidean_dist_linear_sum_assignment_xycats_extra_categories():
    source = np.array([[1, 1, 0], [2, 2, 1], [3, 3, 0], [4, 4, 2], [5, 5, 4]])
    target = np.array([[1.1, 0.9, 0], [2.1, 2.1, 1], [3.1, 3.1, 0], [5, 5, 3]])

    matches = match_euclidean_dist_linear_sum_assignment_xycats(source, target)

    expected = {(0, 0), (1, 1), (2, 2)}
    assert set(matches) == expected


def test_direct_linear_transform_identity():
    source = np.array(
        [
            [1.2, -0.5, 1],
            [3.7, 2.1, 2],
            [-1.4, 0.8, 1],
            [2.3, -1.9, 3],
            [0.6, 4.2, 1],
            [-2.8, 1.5, 2],
            [3.1, -2.4, 1],
        ]
    )
    target = source.copy()

    H = direct_linear_transform(target, source)

    assert np.allclose(H, np.eye(3))


def test_direct_linear_transform_translation():
    source = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    translation = np.array([2, 3])
    target = source.copy()
    target[:, :2] += translation

    H = direct_linear_transform(target, source)

    expected = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    assert np.allclose(H, expected)


def test_direct_linear_transform_rotation():
    # Test 60 degree rotation
    source = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

    # 60 degree CCW rotation matrix
    cos_60 = np.cos(np.pi / 3)
    sin_60 = np.sin(np.pi / 3)
    target = np.array(
        [
            [0, 0, 1],
            [cos_60, sin_60, 1],
            [-sin_60, cos_60, 1],
            [cos_60 - sin_60, sin_60 + cos_60, 1],
        ]
    )

    H = direct_linear_transform(target, source)

    # Expected 60 degree CCW rotation matrix
    expected = np.array([[cos_60, -sin_60, 0], [sin_60, cos_60, 0], [0, 0, 1]])
    assert np.allclose(H, expected)

    # Verify transformation works for all points
    for src, tgt in zip(source, target):
        transformed = H @ src
        transformed = transformed / transformed[2]  # Normalize homogeneous coordinate
        assert np.allclose(transformed, tgt)


def test_direct_linear_transform_scale():
    # Test uniform scale
    source = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    scale = 2.0
    target = source.copy() * scale
    target[:, 2] = 1

    H = direct_linear_transform(target, source)

    expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    assert np.allclose(H, expected)

    # Test non-uniform scale
    source = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.float64)
    target = source.copy()
    target[:, 0] *= 2.0
    target[:, 1] *= 3.0

    H = direct_linear_transform(target, source)

    expected = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
    assert np.allclose(H, expected)


def test_direct_linear_transform_perspective():
    # Test perspective transform
    source = np.array([[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])

    # Create a perspective transform by adding perspective components
    target = np.array(
        [
            [0, 0, 1],
            [2, 0.5, 1.5],  # Point moved with perspective
            [0.5, 2, 1.5],  # Point moved with perspective
            [3, 3, 2],  # Point moved with stronger perspective
        ]
    )

    H = direct_linear_transform(target, source)

    # Verify transformation works for all points
    for src, tgt in zip(source, target):
        transformed = H @ src
        # Normalize homogeneous coordinates for comparison
        transformed = transformed / transformed[2]
        tgt_normalized = tgt / tgt[2]
        assert np.allclose(transformed, tgt_normalized)

    # Verify H is not just affine (has perspective components)
    assert not np.allclose(H[2, :2], 0)  # Last row should have non-zero x,y terms


def test_direct_linear_transform_general():
    # Test general transform
    source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 2], [7, 8, 2]])
    target = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 2], [8, 9, 2]])

    H_t_s = direct_linear_transform(target, source)

    # Verify that H maps source points to target points
    for hxy_s, hxy_t in zip(source, target):
        hxy_t_mapped = H_t_s @ hxy_s
        hxy_t_mapped = hxy_t_mapped / hxy_t_mapped[2]
        hxy_t = hxy_t / hxy_t[2]
        assert np.allclose(hxy_t_mapped, hxy_t)


def test_direct_linear_transform_points_at_infinity():
    source = np.array([[1, 2, 0], [3, 4, 1], [5, 6, 1], [7, 8, 1]])  # w=0 in first point
    target = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 1], [8, 9, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target, source)

    source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1], [7, 8, 1]])
    target = np.array([[2, 3, 0], [4, 5, 1], [6, 7, 1], [8, 9, 1]])  # w=0 in first point

    with pytest.raises(AssertionError):
        direct_linear_transform(target, source)


def test_direct_linear_transform_insufficient_points():
    # Test with fewer than 4 points
    source_few = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])
    target_few = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target_few, source_few)


def test_direct_linear_transform_duplicate_points():
    # Test with duplicate points in source
    source_duplicate = np.array([[1, 2, 1], [1, 2, 1], [5, 6, 1], [7, 8, 1]])
    target_duplicate = np.array([[2, 3, 1], [2, 3, 1], [6, 7, 1], [8, 9, 1]])

    with pytest.raises(np.linalg.LinAlgError):
        direct_linear_transform(target_duplicate, source_duplicate)

    # Test with duplicate points in target
    source_duplicate = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1], [7, 8, 1]])
    target_duplicate = np.array([[2, 3, 1], [4, 5, 1], [4, 5, 1], [8, 9, 1]])

    with pytest.raises(np.linalg.LinAlgError):
        direct_linear_transform(target_duplicate, source_duplicate)


def test_direct_linear_transform_mismatch_rows():
    # Test with mismatched number of rows
    source_mismatch = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1], [7, 8, 1]])
    target_mismatch = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target_mismatch, source_mismatch)


def test_direct_linear_transform_invalid_input():
    # Test with NaN values in source
    source_nan = np.array([[1, 2, 1], [3, np.nan, 1], [5, 6, 1], [7, 8, 1]])
    target_nan = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 1], [8, 9, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target_nan, source_nan)

    # Test with NaN values in target
    source_nan = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1], [7, 8, 1]])
    target_nan = np.array([[2, 3, 1], [np.nan, 5, 1], [6, 7, 1], [8, 9, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target_nan, source_nan)

    # Test with inf values in source
    source_inf = np.array([[1, 2, 1], [3, np.inf, 1], [5, 6, 1], [7, 8, 1]])
    target_inf = np.array([[2, 3, 1], [4, 5, 1], [6, 7, 1], [8, 9, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target_inf, source_inf)

    # Test with inf values in target
    source_inf = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1], [7, 8, 1]])
    target_inf = np.array([[2, 3, 1], [np.inf, 5, 1], [6, 7, 1], [8, 9, 1]])

    with pytest.raises(AssertionError):
        direct_linear_transform(target_inf, source_inf)


def test_direct_linear_transform_singular_matrix():
    # Test with colinear points
    source_colinear = np.array([[0, 0, 1], [1, 1, 1], [2, 2, 1], [3, 3, 1]])
    target_colinear = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1]])

    with pytest.raises(np.linalg.LinAlgError):
        direct_linear_transform(target_colinear, source_colinear)

    # Test with near colinear points
    source_colinear = np.array([[0, 1e-16, 1], [1, 1, 1], [2, 2, 1], [3, 3, 1]])
    target_colinear = np.array([[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1]])

    with pytest.raises(np.linalg.LinAlgError):
        direct_linear_transform(target_colinear, source_colinear)

    # Test with points that result in singular matrix
    source_singular = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]])
    target_singular = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])

    with pytest.raises(np.linalg.LinAlgError):
        direct_linear_transform(target_singular, source_singular)


def test_apply_homography_identity():
    # Test with identity homography
    H_dest_source = np.eye(3)
    hxys_source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])

    hxys_dest = apply_homography(H_dest_source, hxys_source)

    # Points should remain unchanged
    assert np.allclose(hxys_dest, hxys_source)


def test_apply_homography_translation():
    # Test with translation homography
    H_dest_source = np.array([[1, 0, 2], [0, 1, 3], [0, 0, 1]])
    hxys_source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])

    hxys_dest = apply_homography(H_dest_source, hxys_source)

    # Points should be translated by [2, 3]
    expected = np.array([[3, 5, 1], [5, 7, 1], [7, 9, 1]])
    assert np.allclose(hxys_dest, expected)


def test_apply_homography_rotation():
    # Test with rotation homography (60 degrees)
    cos_60 = np.cos(np.pi / 3)
    sin_60 = np.sin(np.pi / 3)
    H_dest_source = np.array([[cos_60, -sin_60, 0], [sin_60, cos_60, 0], [0, 0, 1]])
    hxys_source = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])

    hxys_dest = apply_homography(H_dest_source, hxys_source)

    # Points should be rotated 60 degrees
    expected = np.array(
        [[cos_60, sin_60, 1], [-sin_60, cos_60, 1], [-cos_60, -sin_60, 1], [sin_60, -cos_60, 1]]
    )
    assert np.allclose(hxys_dest, expected)


def test_apply_homography_scale():
    # Test with scaling homography
    H_dest_source = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
    hxys_source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])

    hxys_dest = apply_homography(H_dest_source, hxys_source)

    # Points should be scaled by [2, 3]
    expected = np.array([[2, 6, 1], [6, 12, 1], [10, 18, 1]])
    assert np.allclose(hxys_dest, expected)


def test_apply_homography_perspective():
    # Test with perspective homography
    H_dest_source = np.array([[1, 0, 0], [0, 1, 0], [0.1, 0.2, 1]])
    hxys_source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])

    hxys_dest = apply_homography(H_dest_source, hxys_source)

    # Points should be transformed with perspective
    expected = np.array([[1 / 1.5, 2 / 1.5, 1], [3 / 2.1, 4 / 2.1, 1], [5 / 2.7, 6 / 2.7, 1]])
    assert np.allclose(hxys_dest, expected)


def test_apply_homography_single_point():
    # Test with a single point (1D array)
    H_dest_source = np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]])
    hxy_source = np.array([1, 2, 1])

    hxy_dest = apply_homography(H_dest_source, hxy_source)

    # Point should be scaled by [2, 3]
    expected = np.array([2, 6, 1])
    assert np.allclose(hxy_dest, expected)


def test_apply_homography_invalid_input():
    # Test with NaN values
    H_dest_source = np.eye(3)
    hxys_source = np.array([[1, 2, 1], [3, np.nan, 1], [5, 6, 1]])

    with pytest.raises(AssertionError):
        apply_homography(H_dest_source, hxys_source)

    # Test with inf values
    hxys_source = np.array([[1, 2, 1], [3, np.inf, 1], [5, 6, 1]])

    with pytest.raises(AssertionError):
        apply_homography(H_dest_source, hxys_source)

    # Test with w=0 entries
    hxys_source = np.array([[1, 2, 1], [3, 4, 0], [5, 6, 1]])

    with pytest.raises(AssertionError):
        apply_homography(H_dest_source, hxys_source)

    # Test with wrong shape for H
    H_dest_source = np.eye(2)
    hxys_source = np.array([[1, 2, 1], [3, 4, 1], [5, 6, 1]])

    with pytest.raises(AssertionError):
        apply_homography(H_dest_source, hxys_source)

    # Test with wrong shape for hxys
    H_dest_source = np.eye(3)
    hxys_source = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(AssertionError):
        apply_homography(H_dest_source, hxys_source)


def test_extract_matched_hxys_with_xys():
    # Test with basic 2D coordinates
    matched_a_b_idxs = [(0, 1), (2, 3), (4, 5)]
    xys_a = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    xys_b = np.array([[11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22]])

    matched_a_hxys, matched_b_hxys = extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Check that the correct points were extracted and converted to homogeneous coordinates
    expected_a_hxys = np.array([[1, 2, 1], [5, 6, 1], [9, 10, 1]])
    expected_b_hxys = np.array([[13, 14, 1], [17, 18, 1], [21, 22, 1]])

    assert np.array_equal(matched_a_hxys, expected_a_hxys)
    assert np.array_equal(matched_b_hxys, expected_b_hxys)


def test_extract_matched_hxys_with_extra_columns():
    # Test with coordinates that have extra columns
    matched_a_b_idxs = [(0, 1), (2, 3)]
    xys_a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    xys_b = np.array([[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24], [25, 26, 27, 28]])

    matched_a_hxys, matched_b_hxys = extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Check that only the first two columns were used and converted to homogeneous coordinates
    expected_a_hxys = np.array([[1, 2, 1], [9, 10, 1]])
    expected_b_hxys = np.array([[17, 18, 1], [25, 26, 1]])

    assert np.array_equal(matched_a_hxys, expected_a_hxys)
    assert np.array_equal(matched_b_hxys, expected_b_hxys)


def test_extract_matched_hxys_empty_matches():
    # Test with empty matches list
    matched_a_b_idxs = []
    xys_a = np.array([[1, 2], [3, 4], [5, 6]])
    xys_b = np.array([[7, 8], [9, 10], [11, 12]])

    matched_a_hxys, matched_b_hxys = extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Check that empty arrays with correct shape are returned
    assert matched_a_hxys.shape == (0, 3)
    assert matched_b_hxys.shape == (0, 3)


def test_extract_matched_hxys_single_match():
    # Test with a single match
    matched_a_b_idxs = [(1, 2)]
    xys_a = np.array([[1, 2], [3, 4], [5, 6]])
    xys_b = np.array([[7, 8], [9, 10], [11, 12]])

    matched_a_hxys, matched_b_hxys = extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Check that the correct points were extracted and converted to homogeneous coordinates
    expected_a_hxys = np.array([[3, 4, 1]])
    expected_b_hxys = np.array([[11, 12, 1]])

    assert np.array_equal(matched_a_hxys, expected_a_hxys)
    assert np.array_equal(matched_b_hxys, expected_b_hxys)


def test_extract_matched_hxys_invalid_input():
    # Test with empty xys_a
    matched_a_b_idxs = [(0, 1)]
    xys_a = np.array([]).reshape(0, 2)
    xys_b = np.array([[1, 2], [3, 4]])

    with pytest.raises(AssertionError):
        extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Test with empty xys_b
    xys_a = np.array([[1, 2], [3, 4]])
    xys_b = np.array([]).reshape(0, 2)

    with pytest.raises(AssertionError):
        extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Test with xys_a having less than 2 columns
    xys_a = np.array([[1], [2], [3]])
    xys_b = np.array([[1, 2], [3, 4]])

    with pytest.raises(AssertionError):
        extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Test with xys_b having less than 2 columns
    xys_a = np.array([[1, 2], [3, 4]])
    xys_b = np.array([[1], [2], [3]])

    with pytest.raises(AssertionError):
        extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)

    # Test with out of bounds idxs
    matched_a_b_idxs = [(0, 0), (2, 2)]
    xys_a = np.array([[1, 2], [3, 4]])
    xys_b = np.array([[7, 8]])

    with pytest.raises(AssertionError):
        extract_matched_hxys(matched_a_b_idxs, xys_a, xys_b)


if __name__ == "__main__":
    pytest.main([__file__])
