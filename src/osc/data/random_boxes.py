import numpy as np


def generate_random_boxes(
    area_min: float,
    area_max: float,
    ratio_min: float,
    ratio_max: float,
    num_boxes: int,
    seed: int,
    max_attempts=20,
) -> np.ndarray:
    """Random boxes in the ``(y0 x0 y1 x1)`` format.

    Args:
        area_min: lower bound for the relative area of the random crop
        area_max: upper bound for the relative area of the random crop
        ratio_min: lower bound for the aspect ratio of the random crop
        ratio_max: upper bound for the aspect ratio of the random crop
        num_boxes: number of boxes to generate
        seed: seed for the random generator
        max_attempts: how many times we should try to generate ``num_boxes`` before
            raising an exception. This is necessary because the box conditions may be
            too strict, and we have to run many samples.

    Returns:
        A ``np.float64`` array of shape ``[num_boxes, 4]``.
    """
    rng = np.random.default_rng(seed)
    boxes = []
    for _ in range(max_attempts):
        boxes.append(
            _generate_random_boxes(
                area_min, area_max, ratio_min, ratio_max, num_boxes, rng
            )
        )
        if sum((len(b) for b in boxes), 0) >= num_boxes:
            return np.concatenate(boxes, axis=0)[:num_boxes]
    raise RuntimeError(
        f"Failed to generate {num_boxes} with area({area_min:.1f}, {area_max:.1f}) "
        f"and ratio ({ratio_min:.1f}, {ratio_max:.1f}) after {max_attempts} attempts"
    )


def _generate_random_boxes(
    area_min: float,
    area_max: float,
    ratio_min: float,
    ratio_max: float,
    num_boxes: int,
    rng=None,
) -> np.ndarray:
    """Random boxes in the ``(y0 x0 y1 x1)`` format.

    It generates ``num_boxes`` boxes, but not all of them will be within the bounds
    of the image, therefore fewer boxes than requested may be returned.

    Args:
        area_min:
        area_max:
        ratio_min:
        ratio_max:
        num_boxes:
        rng:

    Returns:
        A ``np.float64`` array of shape ``[N, 4]``, where ``N <= num_boxes``.
    """
    rng = np.random.default_rng(rng)
    area = rng.uniform(area_min, area_max, size=(num_boxes,))
    ratio = rng.uniform(ratio_min, ratio_max, size=(num_boxes,))
    h = np.sqrt(area / ratio)
    w = np.sqrt(area * ratio)
    keep = (h <= 1) & (w <= 1)
    h = h[keep]
    w = w[keep]
    i = rng.uniform(0, 1 - h)
    j = rng.uniform(0, 1 - w)
    # [num_boxes', (y0 x0 y1 x1)]
    return np.stack([i, j, i + h, j + w], axis=-1)
