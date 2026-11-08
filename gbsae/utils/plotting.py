"""Heatmap interpolation helpers adapted from Overcomplete; see THIRD_PARTY_NOTICES.md."""
import cv2
import numpy as np
from gbsae.utils.arrays import to_npf32

def interpolate_cv2(img, target=(224, 224), interpolation=cv2.INTER_CUBIC):
    """Convert to a float32 channel-last image and resize with OpenCV.
    
    target is (width, height), matching cv2.resize, and defaults to (224,224).
    Supports a 2D image or a 3D image with a recognized channel axis; a leading
    singleton batch axis is removed by np_channel_last. interpolation defaults
    to cv2.INTER_CUBIC. Returns a NumPy array.
    """
    img = to_npf32(img)
    img = np_channel_last(img)
    assert img.ndim in (2, 3), f"Expected 2 or 3 dimensions, but got {img.shape}"
    return cv2.resize(img, target, interpolation=interpolation)


def np_channel_last(arr):
    """Convert to float32 and infer a channel-last image layout.
    
    A leading singleton batch axis is removed; 2D arrays gain a trailing
    singleton channel. For 3D arrays, a first dimension of size 1 or 3 is
    treated as channels and moved last. Other 3D arrays are assumed channel-last;
    ambiguous small image dimensions are not resolved from additional metadata.
    """
    arr = to_npf32(arr)

    # if batch dimension, take the first element
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # if grayscale, return as is with extra dimension
    if arr.ndim == 2:
        return arr[:, :, None]
    # assume channels dimension are 3 or 1
    if arr.shape[0] == 3 or arr.shape[0] == 1:
        return np.moveaxis(arr, 0, -1)

    assert arr.ndim == 3, f"Expected 3 dimensions, but got {arr.shape}"

    return arr
