import numpy as np
from scipy.ndimage import uniform_filter


def neighbour_averaging(dataset: np.ndarray) -> np.ndarray:
    """
    Apply a 3x3 mean filter (neighbour averaging) to the dataset.

    Parameters
    ----------
    dataset : np.ndarray
        Input dataset of shape (H, W, C).

    Returns
    -------
    np.ndarray
        Filtered dataset.
    """
    # Use scipy's uniform_filter which is optimized C code
    # size=(3, 3, 1) means 3x3 in spatial dims, 1 in spectral dim (channel independent)
    return uniform_filter(dataset, size=(3, 3, 1), mode="nearest")


def zscore(dataset: np.ndarray) -> np.ndarray:
    """
    Normalize the dataset using z-score (mean=0, std=1) per channel.

    Parameters
    ----------
    dataset : np.ndarray
        Input dataset of shape (H, W, C).

    Returns
    -------
    np.ndarray
        Normalized dataset.
    """
    new_dataset = dataset.copy()
    epsilon = 1e-10  # Prevent division by zero

    # Vectorized implementation for speed
    means = new_dataset.mean(axis=(0, 1), keepdims=True)
    stds = new_dataset.std(axis=(0, 1), keepdims=True)

    # Avoid division by zero where std is 0
    stds[stds == 0] = epsilon

    return (new_dataset - means) / stds


def range_normalization(dataset: np.ndarray) -> np.ndarray:
    """
    Normalize the dataset using Min-Max scaling to [0, 1].

    Parameters
    ----------
    dataset : np.ndarray
        Input dataset of shape (H, W, C) or any shape.

    Returns
    -------
    np.ndarray
        Normalized dataset.
    """
    min_val = np.min(dataset)
    max_val = np.max(dataset)
    if max_val - min_val == 0:
        return np.zeros_like(dataset)
    return (dataset - min_val) / (max_val - min_val)


def softmax(dataset: np.ndarray) -> np.ndarray:
    exp_dataset = np.exp(dataset)
    sum_exp = np.sum(exp_dataset, axis=2)
    sum_exp = np.expand_dims(sum_exp, axis=2)
    sum_exp = np.tile(sum_exp, (1, 1, dataset.shape[2]))
    new_dataset = exp_dataset / sum_exp
    return new_dataset
