import numpy as np
from scipy import fftpack


def fft_denoise2d(image: np.ndarray, keep_fraction: float) -> np.ndarray:
    """
    Apply FFT-based denoising to a 2D image.

    Args:
        image: 2D numpy array input image
        keep_fraction: Fraction of low frequencies to keep (0.0 to 1.0)

    Returns:
        Denoised image (real part of inverse FFT)
    """
    image_fft = fftpack.fft2(image)
    image_fft2 = image_fft.copy()

    # Set r and c to be the number of rows and columns of the array.
    r, c = image_fft2.shape

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    image_fft2[int(r * keep_fraction) : int(r * (1 - keep_fraction))] = 0

    # Similarly with the columns:
    image_fft2[:, int(c * keep_fraction) : int(c * (1 - keep_fraction))] = 0

    # Transformed the filtered image back to real space
    return fftpack.ifft2(image_fft2).real
