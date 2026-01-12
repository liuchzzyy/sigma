import numpy as np
from sigma.utils.signal import fft_denoise2d


def test_fft_denoise2d():
    # Create a simple 2D image (10x10)
    image = np.zeros((10, 10))
    image[3:7, 3:7] = 1.0

    # Run denoising
    denoised = fft_denoise2d(image, keep_fraction=0.1)

    # Check output shape
    assert denoised.shape == image.shape
    # Check it returns a numpy array
    assert isinstance(denoised, np.ndarray)


if __name__ == "__main__":
    test_fft_denoise2d()
    print("Test passed!")
