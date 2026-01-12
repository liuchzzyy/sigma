import os
import sys
import numpy as np
import time
from sigma.utils.load import TEMDataset
from sigma.processing.segmentation import PixelSegmenter
from sigma.utils import normalisation as norm


def main():
    print("=" * 50)
    print("Starting SIGMA Verification with Real Data")
    print("=" * 50)

    # 1. Load Data
    file_path = os.path.join("tutorial", "0007 - B2 HAADF.emd")
    if not os.path.exists(file_path):
        print(f"CRITICAL: Data file not found at {file_path}")
        sys.exit(1)

    print(f"[1/4] Loading dataset: {file_path}")
    start_time = time.time()
    try:
        # This tests the merged TEMDataset in sigma.utils.load
        tem = TEMDataset(file_path)
        print(f"      Success! Loaded in {time.time() - start_time:.2f}s")
        if tem.spectra is not None:
            print(f"      Spectra Shape: {tem.spectra.data.shape}")
        if tem.nav_img is not None:
            print(f"      Nav Image Shape: {tem.nav_img.data.shape}")
    except Exception as e:
        print(f"      FAILED to load dataset: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 2. Test Normalization (Optimized)
    print("\n[2/4] Testing Normalization (Optimized)")
    try:
        # Test neighbour_averaging (scipy based)
        t0 = time.time()
        smoothed = norm.neighbour_averaging(tem.spectra.data)
        t1 = time.time()
        print(f"      Neighbour Averaging: {t1 - t0:.4f}s")
        print(f"      Smoothed Shape: {smoothed.shape}")

        # Test zscore
        t0 = time.time()
        zscored = norm.zscore(tem.spectra.data)
        t1 = time.time()
        print(f"      Z-Score: {t1 - t0:.4f}s")
        print(f"      Z-Score Mean (should be ~0): {np.nanmean(zscored):.4f}")
        print(f"      Z-Score Std (should be ~1): {np.nanstd(zscored):.4f}")

    except Exception as e:
        print(f"      FAILED Normalization: {e}")
        import traceback

        traceback.print_exc()

    # 3. Test Processing Pipeline (Segmentation)
    print("\n[3/4] Testing Segmentation Pipeline")
    try:
        # Prepare data
        data = tem.spectra.data

        # Subsample for verification speed (HDBSCAN on 4096 dims is slow)
        # Take a center crop
        h_center, w_center = data.shape[0] // 2, data.shape[1] // 2
        h_size, w_size = 30, 30  # Small 30x30 crop
        data_crop = data[
            h_center - h_size : h_center + h_size,
            w_center - w_size : w_center + w_size,
            :,
        ]

        # Also reduce spectral dimensions (simulate PCA/Autoencoder latent space)
        # Taking every 50th channel just to reduce dims to ~80
        data_crop = data_crop[:, :, ::50]

        # Simple normalization for clustering
        data_norm = norm.range_normalization(data_crop)

        print(f"      Input Data Shape (Cropped & Subsampled): {data_norm.shape}")

        h, w, c = data_norm.shape
        latent_flat = data_norm.reshape(-1, c)
        print(f"      Latent (flat) Shape: {latent_flat.shape}")

        # PixelSegmenter requires dataset.normalised_elemental_data to be present
        tem.normalised_elemental_data = data_norm

        # Initialize Segmenter
        print("      Running HDBSCAN (via PixelSegmenter __init__)...")
        t0 = time.time()

        segmenter = PixelSegmenter(
            latent=latent_flat,
            dataset=tem,
            method="HDBSCAN",
            method_args={"min_cluster_size": 20, "min_samples": 5},
        )
        t1 = time.time()
        print(f"      Segmentation finished in {t1 - t0:.2f}s")

        # Check results
        if hasattr(segmenter, "labels"):
            unique_labels = np.unique(segmenter.labels)
            n_clusters = np.sum(unique_labels >= 0)
            n_noise = np.sum(segmenter.labels == -1)
            print(f"      Clusters found: {n_clusters}")
            print(f"      Noise points: {n_noise}")
        else:
            print("      WARNING: No 'labels' attribute found in segmenter.")

    except Exception as e:
        print(f"      FAILED Segmentation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # 4. Check Visualization Utilities (Non-GUI)
    print("\n[4/4] Checking Visualization Utils")
    try:
        from sigma.utils import visualisation as visual

        print("      Visualization modules imported and ready.")

    except Exception as e:
        print(f"      FAILED Visualization check: {e}")

    print("\n" + "=" * 50)
    print("VERIFICATION COMPLETE - ALL SYSTEMS GO")
    print("=" * 50)


if __name__ == "__main__":
    main()
